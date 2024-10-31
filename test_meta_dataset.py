import os
import numpy as np
import time
import random
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path
from tabulate import tabulate
from datetime import datetime
import yaml

from engine import evaluate
import utils.deit_util as utils
from datasets import get_sets
from utils.args import get_args_parser
from models import get_model
from datasets import get_loaders
from utils.logger import CompleteLogger

def get_test_loader(args):
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    if args.distributed:
        _, data_loader_val = get_loaders(args, num_tasks, global_rank)
    else:
        _, _, dataset_val = get_sets(args)

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        generator = torch.Generator()
        generator.manual_seed(args.seed + 10000)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            generator=generator
        )
    return data_loader_val


def main(args):
    utils.init_distributed_mode(args)

    args.eval = True
    args.dataset = 'meta_dataset'

    print(args)
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    args.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print(f"Creating model: {args.deploy} {args.arch}")

    model = get_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=args.unused_params)
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if args.model_ema:
            msg = model_without_ddp.load_state_dict(checkpoint['model_ema'], strict=True)
        else:
            msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        print(f'\n\nLoad ckpt from {args.resume} with message: {msg}\n\n')

    criterion = torch.nn.CrossEntropyLoss()
    datasets = args.test_sources
    var_accs = {}

    for domain in datasets:
        args.test_sources = [domain]
        data_loader_val = get_test_loader(args)

        best_acc = 0
        with open(f'configs/{domain}.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            candidate_pool = config['candidate_pool']
            search_scale = config['search_scale']
            search_step = config['search_step']
        beta_list = [i * (search_scale[0] - 0.1) / search_step[0] + 0.1 for i in range(search_step[0])]
        gamma_list = [i * search_scale[1] / search_step[1] for i in range(search_step[1]+1)]

        for beta in beta_list:
            for gamma in gamma_list:
                model_without_ddp.args.blend_beta = beta
                model_without_ddp.args.blend_gamma = gamma
                model_without_ddp.lr = 0
                test_stats = evaluate(data_loader_val, model, criterion, device, seed=1234, ep=5, args=args)
                acc = test_stats['acc1']
                print(f"blend beta={beta}, blend gamma={gamma}: acc1 = {acc}")
                if acc > best_acc:
                    best_acc = acc
                    best_beta = beta
                    best_gamma = gamma
        model_without_ddp.args.blend_beta = best_beta
        model_without_ddp.args.blend_gamma = best_gamma

        best_acc = 0
        best_trim = None
        for trim in candidate_pool:
            if "name" in trim:
                model_without_ddp.svd_trim_name = trim['name']
                model_without_ddp.svd_trim_layer = trim['layer']
                model_without_ddp.svd_trim_p = trim['p']
                model_without_ddp.svd_trim_names = None
                model_without_ddp.svd_trim_ps = None
            else:
                model_without_ddp.svd_trim_names = trim['names']
                model_without_ddp.svd_trim_layer = trim['layer']
                model_without_ddp.svd_trim_ps = trim['ps']
                model_without_ddp.svd_trim_name = None
                model_without_ddp.svd_trim_p = None
            model_without_ddp.lr = 0
            test_stats = evaluate(data_loader_val, model, criterion, device, seed=1234, ep=5, args=args)
            acc = test_stats['acc1']
            if acc > best_acc:
                best_acc = acc
                best_trim = trim
        if "name" in best_trim:
            model_without_ddp.svd_trim_name = best_trim['name']
            model_without_ddp.svd_trim_layer = best_trim['layer']
            model_without_ddp.svd_trim_p = best_trim['p']
            model_without_ddp.svd_trim_names = None
            model_without_ddp.svd_trim_ps = None
        else:
            model_without_ddp.svd_trim_names = best_trim['names']
            model_without_ddp.svd_trim_layer = best_trim['layer']
            model_without_ddp.svd_trim_ps = best_trim['ps']
            model_without_ddp.svd_trim_name = None
            model_without_ddp.svd_trim_p = None

        best_lr = args.ada_lr
        if 'finetune' in args.deploy:
            print("Start selecting the best lr...\n")
            best_acc = 0
            lr_range = [0, 0.0001, 0.001, 0.01]
            for lr in lr_range:
                print(f"\n=> use lr: {lr}\n")
                model_without_ddp.lr = lr
                test_stats = evaluate(data_loader_val, model, criterion, device, seed=1234, ep=5, args=args)
                acc = test_stats['acc1']
                print(f"*lr = {lr}: acc1 = {acc}")
                if acc > best_acc:
                    best_acc = acc
                    best_lr = lr
            model_without_ddp.lr = best_lr

        data_loader_val.generator.manual_seed(args.seed + 10000)
        test_stats = evaluate(data_loader_val, model, criterion, device, args=args)
        var_accs[domain] = (test_stats['acc1'], test_stats['acc_std'], best_lr)

        print(f"{domain}: acc1 on {len(data_loader_val.dataset)} test images: {test_stats['acc1']:.1f}%")

        if args.output_dir and utils.is_main_process():
            test_stats['domain'] = args.test_sources[0]
            test_stats['lr'] = best_lr
            with (output_dir / args.test_log_name).open("a") as f:
                f.write(json.dumps(test_stats) + "\n")

    if utils.is_main_process():
        rows = []
        iid_names = ["ilsvrc_2012", "omniglot", "aircraft", "cu_birds", "dtd", "quickdraw", "fungi", "vgg_flower"]
        ood_names = ["traffic_sign", "mscoco"]
        total_acc, total_conf = [], []
        woin_acc, woin_conf = [], []
        iid_acc, iid_conf = [], []
        ood_acc, ood_conf = [], []
        for dataset_name in datasets:
            row = [dataset_name]
            acc, std, lr = var_accs[dataset_name]
            conf = (1.96 * std) / np.sqrt(len(data_loader_val.dataset))
            row.append(f"{acc:0.2f} +- {conf:0.2f}")
            row.append(f"{lr}")
            rows.append(row)

            total_acc.append(acc)
            total_conf.append(conf)
            if "ilsvrc" not in dataset_name:
                woin_acc.append(acc)
                woin_conf.append(conf)
            if dataset_name in iid_names:
                iid_acc.append(acc)
                iid_conf.append(conf)
            elif dataset_name in ood_names:
                ood_acc.append(acc)
                ood_conf.append(conf)
            else:
                raise ValueError
        rows.append(["", "", ""])
        rows.append(["Total avg", f"{np.mean(total_acc):0.2f} +- {np.mean(total_conf):0.2f}", ""])
        rows.append(["Avg (w/o IN)", f"{np.mean(woin_acc):0.2f} +- {np.mean(woin_conf):0.2f}", ""])
        rows.append(["IID avg", f"{np.mean(iid_acc):0.2f} +- {np.mean(iid_conf):0.2f}", ""])
        rows.append(["OOD avg", f"{np.mean(ood_acc):0.2f} +- {np.mean(ood_conf):0.2f}", ""])

        table = tabulate(rows, headers=['Domain', args.arch, 'lr'], floatfmt=".2e")
        print(table)
        print("\n")

        if args.output_dir:
            with (output_dir / args.test_log_name).open("a") as f:
                f.write(table+"\n\n")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    args.train_tag = 'pt' if args.resume == '' else 'ep'
    args.train_tag += f'_step{args.ada_steps}_lr{args.ada_lr}_prob{args.aug_prob}'
    args.test_model_name = os.path.splitext(os.path.basename(args.resume))[0]
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.test_log_name = f"log_test_{args.deploy}_{args.train_tag}_{args.test_model_name}_{timestamp}.txt"
    args.output_dir = os.path.join(args.output_dir, f"log_test_{args.deploy}_{args.train_tag}_{args.test_model_name}_{timestamp}")

    if utils.is_main_process():
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        import sys
        with (output_dir / args.test_log_name).open("a") as f:
            f.write(" ".join(sys.argv) + "\n")

    if args.logger and utils.is_main_process():
        logger = CompleteLogger(root=args.output_dir, phase="test")
        print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    main(args)
