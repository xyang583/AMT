import sys
import os
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from engine import train_one_epoch, evaluate
import utils.deit_util as utils
from datasets import get_loaders
from utils.args import get_args_parser
from models import get_model
from utils.logger import CompleteLogger

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    args.seed = seed
    set_seed(seed)

    exp_name = os.path.basename(args.output_dir)
    args.output_dir = f"{args.output_dir}"
    output_dir = Path(args.output_dir)
    if utils.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "log.txt").open("a") as f:
            f.write(" ".join(sys.argv) + "\n\n")

        if args.logger and utils.is_main_process():
            logger = CompleteLogger(root=args.output_dir, phase="train")
        print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    data_loader_train, data_loader_val = get_loaders(args, num_tasks, global_rank)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nClsEpisode)
    print(f"Creating model: ProtoNet {args.arch}")
    model = get_model(args)
    model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=args.unused_params)
        model_without_ddp = model.module

    for name, param in model_without_ddp.named_parameters():
        flag = True
        for fil in ["norm", "weight", "bias", "cls_token", "pos_embed"]:
            if fil in name:
                flag = False
        if not flag:
            param.requires_grad = False

    if args.lr_scale:
        scale = 1 / 8
        linear_scaled_lr = args.lr * utils.get_world_size() * scale
        args.lr = linear_scaled_lr

    loss_scaler = NativeScaler() if args.fp16 else None

    optimizer = torch.optim.SGD(
        [p for p in model_without_ddp.parameters() if p.requires_grad],
        args.lr,
        momentum=args.momentum,
        weight_decay=0,
    )

    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.mixup > 0.:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    eval_criterion = torch.nn.CrossEntropyLoss()

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'])

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

        print(f'Resume from {args.resume} at epoch {args.start_epoch}.')

    test_stats = evaluate(data_loader_val, model, eval_criterion, device, args.seed+10000)
    print(f"Accuracy of the network on dataset_val: {test_stats['acc1']:.4f}%")
    if args.output_dir and utils.is_main_process():
        test_stats['epoch'] = -1
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(test_stats) + "\n\n")

    if args.eval:
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = test_stats['acc1']
    max_accuracy_epoch = -1
    max_ema_accuracy = 0
    max_ema_accuracy_epoch = -1

    for epoch in range(args.start_epoch, args.epochs):
        lr_scheduler.step(epoch)
        train_stats = train_one_epoch(
            data_loader_train, model, criterion, optimizer, epoch, device,
            loss_scaler, args.fp16, args.clip_grad, model_ema, mixup_fn,
            set_training_mode=True, args=args
        )

        test_stats = evaluate(data_loader_val, model, eval_criterion, device, args.seed+10000)

        log_stats = {'epoch': epoch,
                     **{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()}}

        if args.model_ema:
            print("=> evaluate ema model...")
            model_ema_test_stats = evaluate(data_loader_val, model_ema.ema, eval_criterion, device, args.seed + 10000)
            log_stats.update({f'model_ema_test_{k}': v for k, v in model_ema_test_stats.items()})

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth', output_dir / 'best.pth']
            for checkpoint_path in checkpoint_paths:
                state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema) if args.model_ema else None,
                    'args': args,
                }
                if loss_scaler is not None:
                    state_dict['scalar'] = loss_scaler.state_dict()
                if args.save_every_epoch:
                    checkpoint_path = checkpoint_path.with_name(
                        checkpoint_path.name.split(".")[0] + f"-{epoch}" + checkpoint_path.suffix)
                print(f"=> save checkpoint to {checkpoint_path}")
                utils.save_on_master(state_dict, checkpoint_path)

                if test_stats["acc1"] <= max_accuracy:
                    break

        max_accuracy_epoch = epoch if test_stats["acc1"] > max_accuracy else max_accuracy_epoch
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        log_stats['best_test_acc'] = max_accuracy
        log_stats['best_test_acc_epoch'] = max_accuracy_epoch
        print(f"\n\nAccuracy of the network on dataset_val: {test_stats['acc1']:.4f}%")
        print(f'Max accuracy: {max_accuracy:.2f}%, at epoch: {max_accuracy_epoch}\n\n')

        if args.model_ema:
            print(f"Accuracy of the EMA network on dataset_val: {model_ema_test_stats['acc1']:.4f}%")
            max_ema_accuracy_epoch = epoch if model_ema_test_stats["acc1"] > max_ema_accuracy else max_ema_accuracy_epoch
            max_ema_accuracy = max(max_ema_accuracy, model_ema_test_stats["acc1"])
            log_stats['best_ema_test_acc'] = max_ema_accuracy
            log_stats['best_ema_test_acc_epoch'] = max_ema_accuracy_epoch
            print(f'Max EMA accuracy: {max_ema_accuracy:.2f}%, at epoch: {max_ema_accuracy_epoch}\n\n')

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if utils.is_main_process():
        import tables
        tables.file._open_files.close_all()


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
