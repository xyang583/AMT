import math
import sys
import warnings
from typing import Iterable, Optional
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils.deit_util as utils
import utils.dist_utils as dist_utils
from utils import AverageMeter, to_device
from adsv import adv_attack, ADSV

def train_one_epoch(data_loader: Iterable,
                    model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    device: torch.device,
                    loss_scaler = None,
                    fp16: bool = False,
                    max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None,
                    mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    args=None,
                    ):

    global_step = epoch * len(data_loader)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('n_ways', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('n_imgs', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    model.train(set_training_mode)
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        batch = to_device(batch, device)
        if args.need_global_label:
            SupportTensor, SupportLabel, x, y = batch
            SupportTensor, SupportLabel, x, y = SupportTensor.squeeze(0), SupportLabel.squeeze(0), x.squeeze(0), y.squeeze(0)

        if mixup_fn is not None:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=torch.max(y).item()+1)
            x, y = mixup_fn(x, y)

        if args.pgd_method == "standard":
            with torch.cuda.amp.autocast(fp16):
                output = model(SupportTensor, SupportLabel, x)

            loss = criterion(output, y)

        elif args.pgd_method == "adsv":
            proxy = deepcopy(model)
            proxy_opt = torch.optim.SGD(proxy.parameters(), lr=args.awp_gamma)
            adsv_adversary = ADSV(model=model, proxy=proxy, proxy_optim=proxy_opt, gamma=args.awp_gamma)
            with torch.cuda.amp.autocast(fp16):
                supp_f = model(SupportTensor, forward_support=True)
            data_query_adv = adv_attack(model, optimizer, criterion, x, y, supp_f, SupportLabel, args, fp16=fp16)
            awp = adsv_adversary.calc_awp(supp_f.detach(), SupportLabel.detach(), inputs_adv=data_query_adv.detach(),
                                          inputs_clean=x.detach(), targets=y.detach(), beta=args.beta, fp16=fp16)
            adsv_adversary.perturb(awp)
            beta = args.beta
            batch_size = x.shape[0]
            criterion_kl = nn.KLDivLoss(size_average=False)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(fp16):
                query_adv_pred_logits = model(supp_f, SupportLabel, data_query_adv, forward_query=True)
                query_pred_logits = model(supp_f, SupportLabel, x, forward_query=True)

            loss_natural = criterion(query_pred_logits, y)
            loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(query_adv_pred_logits, dim=-1),
                                                            F.softmax(query_pred_logits, dim=-1))
            loss = loss_natural + beta * loss_robust
        else:
            raise ValueError(args.pgd_method)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        optimizer.zero_grad()
        if fp16:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()
        if "adsv" in args.pgd_method:
            adsv_adversary.restore(awp)
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)
        metric_logger.update(n_ways=SupportLabel.max()+1)
        metric_logger.update(n_imgs=SupportTensor.shape[0] + x.shape[0])

        if utils.is_main_process() and global_step % print_freq == 0:
            log_stats = {'lr_step': lr, 'loss_train_step': loss_value, 'step': global_step}
            print(log_stats)

        global_step += 1

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(data_loaders, model, criterion, device, seed=None, ep=None, args=None):
    if isinstance(data_loaders, dict):
        test_stats_lst = {}
        test_stats_glb = {}
        for j, (source, data_loader) in enumerate(data_loaders.items()):
            print(f'* Evaluating {source}:')
            seed_j = seed + j if seed else None
            test_stats = _evaluate(data_loader, model, criterion, device, seed_j)
            test_stats_lst[source] = test_stats
            test_stats_glb[source] = test_stats['acc1']

        for k in test_stats_lst[source].keys():
            test_stats_glb[k] = torch.tensor([test_stats[k] for test_stats in test_stats_lst.values()]).mean().item()
        return test_stats_glb
    elif isinstance(data_loaders, torch.utils.data.DataLoader):
        return _evaluate(data_loaders, model, criterion, device, seed, ep)
    else:
        warnings.warn(f'The structure of {data_loaders} is not recognizable.')
        return _evaluate(data_loaders, model, criterion, device, seed, ep)


@torch.no_grad()
def _evaluate(data_loader, model, criterion, device, seed=None, ep=None):
    task_log = []
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('n_ways', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('n_imgs', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('acc1', utils.SmoothedValue(window_size=len(data_loader.dataset)))
    metric_logger.add_meter('acc5', utils.SmoothedValue(window_size=len(data_loader.dataset)))
    header = 'Test:'
    model.eval()

    if seed is not None:
        data_loader.generator.manual_seed(seed)

    for ii, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        model_copy = deepcopy(model)
        if ep is not None:
            if ii > ep:
                break

        batch = to_device(batch, device)
        SupportTensor, SupportLabel, x, y = batch
        SupportTensor, SupportLabel, x, y = SupportTensor.squeeze(0), SupportLabel.squeeze(0), x.squeeze(0), y.squeeze(0)
        with torch.cuda.amp.autocast():
            output, _ = model_copy(SupportTensor, SupportLabel, x)
        loss = criterion(output, y)
        acc1, acc5 = accuracy(output, y, topk=(1, 5))
        batch_size = 1
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.update(n_ways=SupportLabel.max()+1)
        metric_logger.update(n_imgs=SupportTensor.shape[0] + x.shape[0])

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    ret_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    ret_dict['acc_std'] = metric_logger.meters['acc1'].std
    return ret_dict