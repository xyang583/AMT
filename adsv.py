from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np

EPS = 1E-20

def clamp(x, lower_limit, upper_limit):
    return torch.max(torch.min(x, upper_limit), lower_limit)

def adv_jitter(input_image, pyramid_aug_params, pyramid_scale, lower_limit, upper_limit):
    for layer, aug_params in enumerate(pyramid_aug_params):
        local_batch_size, c, num_patches, num_patches_1 = aug_params.shape
        local_batch_size_1, c, h, w = input_image.shape
        patch_sidelength = h // num_patches
        n, c, h, w = input_image.shape
        assert c == 3
        gh, gw = h // patch_sidelength, w // patch_sidelength
        fh, fw = patch_sidelength, patch_sidelength
        x = x.view(n, c, gh, fh, gw, fw)
        grid = x.permute(0, 1, 2, 4, 3, 5)
        jittered_grid = grid + pyramid_scale[layer] * aug_params.unsqueeze(-1).unsqueeze(-1)
        n, c, gh, gw, fh, fw = jittered_grid.shape
        x = x.permute(0, 1, 2, 4, 3, 5)
        input_image = x.view(n, c, gh * fh, gw * fw)
        input_image = clamp(input_image, lower_limit, upper_limit)
    return input_image

def adv_attack(model, optimizer, criterion, x, y, supp_f, SupportLabel, args, fp16):
    mu = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1).cuda()
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1).cuda()
    upper_limit = ((1 - mu) / std).cuda()
    lower_limit = ((0 - mu) / std).cuda()
    epsilon_base = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    patch_sizes = [4, 8, 128]
    patch_scalars = [20, 10, 1]
    drop_rate = 1.0
    model.eval()
    epsilon = epsilon_base.cuda()
    pyramid_delta = []
    if args.delta_init == 'zero':
        for (layer, pyramid_size) in enumerate(patch_sizes):
            delta = torch.zeros((x.shape[0], 3, pyramid_size, pyramid_size)).cuda()
            delta.requires_grad = True
            pyramid_delta.append(delta)
    elif args.delta_init == 'random1':
        for (layer, pyramid_size) in enumerate(patch_sizes):
            delta = torch.zeros((x.shape[0], 3, pyramid_size, pyramid_size)).cuda()
            for i in range(len(epsilon)):
                delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
            delta.requires_grad = True
            pyramid_delta.append(delta)
    elif args.delta_init == 'random2':
        for (layer, pyramid_size) in enumerate(patch_sizes):
            delta = 0.001 * torch.randn(x.shape[0], 3, pyramid_size, pyramid_size).cuda()
            delta.requires_grad = True
            pyramid_delta.append(delta)
    else:
        raise NotImplementedError('Not valid init_mode %s' % args.delta_init)

    criterion_kl = nn.KLDivLoss(size_average=False)
    for _ in range(args.attack_iters):
        x_adv = adv_jitter(x, pyramid_delta, patch_scalars, lower_limit, upper_limit)
        with torch.cuda.amp.autocast(fp16):
            query_adv_pred_logits = model(supp_f, SupportLabel, x_adv, forward_query=True)
            query_pred_logits = model(supp_f, SupportLabel, x, forward_query=True)
        loss_kl = criterion_kl(F.log_softmax(query_adv_pred_logits, dim=-1),
                               F.softmax(query_pred_logits, dim=-1))
        pyramid_delta_grad = torch.autograd.grad(loss_kl, pyramid_delta)
        for layer in range(len(pyramid_delta)):
            pyramid_delta[layer].data = clamp(pyramid_delta[layer] + \
                                              alpha * torch.sign(pyramid_delta_grad[layer].detach()), -epsilon, epsilon)
    pyramid_delta = [delta.detach() for delta in pyramid_delta]
    model.train()
    beta = args.beta
    batch_size = x.shape[0]
    criterion_kl = nn.KLDivLoss(size_average=False)
    data_adv = adv_jitter(x, pyramid_delta, patch_scalars, lower_limit, upper_limit)
    return data_adv

@torch.no_grad()
def diff_in_weights(model, proxy, norm=True):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'lora' in old_k:
            diff_w = new_w - old_w
            if norm:
                diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
            else:
                diff_dict[old_k] = diff_w
    return diff_dict

def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])

class ADSV(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(ADSV, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, supp_f, SupportLabel, inputs_adv, inputs_clean, targets, beta, fp16):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        with torch.cuda.amp.autocast(fp16):
            clean_output = self.proxy(supp_f, SupportLabel, inputs_clean, forward_query=True)
            adv_output = self.proxy(supp_f, SupportLabel, inputs_adv, forward_query=True)
        loss_natural = F.cross_entropy(clean_output, targets)
        loss_robust = F.kl_div(F.log_softmax(adv_output, dim=1),
                               F.softmax(clean_output, dim=1), reduction='batchmean')
        loss = - 1.0 * (loss_natural + beta * loss_robust)
        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)