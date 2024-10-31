import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import trunc_normal_, DiffAugment
from timm.utils import accuracy
import numpy as np
from copy import deepcopy

class ProtoNetMerge(nn.Module):
    def __init__(self, backbone=None, args=None):
        super().__init__()
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)
        if backbone is not None:
            self.backbone = backbone
        self.args = args
        self.svd_trim_layer = None
        self.svd_trim_layers = None
        self.svd_trim_name = None
        self.svd_trim_names = None
        self.svd_trim_p = None
        self.svd_trim_ps = None

        state_dict = self.backbone.state_dict()
        self.backbone_state = deepcopy(state_dict)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)

        state_dict = self.backbone.state_dict()
        self.backbone_state = deepcopy(state_dict)

    def cos_classifier(self, w, f):
        f = F.normalize(f, p=2, dim=-1, eps=1e-12)
        w = F.normalize(w, p=2, dim=-1, eps=1e-12)

        cls_scores = f @ w.transpose(0, 1)
        cls_scores = self.scale_cls * (cls_scores + self.bias)
        return cls_scores

    def cos_classifier_nonparam(self, w, f):
        f = F.normalize(f, p=2, dim=-1, eps=1e-12)
        w = F.normalize(w, p=2, dim=-1, eps=1e-12)

        cls_scores = f @ w.transpose(0, 1)
        return cls_scores

    def forward(self, supp_x, supp_y=None, x=None, forward_support=False, forward_query=False, forward_router=False):
        self.backbone.load_state_dict(self.backbone_state, strict=True)

        if forward_support:
            return self.forward_support(supp_x)
        if forward_query:
            return self.forward_query(supp_x, supp_y, x)
        if forward_router:
            return self.forward_router(supp_x, supp_y)
        num_classes = supp_y.max() + 1

        with torch.cuda.amp.autocast(False):
            if self.args.specified_expert is not None:
                self.backbone.merge_specified_adapter(self.args.specified_expert)
                all_router_logits = []
            else:
                all_router_logits = self.forward_router(supp_x, supp_y)

            if self.svd_trim_name is not None:
                if self.svd_trim_ps is None:
                    self.backbone.set_svd_trim(self.svd_trim_name, self.svd_trim_layer, self.svd_trim_p)
                else:
                    for l, r in zip(self.svd_trim_layers, self.svd_trim_ps):
                        self.backbone.set_svd_trim(self.svd_trim_name, l, r, reinit=False)
            elif self.svd_trim_names is not None:
                for n, l, r in zip(self.svd_trim_names, self.svd_trim_layers, self.svd_trim_ps):
                    self.backbone.set_svd_trim(n, l, r, reinit=False)
            self.backbone.svd_trim()

        supp_f = self.backbone.forward(supp_x)

        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(0, 1)


        prototypes = torch.mm(supp_y_1hot.float(), supp_f)
        prototypes = prototypes / supp_y_1hot.sum(dim=-1, keepdim=True)

        feat = self.backbone.forward(x)
        logits = self.cos_classifier(prototypes, feat)

        self.backbone.unmerge_adapter()

        return logits, all_router_logits

    def single_step(self, supp_x, supp_y, z):
        criterion1 = nn.CrossEntropyLoss()

        nSupp, C, H, W = supp_x.shape
        num_classes = supp_y.max() + 1
        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(0, 1)
        supp_y = supp_y.view(-1)

        with torch.set_grad_enabled(False):
            supp_f = self.backbone.forward(supp_x)
            prototypes = torch.mm(supp_y_1hot.float(), supp_f)
            prototypes = prototypes / supp_y_1hot.sum(dim=-1, keepdim=True)


            feat = self.backbone.forward(z)


            logits = self.cos_classifier_nonparam(prototypes, feat)
            loss = criterion1(logits.view(nSupp, -1), supp_y)
            acc = accuracy(logits, supp_y)[0]

        return logits, acc.item(), loss.item()

    def forward_router(self, supp_x, supp_y):
        z = DiffAugment(supp_x, self.args.aug_types, self.args.aug_prob, detach=True)

        AB_loss_list = []
        AB_acc_list = []
        AB_logits_list = []
        for idx in range(len(self.backbone.peft_config.lora_moe_r)):
            adapter_name_moe = str(idx)
            self.backbone.merge_specified_adapter(adapter_name_moe)

            logits, acc, loss = self.single_step(supp_x, supp_y, z)
            AB_loss_list.append(loss)
            AB_acc_list.append(acc)
            AB_logits_list.append(logits)

            self.backbone.unmerge_adapter()

        beta = self.args.blend_beta
        gamma = self.args.blend_gamma
        router_logits = []
        for lg in AB_logits_list:
            num_classes = supp_y.max() + 1
            supp_one_hot = F.one_hot(supp_y, num_classes)
            yd = lg * supp_one_hot
            intra_cls_dis = torch.mm(supp_one_hot.transpose(0, 1).float(), yd)
            intra_cls_dis = intra_cls_dis / supp_one_hot.transpose(0, 1).sum(dim=-1, keepdim=True)
            intra_cls_dis = intra_cls_dis.sum() / num_classes

            ydd = lg * (1 - supp_one_hot) / (num_classes - 1)
            inter_cls_dis = torch.mm(supp_one_hot.transpose(0, 1).float(), ydd)
            inter_cls_dis = inter_cls_dis / supp_one_hot.transpose(0, 1).sum(dim=-1, keepdim=True)
            inter_cls_dis = inter_cls_dis.sum() / num_classes

            score = gamma * intra_cls_dis - (1-gamma)*inter_cls_dis
            router_logits.append(score)

        router_logits = torch.stack(router_logits)
        routing_weights_before = torch.exp((-1) * beta + beta * router_logits)
        routing_weights, selected_experts = torch.topk(routing_weights_before, self.backbone.peft_config.top_k[-1], dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(supp_x.dtype)
        self.backbone.merge_multi_adapter(routing_weights.view(-1), selected_experts.view(-1))
        return []

    def forward_support(self, supp_x):
        supp_f = self.backbone.forward(supp_x)
        return supp_f

    def forward_query(self, supp_f, supp_y, x):
        num_classes = supp_y.max() + 1

        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(0, 1)

        prototypes = torch.mm(supp_y_1hot.float(), supp_f)
        prototypes = prototypes / supp_y_1hot.sum(dim=-1, keepdim=True)

        feat = self.backbone.forward(x)
        logits = self.cos_classifier(prototypes, feat)

        return logits
