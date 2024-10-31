import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoNet(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)
        if backbone is not None:
            self.backbone = backbone

    def cos_classifier(self, w, f):
        f = F.normalize(f, p=2, dim=-1, eps=1e-12)
        w = F.normalize(w, p=2, dim=-1, eps=1e-12)

        cls_scores = f @ w.transpose(0, 1)
        cls_scores = self.scale_cls * (cls_scores + self.bias)
        return cls_scores

    def forward(self, supp_x, supp_y=None, x=None, forward_support=False, forward_query=False):
        if forward_support:
            return self.forward_support(supp_x)
        if forward_query:
            return self.forward_query(supp_x, supp_y, x)
        num_classes = supp_y.max() + 1
        supp_f = self.backbone.forward(supp_x)
        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(0, 1)
        prototypes = torch.mm(supp_y_1hot.float(), supp_f)
        prototypes = prototypes / supp_y_1hot.sum(dim=-1, keepdim=True)

        feat = self.backbone.forward(x)
        logits = self.cos_classifier(prototypes, feat)

        return logits

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
