import torch
from torch import nn


def t2v(x, f, l0, l1):
    v0 = l0.forward(x)
    v1 = f(l1.forward(x))
    return torch.cat([v1, v0], 2)


class SinActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SinActivation, self).__init__()
        self.out_features = out_features
        self.l0 = nn.Linear(in_features, 1)
        self.l1 = nn.Linear(in_features, out_features-1)
        self.f = torch.sin

    def forward(self, x):
        return t2v(x, self.f, self.l0, self.l1)


class CosActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosActivation, self).__init__()
        self.out_features = out_features
        self.l0 = nn.Linear(in_features, 1)
        self.l1 = nn.Linear(in_features, out_features-1)
        self.f = torch.cos

    def forward(self, x):
        return t2v(x, self.f, self.l0, self.l1)
