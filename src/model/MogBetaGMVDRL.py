import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random

class MoG(nn.Module):
    def __init__(self, D, K, uniform=False):
        super(MoG, self).__init__()
        print("MoG by JT.")
        self.uniform = uniform
        self.D = D
        self.K = K
        self.mu = nn.Parameter(torch.randn(1, self.K, self.D) * 1.5 + 0.5)
        self.log_var = nn.Parameter(-3. * torch.ones(1, self.K, self.D))
        if self.uniform:
            self.w = torch.zeros(1, self.K)
            self.w.requires_grad = False
        else:
            self.w = nn.Parameter(torch.zeros(1, self.K))
        self.PI = torch.from_numpy(np.asarray(np.pi))

    def log_diag_normal(self, x, mu, log_var, reduction="sum", dim=1):
        log_p = -0.5 * torch.log(2. * self.PI) \
                - 0.5 * log_var \
                - 0.5 * torch.exp(-log_var) * (x.unsqueeze(1) - mu)**2.
        return log_p

    def forward(self, x, reduction="mean"):
        log_pi = torch.log(F.softmax(self.w, 1))                # B x K
        log_N = torch.sum(self.log_diag_normal(x, self.mu, self.log_var), 2)  # B x K
        NLL = -torch.logsumexp(log_pi + log_N, 1)               # B
        if reduction == "sum":
            return NLL.sum()
        elif reduction == "mean":
            return NLL.mean()
        elif reduction == "none":
            return NLL
        else:
            raise ValueError("Either 'sum', 'mean' or 'none'.")