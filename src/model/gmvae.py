import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

class EncoderNN(nn.Module):
    def __init__(self, latent_dim=2):
        super(EncoderNN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256,250),
            nn.ReLU(),
            nn.LayerNorm(250),
            nn.Linear(250,200),
        )
        self.fc_mu = nn.Linear(200, latent_dim)
        self.fc_logvar = nn.Linear(200, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        h = self.fc1(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

# ---- DIN DECODER ----
class DecoderNN(nn.Module):
    def __init__(self, latent_dim=2):
        super(DecoderNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512,28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# ---- DIN MoG-prior ----
class MoG(nn.Module):
    def __init__(self, D, K, uniform=False):
        super(MoG, self).__init__()
        print("MoG by JT.")
        self.uniform = uniform
        self.D = D
        self.K = K
        self.mu = nn.Parameter(torch.randn(1, self.K, self.D) * 3 + 0.5)
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

