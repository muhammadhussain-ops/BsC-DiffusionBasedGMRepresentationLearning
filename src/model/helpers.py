import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import random



# ---------- Residual Block ---------- #
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act   = nn.SiLU()
    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(h + x)

# ---------- FiLM Block (concat z_emb + t_emb) ---------- #
class FiLMBlock(nn.Module):
    def __init__(self, channels, latent_dim, time_emb_dim, hidden_dim=128):
        super().__init__()
        self.z_embed = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.t_embed = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.film_proj = nn.Linear(2*hidden_dim, 2*channels)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)
    def forward(self, x, z, t_emb):
        z_h = self.z_embed(z)
        t_h = self.t_embed(t_emb)
        cond = torch.cat([z_h, t_h], dim=1)
        gamma_raw, beta = self.film_proj(cond).chunk(2, dim=1)
        gamma = (1 + gamma_raw).view(-1, x.size(1), 1, 1)
        beta  = beta.view(-1, x.size(1), 1, 1)
        return gamma * x + beta