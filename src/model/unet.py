import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import random

import sys, pathlib, os
sys.path.insert(0, str(pathlib.Path(os.getcwd()).parent))


from .helpers import ResidualBlock, FiLMBlock


latent_dim = 2
time_emb_dim = 128

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device) / half)
    args  = t * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)

class DenoisingNN(nn.Module):
    def __init__(self, latent_dim=2, time_emb_dim=128):
        super().__init__()
        chs = [32, 64, 128, 256, 256]

        # Downsampling path
        self.downs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1,       chs[0], 3, padding=1), nn.SiLU(), ResidualBlock(chs[0])),
            nn.Sequential(nn.Conv2d(chs[0],  chs[1], 3, stride=2, padding=1), nn.SiLU(), ResidualBlock(chs[1])),
            nn.Sequential(nn.Conv2d(chs[1],  chs[2], 3, stride=2, padding=1), nn.SiLU(), ResidualBlock(chs[2])),
            nn.Sequential(nn.Conv2d(chs[2],  chs[3], 3, stride=2, padding=1), nn.SiLU(), ResidualBlock(chs[3])),
            nn.Sequential(nn.Conv2d(chs[3],  chs[4], 3, stride=2, padding=1), nn.SiLU(), ResidualBlock(chs[4])),
        ])
        self.film_down = nn.ModuleList([FiLMBlock(c, latent_dim, time_emb_dim) for c in chs])

        # Upsampling path
        self.ups = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(chs[4],     chs[3], 3, stride=2, padding=1, output_padding=1), nn.SiLU(), ResidualBlock(chs[3])),
            nn.Sequential(nn.ConvTranspose2d(chs[3]*2,   chs[2], 3, stride=2, padding=1, output_padding=0), nn.SiLU(), ResidualBlock(chs[2])),
            nn.Sequential(nn.ConvTranspose2d(chs[2]*2,   chs[1], 3, stride=2, padding=1, output_padding=1), nn.SiLU(), ResidualBlock(chs[1])),
            nn.Sequential(nn.ConvTranspose2d(chs[1]*2,   chs[0], 3, stride=2, padding=1, output_padding=1), nn.SiLU(), ResidualBlock(chs[0])),
        ])
        self.film_up = nn.ModuleList([
            FiLMBlock(chs[3], latent_dim, time_emb_dim),
            FiLMBlock(chs[2], latent_dim, time_emb_dim),
            FiLMBlock(chs[1], latent_dim, time_emb_dim),
            FiLMBlock(chs[0], latent_dim, time_emb_dim),
        ])

        # Final output conv
        self.final = nn.Conv2d(chs[0]*2, 1, 1)

    def forward(self, xt, t, z):
        B = xt.size(0)
        t_emb = sinusoidal_embedding(t, time_emb_dim)
        h = xt.view(B, 1, 28, 28)

        skips = []
        for i, down in enumerate(self.downs):
            h = down(h)
            h = self.film_down[i](h, z, t_emb)
            skips.append(h)

        for i, up in enumerate(self.ups):
            h = up(h)
            h = self.film_up[i](h, z, t_emb)
            skip = skips[-(i+2)]
            h = torch.cat([h, skip], dim=1)

        out = self.final(h)
        return out.view(B, -1)