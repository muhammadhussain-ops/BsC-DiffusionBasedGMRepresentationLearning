import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import random
from typing import List
import sys
import pathlib
import os

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from model.unet import DenoisingNN
from model.encoder import EncoderNN

PATH_DIR = pathlib.Path(__file__).resolve().parents[1] / "loadparameters"
PATH_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sigma(t, smin=0.01, smax=50):
    return smin * (smax / smin) ** t


def lmda(t, bmax=20, bmin=0.1, type="VP"):
    if type == "VP":
        return torch.tensor(
            1 - np.exp(-0.5 * t ** 2 * (bmax - bmin) - t * bmin),
            dtype=torch.float32,
            device=device,
        )
    elif type == "VE":
        return sigma(t) ** 2
    elif type == "subVP":
        cum_beta = t * bmin + 0.5 * t ** 2 * (bmax - bmin)
        return (bmin + t * (bmax - bmin)) * (1 - np.exp(-cum_beta))
    else:
        raise ValueError(f"Unknown type {type}")


def forward_process(x0, t, bmax=20, bmin=0.1, type="VP"):
    if type == "VE":
        var = torch.tensor(sigma(t) ** 2 - sigma(0) ** 2, dtype=torch.float32)
        noise = torch.randn_like(x0)
        std = torch.sqrt(var)
        xt = x0 + std * noise
        logp = -noise / std
    elif type == "VP":
        var = torch.tensor(
            1 - np.exp(-0.5 * t ** 2 * (bmax - bmin) - t * bmin),
            dtype=torch.float32,
            device=x0.device,
        )
        noise = torch.randn_like(x0)
        std = torch.sqrt(var)
        mu = (
            torch.tensor(
                np.exp(-0.25 * t ** 2 * (bmax - bmin) - 0.5 * t * bmin),
                dtype=torch.float32,
                device=x0.device,
            )
            * x0
        )
        xt = mu + std * noise
        logp = -noise / std
    return xt, logp


def kl_divergence(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def norml1(x):
    return torch.sum(torch.abs(x)) / x.size(0)


def train(dmodel, doptimizer, emodel, eoptimizer, train_loader, num_steps=100):
    dmodel.train()
    emodel.train()
    avg_losses = []
    gamma = 10 ** -3
    for epoch in range(1, num_steps + 1):
        total_loss = 0.0
        for x0, _ in train_loader:
            x0 = x0.view(x0.size(0), -1).to(device)
            doptimizer.zero_grad()
            eoptimizer.zero_grad()
            t_val = random.uniform(1e-3, 1)
            t_time = torch.full((x0.size(0), 1), t_val, device=device)
            mu, logvar, z = emodel(x0)
            xt, logp = forward_process(x0, t_val, type="VP")
            slogp = dmodel(xt, t_time, z) 
            score = torch.mean((slogp - logp) ** 2)
            kl = kl_divergence(mu, logvar)
            loss = lmda(t_val, type="VP") * score + gamma * kl
            loss.backward()
            doptimizer.step()
            eoptimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(train_loader)
        avg_losses.append(avg)
        print(f"Epoch {epoch}/{num_steps}  Loss: {avg:.4f}")
    return avg_losses


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

full_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
train_size = 1000
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

dmodel = DenoisingNN().to(device)
emodel = EncoderNN().to(device)

doptimizer = torch.optim.Adam(dmodel.parameters(), lr=3e-4)
eoptimizer = torch.optim.Adam(emodel.parameters(), lr=3e-6)

avg_losses = train(dmodel, doptimizer, emodel, eoptimizer, train_loader=train_loader)

torch.save(dmodel.state_dict(), PATH_DIR / "denoiserVDRLuni.pt")
torch.save(emodel.state_dict(), PATH_DIR / "encoderVDRLuni.pt")

plt.figure(figsize=(8, 5))
plt.plot(avg_losses, marker="o")
plt.title("Average Loss vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.show()
