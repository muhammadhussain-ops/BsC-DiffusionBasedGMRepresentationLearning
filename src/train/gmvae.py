import os, sys, math, random, pathlib
import numpy as np, torch, matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model.gmvae import EncoderNN, DecoderNN, MoG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(enc, eopt, dec, dopt, mog, mopt, loader, epochs=50):
    enc.train(); dec.train(); mog.train(); losses = []
    for ep in range(epochs):
        tot = 0
        for x, _ in loader:
            x = x.view(x.size(0), -1).to(device)
            eopt.zero_grad(); dopt.zero_grad(); mopt.zero_grad()
            mu, logvar, z = enc(x)
            x_hat = dec(z)
            rec = F.binary_cross_entropy(x_hat, x, reduction="sum")
            var = torch.exp(logvar)
            log_q = -0.5 * ((math.log(2*math.pi) + logvar + (z - mu)**2 / var).sum(1))
            kl_mog = (log_q + mog(z, reduction="none")).sum()
            loss = rec + kl_mog
            loss.backward()
            eopt.step(); dopt.step(); mopt.step()
            tot += loss.item()
        losses.append(tot / len(loader.dataset))
        print(f"Epoch {ep+1}/{epochs}  avg loss/ex.: {losses[-1]:.4f}")
    return losses

def plot_reconstructions(enc, dec, loader, n_img=10):
    enc.eval(); dec.eval()
    with torch.no_grad():
        x, _ = next(iter(loader))
        x = x.view(x.size(0), -1).to(device)
        _, _, z = enc(x)
        x_hat = dec(z)
        x = x.cpu().view(-1, 1, 28, 28)
        x_hat = x_hat.cpu().view(-1, 1, 28, 28)
        plt.figure(figsize=(n_img*2, 4))
        for i in range(n_img):
            plt.subplot(2, n_img, i+1); plt.imshow(x[i].squeeze(), cmap="gray"); plt.axis("off")
            plt.subplot(2, n_img, n_img+i+1); plt.imshow(x_hat[i].squeeze(), cmap="gray"); plt.axis("off")
        plt.tight_layout(); plt.show()

def plot_latent_space(enc, mog, loader):
    enc.eval(); mog.eval(); zs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.view(x.size(0), -1).to(device)
            mu, _, _ = enc(x)
            zs.append(mu.cpu()); labels.append(y)
    zs = torch.cat(zs, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    comp_means = mog.mu.detach().squeeze(0).cpu().numpy()
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(zs[:, 0], zs[:, 1], c=labels, alpha=0.6, s=5)
    plt.scatter(comp_means[:, 0], comp_means[:, 1], marker='x', s=100, c='black', lw=2, label='MoG means')
    plt.legend(*sc.legend_elements(), title="Digit")
    plt.title('Latent space with MoG-ComponentMeans')
    plt.xlabel('z₁'); plt.ylabel('z₂')
    plt.grid(False); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parents[1]
    save_dir = root / "loadparameters"; save_dir.mkdir(exist_ok=True)

    transform = transforms.ToTensor()
    ds = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_len = int(0.8 * len(ds)); test_len = len(ds) - train_len
    train_ds, test_ds = random_split(ds, [train_len, test_len])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

    enc, dec, mog = EncoderNN(2).to(device), DecoderNN(2).to(device), MoG(2, 10).to(device)
    eopt = torch.optim.Adam(enc.parameters(), 1e-4)
    dopt = torch.optim.Adam(dec.parameters(), 1e-4)
    mopt = torch.optim.Adam(mog.parameters(), 1e-3)

    losses = train(enc, eopt, dec, dopt, mog, mopt, train_loader)

    torch.save(enc.state_dict(), save_dir / "Ae.pt")
    torch.save(dec.state_dict(), save_dir / "Ad.pt")
    torch.save(mog.state_dict(), save_dir / "Am.pt")

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(losses)+1), losses, marker="o")
    plt.title("Average Loss per Example vs. Epoch")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True); plt.show()

    plot_reconstructions(enc, dec, test_loader, n_img=10)
    plot_latent_space(enc, mog, test_loader)
