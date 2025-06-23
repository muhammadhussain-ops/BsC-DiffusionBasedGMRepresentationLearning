# jgmvdrl_beta.py  (kør fra src/train)

import sys, math, random, pathlib, numpy as np, torch, matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

# --------------------------------------------------
#  Paths & device
# --------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "moguni"))          # hvor dine modeller ligger

from model.unet   import DenoisingNN               # samme definitioner
from model.encoder import EncoderNN
from model.MogBetaGMVDRL import MoG

SAVE_DIR = ROOT / "loadparameters"
SAVE_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Diffusion helpers (VP-parameterisering)
# --------------------------------------------------
def lmda(t: float) -> float:                         # λ_VP(t)
    return 1.0 - math.exp(-0.5 * t * t * 19.9 - 0.1 * t)

def forward_process(x0: torch.Tensor, t: float):
    """q(x_t | x_0) og dens score"""
    var = lmda(t)
    std = math.sqrt(var)
    mu  = math.exp(-0.25 * t * t * 19.9 - 0.05 * t) * x0
    noise = torch.randn_like(x0)
    xt  = mu + std * noise
    logp = -noise / std               # ∇_{x_t} log q(x_0 | x_t)
    return xt, logp

# --------------------------------------------------
# Træning
# --------------------------------------------------
def train(dmodel, emodel, mog_model,
          dopt, eopt, mopt,
          loader, epochs: int = 100):

    dmodel.train(); emodel.train(); mog_model.train()
    losses, gamma = [], 0.0

    for ep in range(1, epochs + 1):
        total = 0.0
        if ep == 10: gamma = 1e-8
        if ep == 20: gamma = 1e-6      # fast resten af træningen

        for x0, _ in loader:
            x0 = x0.view(x0.size(0), -1).to(device)

            dopt.zero_grad(); eopt.zero_grad(); mopt.zero_grad()

            t_val  = np.random.beta(7, 2) if ep <= 50 else random.uniform(1e-3, 1.0)
            t_time = torch.full((x0.size(0), 1), t_val, device=device)

            mu, logvar, z = emodel(x0)
            xt, logp      = forward_process(x0, t_val)
            slogp         = dmodel(xt, t_time, z)

            score_loss = (slogp - logp).pow(2).mean()

            var   = logvar.exp()
            log_q = -0.5 * ((z - mu).pow(2) / var + logvar + math.log(2 * math.pi)).sum(1)
            kl_mog = (log_q + mog_model(z, reduction="none")).sum()

            loss = lmda(t_val) * score_loss + gamma * kl_mog
            loss.backward()

            dopt.step(); mopt.step()
            if ep <= 50:
                eopt.step()

            total += loss.item()

        avg = total / len(loader)
        losses.append(avg)
        print(f"Epoch {ep:3d}/{epochs} | loss {avg:.4f}")

    return losses

# --------------------------------------------------
# Data (1 000 train-samples som tidligere)
# --------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

full_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
train_size = 10000
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# --------------------------------------------------
# Model-initialisering
# --------------------------------------------------
dmodel    = DenoisingNN().to(device)
emodel    = EncoderNN().to(device)
mog_model = MoG(D=2, K=10).to(device)

dopt = torch.optim.Adam(dmodel.parameters(), lr=3e-4)
eopt = torch.optim.Adam(emodel.parameters(), lr=3e-6)
mopt = torch.optim.Adam(mog_model.parameters(), lr=1e-4)

# --------------------------------------------------
# Træning
# --------------------------------------------------
losses = train(dmodel, emodel, mog_model, dopt, eopt, mopt, train_loader)

# --------------------------------------------------
# Gem vægte
# --------------------------------------------------
torch.save(dmodel.state_dict(), SAVE_DIR / "denoiser.pt")
torch.save(emodel.state_dict(), SAVE_DIR / "encoder.pt")
torch.save(mog_model.state_dict(), SAVE_DIR / "mog.pt")

# --------------------------------------------------
# Loss-plot
# --------------------------------------------------
plt.figure(figsize=(7, 4))
plt.plot(losses, marker="o", ms=3, lw=1)
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Training loss"); plt.grid(True)
plt.tight_layout(); plt.show()

# --------------------------------------------------
# Genindlæs modeller til sampling
# --------------------------------------------------
dmodel.load_state_dict(torch.load(SAVE_DIR / "denoiser.pt", map_location=device)); dmodel.eval()
emodel.load_state_dict(torch.load(SAVE_DIR / "encoder.pt",  map_location=device)); emodel.eval()
mog_model.load_state_dict(torch.load(SAVE_DIR / "mog.pt",   map_location=device)); mog_model.eval()

# --------------------------------------------------
# 10 × 10  rekonstruktioner via baglæns diffusion
# --------------------------------------------------
latent_dim, K = 2, 10
n_per, n_steps = 10, 100
delta = 1.0 / n_steps
t_grid = np.linspace(1.0, 0.0, n_steps + 1)[1:]

@torch.no_grad()
def backward(xt, t, z):
    t_time = torch.full((xt.size(0), 1), t, device=xt.device)
    beta_t = 0.1 + t * 19.9
    noise  = torch.randn_like(xt) * math.sqrt(delta * beta_t)
    return xt + beta_t * dmodel(xt, t_time, z) * delta - 0.5 * beta_t * xt * delta + noise

with torch.no_grad():
    mu_k  = mog_model.mu.detach().squeeze(0)
    std_k = torch.exp(0.5 * mog_model.log_var.detach().squeeze(0))

fig, axes = plt.subplots(K, n_per, figsize=(n_per * 1.0, K * 1.2))
fig.subplots_adjust(top=.92, bottom=.03, wspace=.02, hspace=.02)

for k in range(K):
    z_batch = mu_k[k] + std_k[k] * torch.randn(n_per, 2, device=device)
    xt, _   = forward_process(torch.randn(n_per, 28 * 28, device=device), 1.0)
    for t in t_grid:
        xt = backward(xt, float(t), z_batch)
    imgs = xt.view(-1, 28, 28).cpu().numpy()
    for j, img in enumerate(imgs):
        ax = axes[k, j]
        ax.imshow(img, cmap="gray"); ax.axis("off")
    y = axes[k, 0].get_position().y1 + .006
    fig.text(.5, y, f"Comp {k+1}: μ=({mu_k[k,0]:.2f}, {mu_k[k,1]:.2f})",
             ha="center", va="bottom", fontsize=9)
plt.show()

# --------------------------------------------------
# Latent-space scatter + MoG-konturer
# --------------------------------------------------
full_loader = DataLoader(Subset(mnist, range(10000)),
                         batch_size=1, shuffle=True)

zs, labels = [], []
with torch.no_grad():
    for x, y in full_loader:
        x = x.view(1, -1).to(device)
        mu, _, _ = emodel(x)
        zs.append(mu.cpu()); labels.append(y)
zs     = torch.cat(zs).numpy()
labels = torch.cat(labels).numpy()

means   = mog_model.mu.detach().squeeze(0).cpu().numpy()
vars_   = torch.exp(mog_model.log_var.detach().squeeze(0)).cpu().numpy()
weights = F.softmax(mog_model.w.detach(), 1).squeeze(0).cpu().numpy()

x_min, x_max = zs[:,0].min(), zs[:,0].max()
y_min, y_max = zs[:,1].min(), zs[:,1].max()
dx, dy = x_max - x_min, y_max - y_min
x_min -= .1 * dx;  x_max += .1 * dx
y_min -= .1 * dy;  y_max += .1 * dy
nb = 200
X, Y = np.meshgrid(np.linspace(x_min, x_max, nb),
                   np.linspace(y_min, y_max, nb))

def density(xx, yy, mu, var, w):
    pos  = np.stack([xx.ravel(), yy.ravel()], 1)
    dens = np.zeros(pos.shape[0])
    for m, v, pi in zip(mu, var, w):
        norm = pi / (2 * math.pi * math.sqrt(v[0] * v[1]))
        diff = pos - m
        dens += norm * np.exp(-0.5 * ((diff[:,0]**2)/v[0] + (diff[:,1]**2)/v[1]))
    return dens.reshape(xx.shape)

Z = density(X, Y, means, vars_, weights)

fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(zs[:,0], zs[:,1], c=labels, s=4, alpha=.6)
ax.legend(*sc.legend_elements(), title="Digit", loc="upper right", fontsize=8)
levels = np.linspace(Z.min(), Z.max(), 15)[1:]
ax.contour(X, Y, Z, levels=levels, cmap="Reds", linewidths=1.5)
ax.scatter(means[:,0], means[:,1], marker="x", s=80, c="black", lw=2)
ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
ax.set_title("Latent space + MoG contours", fontsize=11)
plt.tight_layout(); plt.show()
