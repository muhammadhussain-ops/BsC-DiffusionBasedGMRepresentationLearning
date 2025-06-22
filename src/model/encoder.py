import torch
import torch.nn as nn


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