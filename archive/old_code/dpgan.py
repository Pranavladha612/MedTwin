import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),  # Increased width
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, data_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, data_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def train_dp_gan(
    data_tensor, epochs=600, batch_size=64, latent_dim=128
):  # Increased epochs and latent_dim
    data_dim = data_tensor.shape[1]
    netG = Generator(latent_dim, data_dim)
    netD = Discriminator(data_dim)

    # Slightly different learning rates to keep D from overpowering G
    optG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for i, (real_data,) in enumerate(loader):
            batch_sz = real_data.size(0)

            # --- Train Discriminator ---
            netD.zero_grad()
            # LABEL SMOOTHING: Use 0.9 instead of 1.0 for real data
            real_labels = torch.full((batch_sz, 1), 0.9)
            fake_labels = torch.full((batch_sz, 1), 0.1)

            out_real = netD(real_data)
            lossD_real = criterion(out_real, real_labels)

            noise = torch.randn(batch_sz, latent_dim)
            fake_data = netG(noise)
            out_fake = netD(fake_data.detach())
            lossD_fake = criterion(out_fake, fake_labels)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optD.step()

            # --- Train Generator ---
            netG.zero_grad()
            # We want G to make D think the fakes are REAL (1.0)
            lossG = criterion(netD(fake_data), torch.ones(batch_sz, 1))
            lossG.backward()
            optG.step()

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] | Loss D: {lossD.item():.4f} | Loss G: {lossG.item():.4f}"
            )

    return netG


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
