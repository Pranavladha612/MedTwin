import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, data_dim),
            nn.Sigmoid(),  # Keeps features scaled between 0 and 1
        )

    def forward(self, z):
        return self.model(z)


class Critic(nn.Module):
    # In WGAN, the Discriminator is called a Critic
    def __init__(self, data_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            # 🚨 NO SIGMOID! WGAN outputs a raw 'Realness' score, not a probability
        )

    def forward(self, x):
        return self.model(x)


def compute_gradient_penalty(critic, real_samples, fake_samples):
    """Calculates the gradient penalty to enforce the Lipschitz constraint (WGAN-GP)"""
    alpha = torch.rand(real_samples.size(0), 1)
    alpha = alpha.expand_as(real_samples)

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = critic(interpolates)

    fake = torch.ones(real_samples.size(0), 1)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_dp_gan(data_tensor, epochs=600, batch_size=64, latent_dim=128):
    print("🧠 Initializing Advanced WGAN-GP Architecture with Correlation Loss...")
    data_dim = data_tensor.shape[1]

    netG = Generator(latent_dim, data_dim)
    netC = Critic(data_dim)

    # WGAN-GP requires a learning rate of 0.0001 and specific betas
    optG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optC = torch.optim.Adam(netC.parameters(), lr=0.0001, betas=(0.0, 0.9))

    loader = DataLoader(
        TensorDataset(data_tensor), batch_size=batch_size, shuffle=True, drop_last=True
    )

    CRITIC_ITERATIONS = (
        5  # The Critic learns 5x faster to guide the Generator accurately
    )
    LAMBDA_GP = 10  # Standard penalty weight

    for epoch in range(epochs):
        for i, (real_data,) in enumerate(loader):
            batch_sz = real_data.size(0)

            # ==============================
            # 1. Train the Critic
            # ==============================
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(batch_sz, latent_dim)
                fake_data = netG(
                    noise
                ).detach()  # Detach prevents Generator from training here

                real_validity = netC(real_data)
                fake_validity = netC(fake_data)

                # Gradient Penalty
                gp = compute_gradient_penalty(netC, real_data.data, fake_data.data)

                # WGAN Loss Math: Maximize distance between Real and Fake scores
                lossC = (
                    -torch.mean(real_validity)
                    + torch.mean(fake_validity)
                    + LAMBDA_GP * gp
                )

                optC.zero_grad()
                lossC.backward()
                optC.step()

            # ==============================
            # 2. Train the Generator
            # ==============================
            optG.zero_grad()
            noise = torch.randn(batch_sz, latent_dim)
            fake_data = netG(noise)

            # WGAN Generator Loss
            lossG = -torch.mean(netC(fake_data))

            # 🔬 ADVANCED: Correlation Loss Penalty
            # Correlation divides by each column's spread, so a column that is
            # constant within the batch gives NaN -- which then poisons every
            # weight for the rest of training. This happens by chance, e.g. a
            # batch of 64 real patients with no fbs=True (a 15% category).
            # Constant columns have no correlation to match, so they're skipped.
            keep = (real_data.std(dim=0) > 1e-4) & (fake_data.detach().std(dim=0) > 1e-4)
            real_corr = torch.corrcoef(real_data[:, keep].T)
            fake_corr = torch.corrcoef(fake_data[:, keep].T)

            # Penalize the Generator if its feature correlations don't match reality
            corr_loss = nn.functional.mse_loss(fake_corr, real_corr)

            # Combine losses (Generator Loss + Correlation Penalty)
            total_lossG = lossG + (corr_loss * 2.0)

            total_lossG.backward()
            optG.step()

        # Print progress
        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] | Critic Loss: {lossC.item():.4f} | Gen Loss: {total_lossG.item():.4f} | Corr Error: {corr_loss.item():.4f}"
            )

    return netG


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
