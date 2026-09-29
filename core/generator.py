import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Gumbel-softmax temperature for categorical outputs (the CTGAN default).
GUMBEL_TAU = 0.2


class Generator(nn.Module):
    """Conditional generator.

    Input: noise plus a one-hot condition (the hospital site). Output layout
    follows `output_info`, a list of (kind, width) segments from
    DataIngestion: "sigmoid" segments are scaled numerics and missing flags in
    [0,1]; each "softmax" segment is one categorical column, produced with
    Gumbel-softmax so exactly one category is chosen.
    """

    def __init__(self, latent_dim, cond_dim, output_info):
        super(Generator, self).__init__()
        self.output_info = output_info
        data_dim = sum(width for _, width in output_info)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, data_dim),
        )

    def forward(self, z, cond, hard=False):
        """`hard=True` (for sampling) returns exact one-hot categories;
        training uses the soft, differentiable version."""
        logits = self.model(torch.cat([z, cond], dim=1))
        outputs, start = [], 0
        for kind, width in self.output_info:
            segment = logits[:, start:start + width]
            if kind == "sigmoid":
                outputs.append(torch.sigmoid(segment))
            else:
                outputs.append(F.gumbel_softmax(segment, tau=GUMBEL_TAU, hard=hard))
            start += width
        return torch.cat(outputs, dim=1)


class Critic(nn.Module):
    # In WGAN, the Discriminator is called a Critic. It scores a record
    # together with its condition, so it judges "is this a realistic patient
    # *for this site*".
    def __init__(self, data_dim, cond_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim + cond_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            # 🚨 NO SIGMOID! WGAN outputs a raw 'Realness' score, not a probability
        )

    def forward(self, x, cond):
        return self.model(torch.cat([x, cond], dim=1))


def compute_gradient_penalty(critic, real_samples, fake_samples, cond):
    """Calculates the gradient penalty to enforce the Lipschitz constraint
    (WGAN-GP), interpolating records while keeping their condition fixed."""
    alpha = torch.rand(real_samples.size(0), 1).expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = critic(interpolates, cond)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(real_samples.size(0), 1),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def train_wgan_gp(data_tensor, cond_tensor, output_info, epochs=600, batch_size=64, latent_dim=128):
    """Trains the conditional WGAN-GP with correlation loss.

    Named for what it is: no differential-privacy mechanism is applied here.
    """
    print("🧠 Initializing Conditional WGAN-GP Architecture with Correlation Loss...")
    data_dim, cond_dim = data_tensor.shape[1], cond_tensor.shape[1]

    netG = Generator(latent_dim, cond_dim, output_info)
    netC = Critic(data_dim, cond_dim)

    # WGAN-GP requires a learning rate of 0.0001 and specific betas
    optG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optC = torch.optim.Adam(netC.parameters(), lr=0.0001, betas=(0.0, 0.9))

    loader = DataLoader(
        TensorDataset(data_tensor, cond_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    CRITIC_ITERATIONS = 5  # The Critic learns 5x faster to guide the Generator accurately
    LAMBDA_GP = 10  # Standard penalty weight

    for epoch in range(epochs):
        for real_data, cond in loader:
            batch_sz = real_data.size(0)

            # ==============================
            # 1. Train the Critic
            # ==============================
            # Fake records are generated for the same sites as the real batch.
            for _ in range(CRITIC_ITERATIONS):
                fake_data = netG(torch.randn(batch_sz, latent_dim), cond).detach()

                gp = compute_gradient_penalty(netC, real_data, fake_data, cond)
                lossC = (
                    -torch.mean(netC(real_data, cond))
                    + torch.mean(netC(fake_data, cond))
                    + LAMBDA_GP * gp
                )

                optC.zero_grad()
                lossC.backward()
                optC.step()

            # ==============================
            # 2. Train the Generator
            # ==============================
            optG.zero_grad()
            fake_data = netG(torch.randn(batch_sz, latent_dim), cond)

            # WGAN Generator Loss
            lossG = -torch.mean(netC(fake_data, cond))

            # 🔬 Correlation Loss Penalty, including correlations with the site.
            # Correlation divides by each column's spread, so a column that is
            # constant within the batch gives NaN -- which then poisons every
            # weight for the rest of training. This happens by chance, e.g. a
            # batch of 64 real patients with no fbs=True (a 15% category).
            # Constant columns have no correlation to match, so they're skipped.
            real_full = torch.cat([real_data, cond], dim=1)
            fake_full = torch.cat([fake_data, cond], dim=1)
            keep = (real_full.std(dim=0) > 1e-4) & (fake_full.detach().std(dim=0) > 1e-4)
            real_corr = torch.corrcoef(real_full[:, keep].T)
            fake_corr = torch.corrcoef(fake_full[:, keep].T)
            corr_loss = F.mse_loss(fake_corr, real_corr)

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
