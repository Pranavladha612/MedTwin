"""Compact reimplementation of TabDDPM (Kotelnikov et al., ICML 2023).

Not the authors' code (their repo isn't a pip package, and synthcity, which
bundles a port, doesn't install on this project's Python 3.9). It follows the
paper's design:

- numeric columns: QuantileTransformer to a normal distribution, then
  Gaussian diffusion (DDPM, epsilon-prediction, MSE loss);
- categorical columns: one-hot, multinomial diffusion (Hoogeboom et al. 2021),
  trained with the KL between true and predicted posteriors;
- one MLP denoiser over both, conditioned on a sinusoidal timestep embedding;
- cosine noise schedule.
"""

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer


def _cosine_alphas_bar(T, s=0.008):
    steps = np.arange(T + 1) / T
    f = np.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    betas = np.clip(1 - f[1:] / f[:-1], 0, 0.999)
    return betas


class _Denoiser(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, t_dim=128):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, hidden[0]))
        self.proj = nn.Linear(in_dim, hidden[0])
        layers = []
        for a, b in zip(hidden[:-1], hidden[1:]):
            layers += [nn.ReLU(), nn.Linear(a, b)]
        self.body = nn.Sequential(*layers, nn.ReLU(), nn.Linear(hidden[-1], out_dim))

    def forward(self, x, t):
        half = self.t_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        return self.body(self.proj(x) + self.t_mlp(emb))


class TabDDPM:
    def __init__(
        self,
        num_timesteps=1000,
        steps=8000,
        batch_size=256,
        lr=1e-3,
        hidden=(256, 512, 512, 256),
        seed=42,
    ):
        self.T = num_timesteps
        self.steps = steps
        self.batch_size = batch_size
        self.lr = lr
        self.hidden = hidden
        self.seed = seed

        betas = torch.tensor(_cosine_alphas_bar(num_timesteps), dtype=torch.float32)
        self.betas = betas
        self.alphas = 1 - betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar_prev = torch.cat([torch.ones(1), self.alphas_bar[:-1]])

    # ---------- preprocessing ----------
    def _encode(self, df):
        x_num = self.qt.transform(df[self.num_cols].values) if self.num_cols else np.zeros((len(df), 0))
        onehots = [
            F.one_hot(
                torch.tensor(pd.Categorical(df[c], categories=cats).codes, dtype=torch.long),
                len(cats),
            ).float()
            for c, cats in zip(self.cat_cols, self.categories)
        ]
        x_cat = torch.cat(onehots, dim=1) if onehots else torch.zeros(len(df), 0)
        return torch.tensor(x_num, dtype=torch.float32), x_cat

    # ---------- categorical diffusion helpers ----------
    def _split(self, x_cat):
        return torch.split(x_cat, self.cat_sizes, dim=1)

    def _cat_q_sample(self, x0_cat, t):
        ab = self.alphas_bar[t][:, None]
        out = []
        for block, K in zip(self._split(x0_cat), self.cat_sizes):
            probs = ab * block + (1 - ab) / K
            idx = torch.multinomial(probs, 1).squeeze(1)
            out.append(F.one_hot(idx, K).float())
        return torch.cat(out, dim=1)

    def _cat_posterior(self, x_t_cat, x0_probs, t):
        """q(x_{t-1} | x_t, x0) for each categorical block (x0 may be soft)."""
        a = self.alphas[t][:, None]
        ab_prev = self.alphas_bar_prev[t][:, None]
        post = []
        for xt, x0, K in zip(self._split(x_t_cat), self._split(x0_probs), self.cat_sizes):
            p = (a * xt + (1 - a) / K) * (ab_prev * x0 + (1 - ab_prev) / K)
            post.append(p / p.sum(dim=1, keepdim=True))
        return post

    def _cat_x0_probs(self, logits):
        return torch.cat([F.softmax(b, dim=1) for b in self._split(logits)], dim=1)

    # ---------- training ----------
    def fit(self, df, discrete_columns):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.columns = list(df.columns)
        self.cat_cols = [c for c in df.columns if c in discrete_columns]
        self.num_cols = [c for c in df.columns if c not in discrete_columns]
        self.categories = [sorted(df[c].unique().tolist()) for c in self.cat_cols]
        self.cat_sizes = [len(c) for c in self.categories]
        self.num_dtypes = {c: df[c].dtype for c in self.num_cols}

        self.qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(len(df) // 30, 1000), 10),
            random_state=self.seed,
        ).fit(df[self.num_cols].values)

        x_num, x_cat = self._encode(df)
        self.n_num = x_num.shape[1]
        dim = self.n_num + x_cat.shape[1]
        self.model = _Denoiser(dim, dim, list(self.hidden))
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        n = len(df)
        for step in range(self.steps):
            idx = torch.randint(0, n, (min(self.batch_size, n),))
            xn0, xc0 = x_num[idx], x_cat[idx]
            t = torch.randint(0, self.T, (len(idx),))

            ab = self.alphas_bar[t][:, None]
            eps = torch.randn_like(xn0)
            xn_t = ab.sqrt() * xn0 + (1 - ab).sqrt() * eps
            xc_t = self._cat_q_sample(xc0, t)

            out = self.model(torch.cat([xn_t, xc_t], dim=1), t)
            loss = F.mse_loss(out[:, : self.n_num], eps) if self.n_num else 0.0

            if self.cat_sizes:
                x0_hat = self._cat_x0_probs(out[:, self.n_num :])
                true_post = self._cat_posterior(xc_t, xc0, t)
                pred_post = self._cat_posterior(xc_t, x0_hat, t)
                kl = sum(
                    (tp * (torch.log(tp + 1e-12) - torch.log(pp + 1e-12))).sum(dim=1)
                    for tp, pp in zip(true_post, pred_post)
                )
                loss = loss + kl.mean() / len(self.cat_sizes)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (step + 1) % 2000 == 0:
                print(f"  TabDDPM step {step + 1}/{self.steps} | loss {loss.item():.4f}")
        return self

    # ---------- sampling ----------
    @torch.no_grad()
    def sample(self, n):
        torch.manual_seed(self.seed + 1)
        self.model.eval()
        xn = torch.randn(n, self.n_num)
        xc = torch.cat(
            [F.one_hot(torch.randint(0, K, (n,)), K).float() for K in self.cat_sizes], dim=1
        ) if self.cat_sizes else torch.zeros(n, 0)

        for i in reversed(range(self.T)):
            t = torch.full((n,), i, dtype=torch.long)
            out = self.model(torch.cat([xn, xc], dim=1), t)

            if self.n_num:
                eps_hat = out[:, : self.n_num]
                a, ab, ab_prev, b = self.alphas[i], self.alphas_bar[i], self.alphas_bar_prev[i], self.betas[i]
                mean = (xn - b / (1 - ab).sqrt() * eps_hat) / a.sqrt()
                if i > 0:
                    var = b * (1 - ab_prev) / (1 - ab)
                    xn = mean + var.sqrt() * torch.randn_like(xn)
                else:
                    xn = mean

            if self.cat_sizes:
                x0_hat = self._cat_x0_probs(out[:, self.n_num :])
                post = self._cat_posterior(xc, x0_hat, t)
                xc = torch.cat(
                    [F.one_hot(torch.multinomial(p, 1).squeeze(1), p.shape[1]).float() for p in post],
                    dim=1,
                )

        result = {}
        if self.n_num:
            nums = self.qt.inverse_transform(xn.numpy())
            for j, c in enumerate(self.num_cols):
                result[c] = nums[:, j]
        for block, c, cats in zip(self._split(xc), self.cat_cols, self.categories):
            result[c] = [cats[k] for k in block.argmax(dim=1).tolist()]
        return pd.DataFrame(result)[self.columns]
