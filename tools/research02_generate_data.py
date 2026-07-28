from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 100
N_GENERATE = 1000
TEMPORAL_BLOCK_RATIO = 0.25
FEATURE_GROUP_RATIO = 0.4
RESIDUAL_SCALE = 0.35
DIFFUSION_TIMESTEPS = 80
DIFFUSION_START_STEP = 20
TSNE_GENERATED_SAMPLE_SIZE = None
ADAPTIVE_MASK_NORMAL_SAMPLE_SIZE = 5000
EPS = 1e-6


def set_seed() -> None:
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)


def make_structured_mask(
    x: torch.Tensor,
    temporal_block_ratio: float = TEMPORAL_BLOCK_RATIO,
    feature_group_ratio: float = FEATURE_GROUP_RATIO,
    adaptive_config: dict | None = None,
) -> torch.Tensor:
    """Mask contiguous time blocks and feature groups for temporal restoration."""
    if x.ndim != 3:
        raise ValueError(f"Structured masking expects 3D windows, got shape {tuple(x.shape)}")

    bsz, window_size, n_features = x.shape
    mask = torch.ones_like(x)
    block_len = max(1, int(round(window_size * temporal_block_ratio)))
    group_len = max(1, int(round(n_features * feature_group_ratio)))

    if adaptive_config is not None:
        time_probs = torch.tensor(
            adaptive_config["time_start_probs"],
            dtype=torch.float32,
            device=x.device,
        )
        group_probs = torch.tensor(
            adaptive_config["feature_group_probs"],
            dtype=torch.float32,
            device=x.device,
        )
        feature_groups = adaptive_config["feature_groups"]

    for i in range(bsz):
        if adaptive_config is None:
            time_start = torch.randint(0, window_size - block_len + 1, (1,), device=x.device).item()
            feature_start = torch.randint(0, n_features - group_len + 1, (1,), device=x.device).item()
            feature_idx = slice(feature_start, feature_start + group_len)
        else:
            time_start = torch.multinomial(time_probs, 1).item()
            group_idx = torch.multinomial(group_probs, 1).item()
            feature_idx = feature_groups[group_idx]
        mask[i, time_start : time_start + block_len, :] = 0
        mask[i, :, feature_idx] = 0

    return mask


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32)
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    return ((cumsum[window:] - cumsum[:-window]) / window).astype(np.float32)


def build_adaptive_mask_config(
    normal_x: np.ndarray,
    anomaly_x: np.ndarray,
    feature_names: np.ndarray,
) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    normal_sample_size = min(ADAPTIVE_MASK_NORMAL_SAMPLE_SIZE, len(normal_x))
    normal_idx = rng.choice(len(normal_x), size=normal_sample_size, replace=False)
    normal_sample = normal_x[normal_idx]

    window_size = anomaly_x.shape[1]
    n_features = anomaly_x.shape[2]
    block_len = max(1, int(round(window_size * TEMPORAL_BLOCK_RATIO)))
    group_len = max(1, int(round(n_features * FEATURE_GROUP_RATIO)))

    combined = np.concatenate([normal_sample, anomaly_x], axis=0)
    feature_std = combined.reshape(-1, n_features).std(axis=0) + EPS
    diff = np.abs(anomaly_x.mean(axis=0) - normal_sample.mean(axis=0)) / feature_std.reshape(1, -1)

    time_scores = diff.mean(axis=1)
    if len(time_scores) >= 3:
        time_scores = np.convolve(time_scores, np.array([0.25, 0.5, 0.25]), mode="same")
    time_start_scores = rolling_mean(time_scores, block_len)
    time_start_probs = time_start_scores + EPS
    time_start_probs = time_start_probs / time_start_probs.sum()

    feature_scores = diff.mean(axis=0)
    corr = np.nan_to_num(np.abs(np.corrcoef(anomaly_x.reshape(-1, n_features).T)))
    feature_groups = []
    for anchor in np.argsort(feature_scores)[::-1]:
        members = tuple(sorted(np.argsort(corr[anchor])[::-1][:group_len].tolist()))
        if members not in feature_groups:
            feature_groups.append(members)
    if not feature_groups:
        feature_groups = [tuple(range(n_features))]

    group_scores = np.array(
        [feature_scores[list(group)].mean() for group in feature_groups],
        dtype=np.float32,
    )
    feature_group_probs = group_scores + EPS
    feature_group_probs = feature_group_probs / feature_group_probs.sum()

    return {
        "strategy": "data_driven_adaptive_temporal_feature_group_masking",
        "normal_sample_size": int(normal_sample_size),
        "temporal_block_ratio": TEMPORAL_BLOCK_RATIO,
        "feature_group_ratio": FEATURE_GROUP_RATIO,
        "block_len": int(block_len),
        "group_len": int(group_len),
        "time_start_probs": time_start_probs.astype(np.float32),
        "feature_groups": [np.array(group, dtype=np.int64) for group in feature_groups],
        "feature_group_probs": feature_group_probs.astype(np.float32),
        "top_time_starts": np.argsort(time_start_probs)[::-1][:5].astype(int).tolist(),
        "top_feature_groups": [
            [str(feature_names[idx]) for idx in group]
            for group in feature_groups[:5]
        ],
        "feature_scores": {
            str(feature_names[idx]): float(feature_scores[idx])
            for idx in range(n_features)
        },
    }


def summarize_adaptive_mask_config(config: dict) -> dict:
    return {
        "strategy": config["strategy"],
        "normal_sample_size": config["normal_sample_size"],
        "temporal_block_ratio": config["temporal_block_ratio"],
        "feature_group_ratio": config["feature_group_ratio"],
        "block_len": config["block_len"],
        "group_len": config["group_len"],
        "top_time_starts": config["top_time_starts"],
        "top_feature_groups": config["top_feature_groups"],
        "feature_scores": config["feature_scores"],
    }


class ResidualGRUGenerator(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.fc(h)


class GRUDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.fc(h[-1])


class DiffusionDenoiser(nn.Module):
    def __init__(self, data_dim: int, cond_dim: int = 0, hidden_dim: int = 256, time_dim: int = 32):
        super().__init__()
        self.time_embed = nn.Embedding(1000, time_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + cond_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        t_emb = self.time_embed(t)
        if cond is None:
            model_input = torch.cat([x, t_emb], dim=1)
        else:
            model_input = torch.cat([x, cond, t_emb], dim=1)
        return self.net(model_input)


class TemporalDiffusionDenoiser(nn.Module):
    def __init__(self, n_features: int, cond_dim: int, hidden_dim: int = 128, time_dim: int = 32):
        super().__init__()
        self.time_embed = nn.Embedding(1000, time_dim)
        self.gru = nn.GRU(n_features + cond_dim + time_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t).unsqueeze(1).expand(-1, x.size(1), -1)
        model_input = torch.cat([x, cond, t_emb], dim=-1)
        h, _ = self.gru(model_input)
        return self.fc(h)


def clamp_to_real_range(generated: np.ndarray, real_x: np.ndarray, margin: float = 0.25) -> np.ndarray:
    feature_min = real_x.min(axis=(0, 1))
    feature_max = real_x.max(axis=(0, 1))
    feature_std = real_x.reshape(-1, real_x.shape[-1]).std(axis=0)
    lower = feature_min - margin * feature_std
    upper = feature_max + margin * feature_std
    return np.clip(generated, lower.reshape(1, 1, -1), upper.reshape(1, 1, -1)).astype(np.float32)


def generate_gtgan_series(real_x: np.ndarray, device: str, n_generate: int = N_GENERATE) -> np.ndarray:
    real = torch.tensor(real_x, dtype=torch.float32)
    loader = DataLoader(TensorDataset(real), batch_size=BATCH_SIZE, shuffle=True)

    window_size = real_x.shape[1]
    n_features = real_x.shape[2]
    g = ResidualGRUGenerator(n_features, n_features).to(device)
    d = GRUDiscriminator(n_features).to(device)

    criterion_adv = nn.BCELoss()
    g_opt = torch.optim.Adam(g.parameters(), lr=1e-3)
    d_opt = torch.optim.Adam(d.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        for (batch,) in loader:
            batch = batch.to(device)
            bsz = batch.size(0)
            noise = torch.randn(bsz, window_size, n_features, device=device)
            fake = g(noise).detach()

            real_label = torch.ones(bsz, 1, device=device)
            fake_label = torch.zeros(bsz, 1, device=device)

            d_loss = criterion_adv(d(batch), real_label) + criterion_adv(d(fake), fake_label)
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            noise = torch.randn(bsz, window_size, n_features, device=device)
            fake = g(noise)
            g_loss = criterion_adv(d(fake), real_label)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

        if (epoch + 1) % 20 == 0:
            print(f"[noise GT-GAN series] epoch {epoch + 1}/{EPOCHS}")

    g.eval()
    generated = []
    with torch.no_grad():
        while len(generated) * BATCH_SIZE < n_generate:
            noise = torch.randn(BATCH_SIZE, window_size, n_features, device=device)
            fake = g(noise)
            generated.append(fake.cpu().numpy())

    generated_x = np.concatenate(generated, axis=0)[:n_generate]
    return clamp_to_real_range(generated_x, real_x)


def generate_gtgan_masked(
    real_x: np.ndarray,
    device: str,
    n_generate: int = N_GENERATE,
    mask_config: dict | None = None,
) -> np.ndarray:
    real = torch.tensor(real_x, dtype=torch.float32)
    loader = DataLoader(TensorDataset(real), batch_size=BATCH_SIZE, shuffle=True)

    n_features = real_x.shape[2]
    g = ResidualGRUGenerator(n_features * 3, n_features).to(device)
    d = GRUDiscriminator(n_features).to(device)

    criterion_adv = nn.BCELoss()
    criterion_rec = nn.MSELoss()
    g_opt = torch.optim.Adam(g.parameters(), lr=1e-3)
    d_opt = torch.optim.Adam(d.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        for (batch,) in loader:
            batch = batch.to(device)
            bsz = batch.size(0)
            mask = make_structured_mask(batch, adaptive_config=mask_config).to(device)
            noise = torch.randn_like(batch)
            masked_batch = batch * mask
            g_input = torch.cat([masked_batch, mask, noise], dim=-1)
            proposal = batch + RESIDUAL_SCALE * torch.tanh(g(g_input))
            fake = mask * batch + (1 - mask) * proposal

            real_label = torch.ones(bsz, 1, device=device)
            fake_label = torch.zeros(bsz, 1, device=device)

            d_loss = criterion_adv(d(batch), real_label) + criterion_adv(d(fake.detach()), fake_label)
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            proposal = batch + RESIDUAL_SCALE * torch.tanh(g(g_input))
            fake = mask * batch + (1 - mask) * proposal
            adv_loss = criterion_adv(d(fake), real_label)
            rec_loss = criterion_rec(fake * (1 - mask), batch * (1 - mask))
            smooth_loss = criterion_rec(fake[:, 1:] - fake[:, :-1], batch[:, 1:] - batch[:, :-1])
            g_loss = adv_loss + 8.0 * rec_loss + 1.0 * smooth_loss

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

        if (epoch + 1) % 20 == 0:
            print(f"[conditioned GT-GAN masked] epoch {epoch + 1}/{EPOCHS}")

    g.eval()
    generated = []
    with torch.no_grad():
        while len(generated) * BATCH_SIZE < n_generate:
            idx = np.random.choice(len(real_x), size=BATCH_SIZE, replace=True)
            seed = torch.tensor(real_x[idx], dtype=torch.float32, device=device)
            mask = make_structured_mask(seed, adaptive_config=mask_config).to(device)
            noise = torch.randn_like(seed)
            g_input = torch.cat([seed * mask, mask, noise], dim=-1)
            proposal = seed + RESIDUAL_SCALE * torch.tanh(g(g_input))
            fake = mask * seed + (1 - mask) * proposal
            generated.append(fake.cpu().numpy())

    generated_x = np.concatenate(generated, axis=0)[:n_generate]
    return clamp_to_real_range(generated_x, real_x)


def train_diffusion_model(
    real_flat: np.ndarray,
    device: str,
) -> tuple[DiffusionDenoiser, torch.Tensor, torch.Tensor, torch.Tensor]:
    real = torch.tensor(real_flat, dtype=torch.float32)
    loader = DataLoader(TensorDataset(real), batch_size=BATCH_SIZE, shuffle=True)

    data_dim = real_flat.shape[1]
    beta = torch.linspace(1e-4, 0.02, DIFFUSION_TIMESTEPS, device=device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    model = DiffusionDenoiser(data_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        for (x0,) in loader:
            x0 = x0.to(device)
            bsz = x0.size(0)
            t = torch.randint(0, DIFFUSION_TIMESTEPS, (bsz,), device=device)
            noise = torch.randn_like(x0)
            a_bar = alpha_bar[t].unsqueeze(1)
            xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

            pred_noise = model(xt, t)
            loss = criterion(pred_noise, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

        if (epoch + 1) % 20 == 0:
            print(f"[diffusion noise] epoch {epoch + 1}/{EPOCHS}")

    return model, beta, alpha, alpha_bar


def train_temporal_masked_diffusion_model(
    real_x: np.ndarray,
    device: str,
    mask_config: dict | None = None,
) -> tuple[TemporalDiffusionDenoiser, torch.Tensor, torch.Tensor, torch.Tensor]:
    real = torch.tensor(real_x, dtype=torch.float32)
    loader = DataLoader(TensorDataset(real), batch_size=BATCH_SIZE, shuffle=True)

    n_features = real_x.shape[2]
    beta = torch.linspace(1e-4, 0.02, DIFFUSION_TIMESTEPS, device=device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    model = TemporalDiffusionDenoiser(n_features, cond_dim=n_features * 2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        for (x0,) in loader:
            x0 = x0.to(device)
            bsz = x0.size(0)
            t = torch.randint(0, DIFFUSION_TIMESTEPS, (bsz,), device=device)
            noise = torch.randn_like(x0)
            a_bar = alpha_bar[t].view(bsz, 1, 1)
            xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise
            mask = make_structured_mask(x0, adaptive_config=mask_config).to(device)
            cond = torch.cat([x0 * mask, mask], dim=-1)

            pred_noise = model(xt, t, cond)
            loss = criterion(pred_noise, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

        if (epoch + 1) % 20 == 0:
            print(f"[temporal conditioned diffusion masked] epoch {epoch + 1}/{EPOCHS}")

    return model, beta, alpha, alpha_bar


def denoise_from_seed(
    model: DiffusionDenoiser,
    seed: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    alpha_bar: torch.Tensor,
    device: str,
    cond: torch.Tensor | None = None,
) -> torch.Tensor:
    noise = torch.randn_like(seed)
    x = torch.sqrt(alpha_bar[DIFFUSION_START_STEP]) * seed + torch.sqrt(
        1 - alpha_bar[DIFFUSION_START_STEP]
    ) * noise

    for step in reversed(range(DIFFUSION_START_STEP + 1)):
        t = torch.full((seed.size(0),), step, dtype=torch.long, device=device)
        pred_noise = model(x, t, cond)
        x = (1 / torch.sqrt(alpha[step])) * (
            x - ((1 - alpha[step]) / torch.sqrt(1 - alpha_bar[step])) * pred_noise
        )
        if step > 0:
            x = x + torch.sqrt(beta[step]) * torch.randn_like(x)

    return x


def denoise_temporal_from_seed(
    model: TemporalDiffusionDenoiser,
    seed: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    alpha_bar: torch.Tensor,
    device: str,
    cond: torch.Tensor,
) -> torch.Tensor:
    noise = torch.randn_like(seed)
    x = torch.sqrt(alpha_bar[DIFFUSION_START_STEP]) * seed + torch.sqrt(
        1 - alpha_bar[DIFFUSION_START_STEP]
    ) * noise

    for step in reversed(range(DIFFUSION_START_STEP + 1)):
        t = torch.full((seed.size(0),), step, dtype=torch.long, device=device)
        pred_noise = model(x, t, cond)
        x = (1 / torch.sqrt(alpha[step])) * (
            x - ((1 - alpha[step]) / torch.sqrt(1 - alpha_bar[step])) * pred_noise
        )
        if step > 0:
            x = x + torch.sqrt(beta[step]) * torch.randn_like(x)

    return x


def sample_diffusion_from_noise(
    model: DiffusionDenoiser,
    batch_size: int,
    data_dim: int,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    alpha_bar: torch.Tensor,
    device: str,
) -> torch.Tensor:
    x = torch.randn(batch_size, data_dim, device=device)

    for step in reversed(range(DIFFUSION_TIMESTEPS)):
        t = torch.full((batch_size,), step, dtype=torch.long, device=device)
        pred_noise = model(x, t)
        x = (1 / torch.sqrt(alpha[step])) * (
            x - ((1 - alpha[step]) / torch.sqrt(1 - alpha_bar[step])) * pred_noise
        )
        if step > 0:
            x = x + torch.sqrt(beta[step]) * torch.randn_like(x)

    return x


def generate_diffusion_series(real_x: np.ndarray, device: str, n_generate: int = N_GENERATE) -> np.ndarray:
    flat = real_x.reshape(real_x.shape[0], -1)
    data_dim = flat.shape[1]
    model, beta, alpha, alpha_bar = train_diffusion_model(flat, device)
    model.eval()

    generated = []
    with torch.no_grad():
        while len(generated) * BATCH_SIZE < n_generate:
            denoised = sample_diffusion_from_noise(
                model,
                BATCH_SIZE,
                data_dim,
                beta,
                alpha,
                alpha_bar,
                device,
            )
            generated.append(denoised.cpu().numpy())

    generated_x = np.concatenate(generated, axis=0)[:n_generate]
    generated_x = generated_x.reshape(n_generate, real_x.shape[1], real_x.shape[2])
    return clamp_to_real_range(generated_x, real_x)


def generate_diffusion_masked(
    real_x: np.ndarray,
    device: str,
    n_generate: int = N_GENERATE,
    mask_config: dict | None = None,
) -> np.ndarray:
    model, beta, alpha, alpha_bar = train_temporal_masked_diffusion_model(
        real_x,
        device,
        mask_config=mask_config,
    )
    model.eval()

    generated = []
    with torch.no_grad():
        while len(generated) * BATCH_SIZE < n_generate:
            idx = np.random.choice(len(real_x), size=BATCH_SIZE, replace=True)
            seed = torch.tensor(real_x[idx], dtype=torch.float32, device=device)
            mask = make_structured_mask(seed, adaptive_config=mask_config).to(device)
            cond = torch.cat([seed * mask, mask], dim=-1)
            denoised = denoise_temporal_from_seed(model, seed, beta, alpha, alpha_bar, device, cond=cond)
            restored = mask * seed + (1 - mask) * denoised
            generated.append(restored.cpu().numpy())

    generated_x = np.concatenate(generated, axis=0)[:n_generate]
    return clamp_to_real_range(generated_x, real_x)


def window_feature_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_2d = x.reshape(-1, x.shape[-1])
    mean = x_2d.mean(axis=0)
    std = x_2d.std(axis=0)
    corr = np.nan_to_num(np.corrcoef(x_2d.T))
    return mean, std, corr


def generated_quality_metrics(real_x: np.ndarray, generated_x: np.ndarray, method: str) -> dict:
    real_mean, real_std, real_corr = window_feature_stats(real_x)
    gen_mean, gen_std, gen_corr = window_feature_stats(generated_x)
    return {
        "method": method,
        "n_real": len(real_x),
        "n_generated": len(generated_x),
        "mean_abs_diff": float(np.mean(np.abs(real_mean - gen_mean))),
        "std_abs_diff": float(np.mean(np.abs(real_std - gen_std))),
        "corr_abs_diff": float(np.mean(np.abs(real_corr - gen_corr))),
        "real_min": float(real_x.min()),
        "real_max": float(real_x.max()),
        "generated_min": float(generated_x.min()),
        "generated_max": float(generated_x.max()),
    }


def flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def sample_windows(x: np.ndarray, n_sample: int | None, seed: int) -> np.ndarray:
    if n_sample is None or len(x) <= n_sample:
        return x
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x), size=n_sample, replace=False)
    return x[indices]


def save_tsne_figure(real_x: np.ndarray, generated_sets: dict[str, np.ndarray], figure_dir: Path) -> None:
    generated_sample_size = TSNE_GENERATED_SAMPLE_SIZE or len(real_x)
    sampled_generated_sets = {
        name: sample_windows(generated_x, generated_sample_size, seed=RANDOM_STATE + i)
        for i, (name, generated_x) in enumerate(generated_sets.items())
    }

    arrays = [flatten_windows(real_x)]
    labels = ["Real Data"] * len(real_x)

    for name, generated_sample in sampled_generated_sets.items():
        arrays.append(flatten_windows(generated_sample))
        labels.extend([name] * len(generated_sample))

    x_all = np.vstack(arrays)
    labels = np.asarray(labels)
    x_scaled = StandardScaler().fit_transform(x_all)
    perplexity = min(30, max(5, (len(x_scaled) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE,
    )
    embedded = tsne.fit_transform(x_scaled)
    real_points = embedded[labels == "Real Data"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    titles = {
        "GT-GAN": "Data generated by GT-GAN t-SNE",
        "Masking GT-GAN": "Data generated by masking-GT-GAN t-SNE",
        "Diffusion": "Data generated by Diffusion Model t-SNE",
        "Masking Diffusion": "Data generated by Masking-based Diffusion t-SNE",
    }

    for i, (name, generated_sample) in enumerate(sampled_generated_sets.items()):
        gen_points = embedded[labels == name]

        ax = axes[i]
        ax.scatter(
            gen_points[:, 0],
            gen_points[:, 1],
            s=28,
            alpha=0.45,
            label="Generated Data",
            color="#FAA43A",
            marker="o",
            edgecolor="none",
        )
        ax.scatter(
            real_points[:, 0],
            real_points[:, 1],
            s=24,
            alpha=0.75,
            label="Real Data",
            color="#5DA5DA",
            marker="o",
            edgecolor="white",
            linewidth=0.25,
        )
        ax.set_title(f"{titles[name]}\nReal n={len(real_points)}, Generated n={len(gen_points)}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(figure_dir / "tsne_2x2_real_vs_generated_full.png", dpi=200)
    plt.close()


def main() -> None:
    set_seed()
    sns.set_theme(style="whitegrid")

    root = Path.cwd()
    split_dir = root / "data" / "augmentation_split"
    out_dir = root / "data" / "research02" / "generated"
    result_dir = root / "data" / "research02" / "results"
    figure_dir = root / "data" / "research02" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(split_dir / "anomaly_train_seed_windows.npz", allow_pickle=True)
    real_x = data["X"].astype(np.float32)
    features = data["features"] if "features" in data.files else np.array(
        ["RealPower", "GateOnTime", "SetPower", "Speed", "Length"]
    )
    normal_data = np.load(split_dir / "normal_train_windows.npz", allow_pickle=True)
    normal_x = normal_data["X"].astype(np.float32)
    mask_config = build_adaptive_mask_config(normal_x, real_x, features)

    print("device:", device)
    print("real anomaly:", real_x.shape)
    print("adaptive mask:", summarize_adaptive_mask_config(mask_config))

    generated_sets = {
        "GT-GAN": generate_gtgan_series(real_x, device),
        "Masking GT-GAN": generate_gtgan_masked(real_x, device, mask_config=mask_config),
        "Diffusion": generate_diffusion_series(real_x, device),
        "Masking Diffusion": generate_diffusion_masked(real_x, device, mask_config=mask_config),
    }

    file_names = {
        "GT-GAN": "gtgan_series_windows.npz",
        "Masking GT-GAN": "gtgan_masked_windows.npz",
        "Diffusion": "diffusion_series_windows.npz",
        "Masking Diffusion": "diffusion_masked_windows.npz",
    }
    for method, generated_x in generated_sets.items():
        np.savez_compressed(
            out_dir / file_names[method],
            X=generated_x,
            y=np.ones(len(generated_x), dtype=np.int8),
            features=np.array(features),
        )
        print(method, generated_x.shape)

    quality_df = pd.DataFrame(
        [generated_quality_metrics(real_x, generated_x, method) for method, generated_x in generated_sets.items()]
    ).sort_values(["mean_abs_diff", "std_abs_diff", "corr_abs_diff"])
    quality_df.to_csv(result_dir / "generated_quality_metrics.csv", index=False)

    save_tsne_figure(real_x, generated_sets, figure_dir)

    summary = {
        "generation_fix": (
            "Generation follows the paper-style comparison: unmasked GT-GAN and Diffusion sample "
            "full windows from noise, while masking-based methods use real anomaly windows with "
            "structured masked-value restoration."
        ),
        "masking_strategy": (
            "Masking-based methods use data-driven adaptive temporal block masking and "
            "feature-group masking instead of element-wise random masking. Time regions are "
            "sampled with higher probability when normal-anomaly distributional differences "
            "are larger, and feature groups are formed from anomaly-feature correlation and "
            "normal-anomaly feature differences. The observed seed values are preserved, and "
            "only masked time/feature regions are restored by the generator."
        ),
        "adaptive_masking": summarize_adaptive_mask_config(mask_config),
        "masking_diffusion_denoiser": (
            "Masking Diffusion uses a GRU temporal denoiser conditioned on the masked seed window "
            "and the binary mask, rather than flattening the window into an MLP denoiser."
        ),
        "n_real": int(len(real_x)),
        "n_generated_per_method": N_GENERATE,
        "n_generated_shown_in_tsne": TSNE_GENERATED_SAMPLE_SIZE or int(len(real_x)),
        "temporal_block_ratio": TEMPORAL_BLOCK_RATIO,
        "feature_group_ratio": FEATURE_GROUP_RATIO,
        "residual_scale": RESIDUAL_SCALE,
        "quality_ranking": quality_df["method"].tolist(),
    }
    with open(result_dir / "research02_generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(quality_df)


if __name__ == "__main__":
    main()

