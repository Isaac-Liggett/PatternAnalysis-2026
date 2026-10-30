"""Evaluate VAE checkpoints against test and train datasets.

For every VAE checkpoint found in --checkpoint-dir (sorted by epoch), this script
computes:

1. **Reconstruction Performance** — Mean SSIM, L1 reconstruction loss, and KL
   divergence against the held-out test dataset.
2. **SSIM of Generated Samples vs Test Set** — Best-match and mean pairwise SSIM
   between generated samples and real test images.
3. **Meehan et al. Three-Sample Non-Parametric Test for Data-Copying** — Compares
   nearest-neighbour distances from generated samples to the training set against
   held-out test samples to the training set (Meehan et al., AISTATS 2020),
   computing the U-statistic, Z_U score, and copying verdict.
4. **Visualisations & Metrics Persistence** — Per-checkpoint diagnostic plots
   (distance distributions, SSIM histogram, sample grid) and JSON, plus a
   cross-checkpoint summary (summary.json, summary.csv, and summary_trends.png).

Usage
-----
    python evaluate_vae.py \\
        --checkpoint-dir ./data/checkpoints \\
        --train-data-dir ./keras_png_slices_data/keras_png_slices_train \\
        --test-data-dir  ./keras_png_slices_data/keras_png_slices_test \\
        --num-samples    200 \\
        --output-dir     ./eval_vae_results
"""

from pathlib import Path
import argparse
import csv
import gc
import json
import math
import re
import time
import traceback

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

from vae import VAE
from dataset import get_dataloader


# ---------------------------------------------------------------------------
# Checkpoint Discovery & Epoch Parsing
# ---------------------------------------------------------------------------

def parse_epoch(path):
    """Return the last integer in the filename stem, or None if there is none."""
    nums = re.findall(r"\d+", Path(path).stem)
    return int(nums[-1]) if nums else None


def checkpoint_sort_key(path):
    epoch = parse_epoch(path)
    return (epoch if epoch is not None else math.inf, Path(path).name)


def scan_checkpoints(directory: Path, pattern: str = "*.pt", stride: int = 1):
    """Find valid VAE checkpoint files in *directory* matching *pattern*.

    A file is considered a valid VAE checkpoint if it is a .pt file
    that can be loaded and contains a 'model_state_dict' key.
    Checkpoints are sorted numerically by epoch.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {directory}")

    candidates = sorted(directory.glob(pattern), key=checkpoint_sort_key)
    if not candidates:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {directory}")

    valid = []
    for p in candidates:
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt:
                valid.append(p)
            else:
                print(f"  [!] Skipping {p.name}: missing 'model_state_dict' key (not a VAE checkpoint?)")
        except Exception as e:
            print(f"  [!] Skipping {p.name}: failed to load ({e})")

    if not valid:
        raise RuntimeError(
            f"No valid VAE checkpoints found in {directory}. "
            f"Expected .pt files with a 'model_state_dict' key."
        )

    if stride > 1:
        selected = valid[::stride]
        if selected[-1] != valid[-1]:
            selected.append(valid[-1])
        valid = selected

    return valid


# ---------------------------------------------------------------------------
# SSIM (pure-PyTorch, no extra dependencies)
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(size: int, sigma: float, device):
    """1-D Gaussian kernel normalised to sum to 1."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_kernel_2d(size: int, sigma: float, channels: int, device):
    """Separable 2-D Gaussian kernel shaped (channels, 1, size, size)."""
    k1d = _gaussian_kernel_1d(size, sigma, device)
    k2d = k1d[:, None] * k1d[None, :]
    return k2d.expand(channels, 1, size, size).contiguous()


def ssim_batch(x, y, window_size=11, sigma=1.5, C1=0.01 ** 2, C2=0.03 ** 2):
    """Compute per-image SSIM between batches x and y (B, C, H, W).

    Both tensors are expected in [0, 1]. Returns a 1D tensor of shape (B,).
    """
    channels = x.size(1)
    kernel = _gaussian_kernel_2d(window_size, sigma, channels, x.device)
    pad = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=channels)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=channels) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    ssim_map = num / den  # (B, C, H, W)
    return ssim_map.mean(dim=(1, 2, 3))


# ---------------------------------------------------------------------------
# KL Divergence
# ---------------------------------------------------------------------------

def kl_divergence(mu, log_var):
    """Per-sample KL divergence: KL(q(z|x) || N(0, I)).

    Returns a tensor of shape (B,).
    """
    return -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)


# ---------------------------------------------------------------------------
# SSIM Evaluation: Generated Samples vs Test Dataset
# ---------------------------------------------------------------------------

@torch.no_grad()
def load_images(dataloader):
    """Collect every batch of a dataloader into one (N, C, H, W) CPU tensor in [0, 1]."""
    return torch.cat([batch.clamp(0, 1) for batch in dataloader], dim=0)


@torch.no_grad()
def compute_ssim_against_test(gen_images, test_images, device, chunk_size=32):
    """Compute SSIM metrics comparing generated samples against the test set.

    For each generated image, find its best structural match (max SSIM)
    across the entire test dataset, as well as the average similarity.
    """
    gen_images = gen_images.clamp(0, 1)
    m = gen_images.size(0)
    n = test_images.size(0)

    best_ssim_per_gen = torch.full((m,), -1.0, dtype=torch.float32)
    best_match_idx = torch.zeros((m,), dtype=torch.long)
    total_pairwise_ssim = 0.0

    for g_start in range(0, m, chunk_size):
        g_end = min(g_start + chunk_size, m)
        g_chunk = gen_images[g_start:g_end].to(device)  # (Bg, C, H, W)
        bg = g_chunk.size(0)

        chunk_best_ssim = torch.full((bg,), -1.0, device=device)
        chunk_best_idx = torch.zeros((bg,), dtype=torch.long, device=device)

        for t_start in range(0, n, chunk_size):
            t_end = min(t_start + chunk_size, n)
            t_chunk = test_images[t_start:t_end].to(device)  # (Bt, C, H, W)
            bt = t_chunk.size(0)

            # Pairwise broadcast -> (Bg * Bt, C, H, W)
            g_exp = g_chunk.repeat_interleave(bt, dim=0)
            t_exp = t_chunk.repeat(bg, 1, 1, 1)

            pair_ssim = ssim_batch(g_exp, t_exp).view(bg, bt)
            total_pairwise_ssim += pair_ssim.sum().item()

            max_in_batch, argmax_in_batch = pair_ssim.max(dim=1)
            better = max_in_batch > chunk_best_ssim
            chunk_best_ssim = torch.where(better, max_in_batch, chunk_best_ssim)
            chunk_best_idx = torch.where(better, t_start + argmax_in_batch, chunk_best_idx)

        best_ssim_per_gen[g_start:g_end] = chunk_best_ssim.cpu()
        best_match_idx[g_start:g_end] = chunk_best_idx.cpu()

    return {
        "best_match_ssim_mean": best_ssim_per_gen.mean().item(),
        "best_match_ssim_std": best_ssim_per_gen.std().item(),
        "best_match_ssim_min": best_ssim_per_gen.min().item(),
        "best_match_ssim_max": best_ssim_per_gen.max().item(),
        "mean_pairwise_ssim": total_pairwise_ssim / (m * n),
        "per_sample_best_ssim": best_ssim_per_gen.tolist(),
        "best_match_indices": best_match_idx.tolist(),
    }


# ---------------------------------------------------------------------------
# Meehan et al. Three-Sample Non-Parametric Test for Data-Copying
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_representations(dataloader, model=None, space="pixel", max_samples=None, device="cpu"):
    """Extract representations for a dataset in either latent or pixel space."""
    features = []
    total = 0

    for batch in dataloader:
        batch = batch.to(device)
        if space == "latent":
            if model is None:
                raise ValueError("model must be provided when space='latent'")
            mu, _ = model.encode(batch)
            feats = mu.flatten(start_dim=1)
        elif space == "pixel":
            feats = batch.flatten(start_dim=1)
        else:
            raise ValueError(f"Unknown representation space: {space}")

        features.append(feats.cpu())
        total += feats.size(0)
        if max_samples is not None and total >= max_samples:
            break

    out = torch.cat(features, dim=0)
    if max_samples is not None and out.size(0) > max_samples:
        out = out[:max_samples]
    return out


@torch.no_grad()
def extract_representations_from_tensor(images, model=None, space="pixel", batch_size=32, device="cpu"):
    """Extract representations from an in-memory image tensor (M, C, H, W)."""
    if space == "pixel":
        return images.flatten(start_dim=1)
    elif space == "latent":
        if model is None:
            raise ValueError("model must be provided when space='latent'")
        features = []
        for i in range(0, images.size(0), batch_size):
            batch = images[i:i + batch_size].to(device)
            mu, _ = model.encode(batch)
            features.append(mu.flatten(start_dim=1).cpu())
        return torch.cat(features, dim=0)
    else:
        raise ValueError(f"Unknown representation space: {space}")


@torch.no_grad()
def compute_nearest_neighbor_distances(queries, targets, chunk_size=256, device="cpu"):
    """Compute Euclidean distance from each query to its nearest neighbor in targets.

    d(q, targets) = min_{t in targets} ||q - t||_2
    """
    q_count = queries.size(0)
    min_distances = torch.zeros(q_count, dtype=torch.float32)

    targets_dev = targets.to(device)
    target_norms_sq = (targets_dev ** 2).sum(dim=1)  # (T,)

    for i in range(0, q_count, chunk_size):
        end = min(i + chunk_size, q_count)
        q_chunk = queries[i:end].to(device)  # (B, D)

        q_norms_sq = (q_chunk ** 2).sum(dim=1, keepdim=True)  # (B, 1)
        dists_sq = q_norms_sq - 2.0 * (q_chunk @ targets_dev.t()) + target_norms_sq.unsqueeze(0)
        dists_sq = dists_sq.clamp(min=0.0)

        min_dists, _ = dists_sq.min(dim=1)
        min_distances[i:end] = torch.sqrt(min_dists).cpu()

    return min_distances


def _average_ranks(a):
    """Ranks (1-based) of a 1-D array, with tied values given their average rank."""
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def mann_whitney_u_test(x, y):
    """Two-sample Mann-Whitney U test between 1D arrays x and y.

    Tests whether x is stochastically smaller than y (the data-copying
    alternative when x = generated distances and y = test distances).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    m = len(y)

    if n == 0 or m == 0:
        raise ValueError("Inputs x and y must both be non-empty.")

    ranks = _average_ranks(np.concatenate([x, y]))
    u_x = ranks[:n].sum() - n * (n + 1) / 2.0
    mean_u = (n * m) / 2.0
    std_u = math.sqrt((n * m * (n + m + 1)) / 12.0)
    z_score = (u_x - mean_u) / std_u if std_u > 0 else 0.0

    try:
        from scipy import stats
        p_two = float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)
        p_less = float(stats.mannwhitneyu(x, y, alternative="less").pvalue)
    except ImportError:
        p_less = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
        p_two = 2.0 * min(p_less, 1.0 - p_less)

    return {
        "u_statistic": float(u_x),
        "z_score": float(z_score),
        "p_value_two_sided": p_two,
        "p_value_copying_one_sided": p_less,
        "n_x": n,
        "n_y": m,
    }


def meehan_three_sample_test(train_feats, test_feats, gen_feats, device="cpu", d_test=None):
    """Execute the Meehan et al. Three-Sample Non-Parametric Test for Data-Copying."""
    if d_test is None:
        d_test = compute_nearest_neighbor_distances(test_feats, train_feats, device=device)
    d_gen = compute_nearest_neighbor_distances(gen_feats, train_feats, device=device)

    d_test_np = d_test.numpy()
    d_gen_np = d_gen.numpy()

    test_res = mann_whitney_u_test(d_gen_np, d_test_np)
    z_u = test_res["z_score"]
    p_two = test_res["p_value_two_sided"]
    p_copy = test_res["p_value_copying_one_sided"]

    if z_u < -1.96 and p_copy < 0.05:
        verdict = (
            "DATA-COPYING DETECTED: Synthetic samples are significantly closer to "
            "training images than held-out test images are (Z_U << 0, p < 0.05)."
        )
        status = "copying"
    elif z_u > 1.96:
        verdict = (
            "NO COPYING (UNDERFITTING / SHIFT): Synthetic samples are farther from "
            "training images than held-out test images are (Z_U > 0)."
        )
        status = "underfitting"
    else:
        verdict = (
            "NO DATA-COPYING (GENERALIZING): Synthetic samples match the test distance "
            "distribution relative to training data (Z_U ~ 0, p >= 0.05)."
        )
        status = "generalizing"

    return {
        "z_u_score": z_u,
        "u_statistic": test_res["u_statistic"],
        "p_value_two_sided": p_two,
        "p_value_copying_one_sided": p_copy,
        "verdict": verdict,
        "status": status,
        "train_set_size": int(train_feats.size(0)),
        "test_set_size": int(test_feats.size(0)),
        "gen_set_size": int(gen_feats.size(0)),
        "mean_dist_test_to_train": float(d_test_np.mean()),
        "std_dist_test_to_train": float(d_test_np.std()),
        "mean_dist_gen_to_train": float(d_gen_np.mean()),
        "std_dist_gen_to_train": float(d_gen_np.std()),
        "dist_test_to_train": d_test_np.tolist(),
        "dist_gen_to_train": d_gen_np.tolist(),
    }


# ---------------------------------------------------------------------------
# Sampling from VAE Prior
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_vae_samples(model, num_samples, device, batch_size=16, seed=42):
    """Generate num_samples images from the VAE by sampling z ~ N(0, I)."""
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    samples = []
    generated = 0
    while generated < num_samples:
        b = min(batch_size, num_samples - generated)
        z = torch.randn(b, model.latent_dim, generator=generator).to(device)
        imgs = model.decode(z).clamp(0, 1)
        samples.append(imgs.cpu())
        generated += b
    return torch.cat(samples, dim=0)


# ---------------------------------------------------------------------------
# Visualisations & Reporting
# ---------------------------------------------------------------------------

def plot_evaluation(ssim_res, meehan_res, gen_images, output_dir, title_prefix=""):
    """Generate diagnostic figures for SSIM and the Meehan Three-Sample test."""
    output_dir = Path(output_dir)

    # 1. Distance distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    d_test = np.array(meehan_res["dist_test_to_train"])
    d_gen = np.array(meehan_res["dist_gen_to_train"])

    bins = np.histogram_bin_edges(np.concatenate([d_test, d_gen]), bins=30)
    ax1.hist(d_test, bins=bins, alpha=0.6, label=f"Test -> Train (N={len(d_test)})", color="royalblue", density=True)
    ax1.hist(d_gen, bins=bins, alpha=0.6, label=f"Synthetic -> Train (N={len(d_gen)})", color="crimson", density=True)
    ax1.axvline(d_test.mean(), color="darkblue", linestyle="--", label=f"Mean Test ({d_test.mean():.2f})")
    ax1.axvline(d_gen.mean(), color="darkred", linestyle="--", label=f"Mean Gen ({d_gen.mean():.2f})")
    ax1.set_xlabel("Nearest-Neighbor Distance to Training Set")
    ax1.set_ylabel("Density")
    ax1.set_title(f"{title_prefix}Meehan et al. Distance Distributions\n"
                  f"$Z_U = {meehan_res['z_u_score']:.3f}$ ({meehan_res['status']})")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.boxplot([d_test, d_gen], tick_labels=["Test -> Train", "Synthetic -> Train"], patch_artist=True,
                boxprops=dict(facecolor="lightblue", color="blue"),
                medianprops=dict(color="black", linewidth=1.5))
    ax2.set_ylabel("Nearest-Neighbor Euclidean Distance")
    ax2.set_title("Distance Comparison (Median & IQR)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "meehan_three_sample_test.png", dpi=150)
    plt.close(fig)

    # 2. SSIM distribution
    best_ssims = np.array(ssim_res["per_sample_best_ssim"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(best_ssims, bins=25, color="teal", edgecolor="black", alpha=0.7)
    ax.axvline(ssim_res["best_match_ssim_mean"], color="red", linestyle="--",
               label=f"Mean Best SSIM: {ssim_res['best_match_ssim_mean']:.4f}")
    ax.set_xlabel("Best-Match SSIM against Test Set")
    ax.set_ylabel("Count of Generated Images")
    ax.set_title(f"{title_prefix}Synthetic Image SSIM against Test Dataset")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "ssim_distribution.png", dpi=150)
    plt.close(fig)

    # 3. Grid of generated samples
    n_display = min(16, gen_images.size(0))
    grid = make_grid(gen_images[:n_display].clamp(0, 1), nrow=4, padding=4, normalize=False)
    save_image(grid, output_dir / "generated_samples_grid.png")


SUMMARY_FIELDS = [
    "checkpoint", "epoch", "train_loss",
    "mean_recon_ssim", "mean_recon_l1", "mean_kl",
    "best_match_ssim_mean", "best_match_ssim_std", "mean_pairwise_ssim",
    "z_u_score", "u_statistic", "p_value_two_sided", "p_value_copying_one_sided",
    "status", "mean_dist_gen_to_train", "mean_dist_test_to_train",
    "generation_seconds", "gen_images_per_sec", "recon_eval_seconds", "error",
]


def write_summary(records, output_dir):
    """Write summary JSON, CSV and trend plot for all checkpoints evaluated."""
    output_dir = Path(output_dir)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(records, f, indent=2)

    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(records)

    ok = [r for r in records if "error" not in r]
    if len(ok) < 2:
        return

    if all(r.get("epoch") is not None and r["epoch"] >= 0 for r in ok):
        x = [r["epoch"] for r in ok]
        xlabel = "Epoch"
    else:
        x = list(range(len(ok)))
        xlabel = "Checkpoint Index"

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Reconstruction SSIM
    ax[0, 0].plot(x, [r["mean_recon_ssim"] for r in ok], marker="o", color="royalblue")
    ax[0, 0].set_title("Reconstruction SSIM vs Test Set (↑)")
    ax[0, 0].set_ylabel("SSIM")

    # 2. Generated SSIM vs Test Set
    ax[0, 1].plot(x, [r["best_match_ssim_mean"] for r in ok], marker="o", color="teal", label="Best-match SSIM")
    ax[0, 1].plot(x, [r["mean_pairwise_ssim"] for r in ok], marker="s", color="darkorange", label="Pairwise SSIM")
    ax[0, 1].set_title("Generated Image SSIM vs Test Set (↑)")
    ax[0, 1].set_ylabel("SSIM")
    ax[0, 1].legend()

    # 3. Training Loss & Reconstruction L1
    ax[0, 2].plot(x, [r["mean_recon_l1"] for r in ok], marker="o", color="green", label="Recon L1")
    losses = [r.get("train_loss", float("nan")) for r in ok]
    if any(not math.isnan(l) for l in losses):
        ax[0, 2].plot(x, losses, marker="^", color="darkorange", label="Train Loss")
    ax[0, 2].set_title("Losses (↓)")
    ax[0, 2].set_ylabel("Loss")
    ax[0, 2].legend()

    # 4. KL Divergence
    ax[1, 0].plot(x, [r["mean_kl"] for r in ok], marker="s", color="crimson")
    ax[1, 0].set_title("KL Divergence: KL(q(z|x) || N(0, I))")
    ax[1, 0].set_ylabel("KL")

    # 5. Meehan Z_U Score
    ax[1, 1].plot(x, [r["z_u_score"] for r in ok], marker="o", color="crimson")
    ax[1, 1].axhline(-1.96, color="gray", linestyle="--", label="±1.96 threshold")
    ax[1, 1].axhline(1.96, color="gray", linestyle="--")
    ax[1, 1].axhline(0, color="black", linewidth=0.8)
    ax[1, 1].set_title("Meehan $Z_U$ (negative = data copying)")
    ax[1, 1].set_ylabel("$Z_U$")
    ax[1, 1].legend()

    # 6. Nearest-Neighbor Distances to Train Set
    ax[1, 2].plot(x, [r["mean_dist_gen_to_train"] for r in ok], marker="o", color="green", label="Synthetic -> Train")
    ax[1, 2].axhline(ok[0]["mean_dist_test_to_train"], color="royalblue", linestyle="--", label="Test -> Train")
    ax[1, 2].set_title("Mean NN Distance to Training Set")
    ax[1, 2].set_ylabel("Distance")
    ax[1, 2].legend()

    for a in ax.flat:
        a.set_xlabel(xlabel)
        a.grid(True, alpha=0.3)

    fig.suptitle("VAE Evaluation Summary Across Checkpoints", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "summary_trends.png", dpi=150)
    plt.close(fig)
    print(f"Saved summary trend plots -> {output_dir / 'summary_trends.png'}")


# ---------------------------------------------------------------------------
# Reference Cache & Per-Checkpoint Evaluation
# ---------------------------------------------------------------------------

def ensure_reference_cache(cache, model, train_loader, test_loader, args, device):
    """Compute and cache test images and train/test representations."""
    if "test_images" not in cache:
        print("  Loading test images for SSIM evaluation ...")
        cache["test_images"] = load_images(test_loader)

    if args.metric_space == "pixel":
        if "d_test" not in cache:
            print("  Extracting training pixel representations ...")
            cache["train_feats"] = extract_representations(
                train_loader, model=None, space="pixel", max_samples=args.max_train_samples, device=device
            )
            print("  Extracting test pixel representations ...")
            cache["test_feats"] = extract_representations(
                test_loader, model=None, space="pixel", max_samples=args.max_test_samples, device=device
            )
            print("  Computing test -> train nearest-neighbor distances ...")
            cache["d_test"] = compute_nearest_neighbor_distances(
                cache["test_feats"], cache["train_feats"], device=device
            )
    elif args.metric_space == "latent":
        print("  Extracting training latent representations ...")
        cache["train_feats"] = extract_representations(
            train_loader, model=model, space="latent", max_samples=args.max_train_samples, device=device
        )
        print("  Extracting test latent representations ...")
        cache["test_feats"] = extract_representations(
            test_loader, model=model, space="latent", max_samples=args.max_test_samples, device=device
        )
        print("  Computing test -> train nearest-neighbor distances ...")
        cache["d_test"] = compute_nearest_neighbor_distances(
            cache["test_feats"], cache["train_feats"], device=device
        )


def evaluate_checkpoint(ckpt_path, config, train_loader, test_loader, cache, args, device):
    """Evaluate one VAE checkpoint: reconstruction metrics + generation + Meehan test."""
    label = ckpt_path.stem
    out_dir = args.output_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = VAE(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", parse_epoch(ckpt_path) or -1)
    train_loss = ckpt.get("loss", float("nan"))

    try:
        ensure_reference_cache(cache, model, train_loader, test_loader, args, device)

        # 1. Test set reconstruction evaluation
        total_ssim = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_images = 0
        recon_start = time.perf_counter()

        for batch in test_loader:
            batch = batch.to(device)
            x_rec, _input, mu, log_var = model(batch)
            batch_ssim = ssim_batch(batch, x_rec.clamp(0, 1))
            total_ssim += batch_ssim.sum().item()
            total_recon += F.l1_loss(x_rec, batch, reduction="sum").item()
            total_kl += kl_divergence(mu, log_var).sum().item()
            n_images += batch.size(0)

        recon_elapsed = time.perf_counter() - recon_start
        mean_recon_ssim = total_ssim / n_images
        mean_recon_l1 = total_recon / n_images
        mean_kl = total_kl / n_images

        # 2. Generation of synthetic samples
        print(f"  Generating {args.num_samples} samples from latent prior N(0, I) ...")
        gen_start = time.perf_counter()
        gen_images = generate_vae_samples(
            model=model,
            num_samples=args.num_samples,
            device=device,
            batch_size=args.sample_batch_size,
            seed=args.seed,
        )
        gen_elapsed = time.perf_counter() - gen_start
        gen_images = gen_images.clamp(0, 1).cpu()
        print(f"  Generated in {gen_elapsed:.2f}s ({args.num_samples / gen_elapsed:.1f} img/s)")

        if args.save_all_samples:
            samples_dir = out_dir / "samples"
            samples_dir.mkdir(exist_ok=True)
            for idx in range(gen_images.size(0)):
                save_image(gen_images[idx], samples_dir / f"sample_{idx:05d}.png")

        # 3. SSIM of generated images against test dataset
        ssim_results = compute_ssim_against_test(gen_images, cache["test_images"], device)

        # 4. Meehan et al. Three-Sample Data-Copying Test
        gen_feats = extract_representations_from_tensor(
            gen_images, model=model, space=args.metric_space, batch_size=args.batch_size, device=device
        )
        meehan_results = meehan_three_sample_test(
            train_feats=cache["train_feats"],
            test_feats=cache["test_feats"],
            gen_feats=gen_feats,
            device=device,
            d_test=cache["d_test"],
        )

        # 5. Diagnostic plots
        plot_evaluation(ssim_results, meehan_results, gen_images, out_dir, title_prefix=f"{label} (Epoch {epoch})\n")

        record = {
            "checkpoint": label,
            "epoch": epoch,
            "path": str(ckpt_path),
            "train_loss": train_loss,
            "mean_recon_ssim": mean_recon_ssim,
            "mean_recon_l1": mean_recon_l1,
            "mean_kl": mean_kl,
            "recon_eval_seconds": recon_elapsed,
            "generation_seconds": gen_elapsed,
            "gen_images_per_sec": args.num_samples / gen_elapsed if gen_elapsed > 0 else 0.0,
            "best_match_ssim_mean": ssim_results["best_match_ssim_mean"],
            "best_match_ssim_std": ssim_results["best_match_ssim_std"],
            "best_match_ssim_min": ssim_results["best_match_ssim_min"],
            "best_match_ssim_max": ssim_results["best_match_ssim_max"],
            "mean_pairwise_ssim": ssim_results["mean_pairwise_ssim"],
            "status": meehan_results["status"],
            "verdict": meehan_results["verdict"],
            "z_u_score": meehan_results["z_u_score"],
            "u_statistic": meehan_results["u_statistic"],
            "p_value_two_sided": meehan_results["p_value_two_sided"],
            "p_value_copying_one_sided": meehan_results["p_value_copying_one_sided"],
            "train_set_size": meehan_results["train_set_size"],
            "test_set_size": meehan_results["test_set_size"],
            "gen_set_size": meehan_results["gen_set_size"],
            "mean_dist_test_to_train": meehan_results["mean_dist_test_to_train"],
            "std_dist_test_to_train": meehan_results["std_dist_test_to_train"],
            "mean_dist_gen_to_train": meehan_results["mean_dist_gen_to_train"],
            "std_dist_gen_to_train": meehan_results["std_dist_gen_to_train"],
        }

        with open(out_dir / "vae_eval_metrics.json", "w") as f:
            json.dump({
                "config": {
                    "checkpoint": str(ckpt_path),
                    "num_samples": args.num_samples,
                    "metric_space": args.metric_space,
                    "seed": args.seed,
                    "vae_config": config,
                },
                "metrics": record,
            }, f, indent=2)

        return record

    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI & Execution
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate VAE checkpoints: reconstruction, generation, and Meehan et al. Data-Copying Test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Checkpoints
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="Directory containing VAE checkpoint (.pt) files")
    parser.add_argument("--checkpoint-glob", type=str, default="*.pt",
                        help="Glob pattern matching checkpoint filenames (default: '*.pt')")
    parser.add_argument("--stride", type=int, default=1,
                        help="Evaluate every k-th checkpoint (the last one is always included)")

    # Data directories
    parser.add_argument("--train-data-dir", type=Path, default=None,
                        help="Root directory of training images (for Meehan test)")
    parser.add_argument("--test-data-dir", type=Path, default=None,
                        help="Root directory of test images (for reconstruction & Meehan test)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Fallback / parent dataset directory")

    # Generation parameters
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of synthetic samples to generate per checkpoint (default: 200)")
    parser.add_argument("--sample-batch-size", type=int, default=16,
                        help="Batch size during sample generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed; identical for every checkpoint so latents match")

    # Evaluation configuration
    parser.add_argument("--image-size", type=int, default=256,
                        help="Spatial resolution of images")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for dataloaders")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Worker processes for data loading")
    parser.add_argument("--metric-space", type=str, default="pixel",
                        choices=["pixel", "latent"],
                        help="Feature space for Meehan test ('pixel' or 'latent', default: 'pixel')")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Optional cap on training samples for nearest neighbor distance")
    parser.add_argument("--max-test-samples", type=int, default=None,
                        help="Optional cap on test samples for nearest neighbor distance")
    parser.add_argument("--output-dir", type=Path, default=Path("./eval_vae_results"),
                        help="Directory for saving metrics and diagnostic plots")
    parser.add_argument("--save-all-samples", action="store_true",
                        help="Save all individual generated images as PNGs (per checkpoint)")

    # VAE architecture parameters (must match training parameters)
    parser.add_argument("--in-channels", type=int, default=1,
                        help="Number of input image channels")
    parser.add_argument("--latent-dim", type=int, default=128,
                        help="Dimensionality of the latent space")
    parser.add_argument("--hidden-dims", type=int, nargs="+",
                        default=[32, 64, 128, 256, 512],
                        help="Hidden layer channel sizes for encoder/decoder")

    return parser.parse_args()


def resolve_data_dirs(args):
    """Resolve train and test data directories with sensible defaults."""
    train_dir = args.train_data_dir
    test_dir = args.test_data_dir

    if train_dir is None and args.data_dir is not None:
        cand_train = [
            args.data_dir / "keras_png_slices_train",
            args.data_dir / "train",
            args.data_dir / "AD",
            args.data_dir / "NC",
        ]
        for c in cand_train:
            if c.is_dir():
                train_dir = c
                break
        if train_dir is None and args.data_dir.is_dir():
            train_dir = args.data_dir

    if test_dir is None and args.data_dir is not None:
        cand_test = [
            args.data_dir / "keras_png_slices_test",
            args.data_dir / "test",
            args.data_dir / "val",
        ]
        for c in cand_test:
            if c.is_dir():
                test_dir = c
                break
        if test_dir is None and args.data_dir.is_dir():
            test_dir = args.data_dir

    if train_dir is None or not Path(train_dir).is_dir():
        raise FileNotFoundError(
            f"Train data directory not found or not specified ({train_dir}). "
            f"Please specify --train-data-dir."
        )
    if test_dir is None or not Path(test_dir).is_dir():
        raise FileNotFoundError(
            f"Test data directory not found or not specified ({test_dir}). "
            f"Please specify --test-data-dir (or --data-dir)."
        )

    return Path(train_dir), Path(test_dir)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build VAE config from CLI args
    config = dict(
        in_channels=args.in_channels,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        image_size=args.image_size,
    )
    print(f"VAE config: {config}")

    train_dir, test_dir = resolve_data_dirs(args)
    print(f"Train data directory: {train_dir}")
    print(f"Test data directory:  {test_dir}")

    train_loader = get_dataloader(
        str(train_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    test_loader = get_dataloader(
        str(test_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    print(f"Train set: {len(train_loader.dataset)} images")
    print(f"Test set:  {len(test_loader.dataset)} images")

    # Scan checkpoint directory
    print(f"\nScanning {args.checkpoint_dir} for VAE checkpoints matching '{args.checkpoint_glob}' ...")
    ckpt_paths = scan_checkpoints(args.checkpoint_dir, pattern=args.checkpoint_glob, stride=args.stride)
    print(f"Found {len(ckpt_paths)} valid checkpoint(s) to evaluate.")

    cache = {}
    records = []

    for idx, ckpt_path in enumerate(ckpt_paths, start=1):
        print(f"\n{'=' * 70}")
        print(f"[{idx}/{len(ckpt_paths)}] Evaluating: {ckpt_path.name}")
        print(f"{'-' * 70}")
        try:
            rec = evaluate_checkpoint(ckpt_path, config, train_loader, test_loader, cache, args, device)
            records.append(rec)
            print(f"  Recon SSIM: {rec['mean_recon_ssim']:.4f}  |  "
                  f"Recon L1: {rec['mean_recon_l1']:.4f}  |  "
                  f"KL: {rec['mean_kl']:.4f}")
            print(f"  Best Gen SSIM vs Test: {rec['best_match_ssim_mean']:.4f} +- {rec['best_match_ssim_std']:.4f}  |  "
                  f"Pairwise: {rec['mean_pairwise_ssim']:.4f}")
            print(f"  Meehan Z_U: {rec['z_u_score']:+.3f} "
                  f"(p_copy={rec['p_value_copying_one_sided']:.3e}) -> {rec['status']}")
        except Exception as e:
            traceback.print_exc()
            records.append({
                "checkpoint": ckpt_path.stem,
                "epoch": parse_epoch(ckpt_path),
                "path": str(ckpt_path),
                "error": str(e),
            })

        # Update running summaries after each checkpoint
        write_summary(records, args.output_dir)

    # Final summary table
    print(f"\n{'=' * 105}")
    print(f"{'Epoch':>6}  {'Recon SSIM':>10}  {'Recon L1':>9}  {'KL':>8}  "
          f"{'Gen SSIM':>9}  {'Z_U':>8}  {'p_copy':>10}  {'Status':>14}  {'img/s':>7}")
    print(f"{'-' * 105}")
    for r in records:
        if "error" in r:
            print(f"{str(r.get('epoch', '?')):>6}  ERROR: {r['error']}")
            continue
        print(f"{r['epoch']:>6d}  {r['mean_recon_ssim']:>10.4f}  "
              f"{r['mean_recon_l1']:>9.4f}  {r['mean_kl']:>8.4f}  "
              f"{r['best_match_ssim_mean']:>9.4f}  {r['z_u_score']:>+8.3f}  "
              f"{r['p_value_copying_one_sided']:>10.3e}  {r['status']:>14}  "
              f"{r['gen_images_per_sec']:>7.1f}")
    print(f"{'=' * 105}")
    print(f"All results saved to -> {args.output_dir}")


if __name__ == "__main__":
    main()
