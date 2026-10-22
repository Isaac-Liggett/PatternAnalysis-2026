"""Evaluate many Latent Diffusion Model UNet checkpoints against one VQ-VAE.

For every UNet checkpoint found in --unet-ckpt-dir (sorted by the trailing
number in the filename, i.e. by epoch), this script computes:

1. **SSIM against Test Dataset** — best-match and mean pairwise SSIM between
   generated samples and held-out real test images.
2. **Meehan et al. Three-Sample Non-Parametric Test** — tests whether the model
   copies training samples by comparing nearest-neighbour distances to the
   training set for generated samples vs held-out test samples
   (Meehan et al., AISTATS 2020).
3. **Visualisations & Metrics Persistence** — per-checkpoint plots and JSON,
   plus a cross-checkpoint summary (JSON, CSV and trend plot).

Everything that does not depend on the UNet (test images, train/test features,
test->train nearest-neighbour distances) is computed once and reused. Every
checkpoint is sampled with the same seed, so all checkpoints start from the
same initial noise and are directly comparable.

Usage:
------
    python evaluate_diffusion.py \
        --vqvae-ckpt     ./data/checkpoints/vqvae_epoch_50.pt \
        --unet-ckpt-dir  ./data/checkpoints \
        --unet-glob      "denoising_unet_epoch*.pt" \
        --train-data-dir ./keras_png_slices_data/keras_png_slices_train \
        --test-data-dir  ./keras_png_slices_data/keras_png_slices_test \
        --num-samples    100 \
        --output-dir     ./eval_diffusion_results

Outputs:
    <output-dir>/<checkpoint-stem>/   per-checkpoint plots + metrics JSON
    <output-dir>/summary.json         metrics for all checkpoints
    <output-dir>/summary.csv          same, as a table
    <output-dir>/summary_trends.png   metrics vs epoch
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

from diffusion_model import StableDiffusion
from dataset import get_dataloader


# Checkpoint discovery
def parse_epoch(path):
    """Return the last integer in the filename stem, or None if there is none."""
    nums = re.findall(r"\d+", Path(path).stem)
    return int(nums[-1]) if nums else None


def checkpoint_sort_key(path):
    epoch = parse_epoch(path)
    return (epoch if epoch is not None else math.inf, Path(path).name)


def find_unet_checkpoints(ckpt_dir, pattern, stride=1):
    """Find UNet checkpoints in ckpt_dir matching a glob, sorted by epoch."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"UNet checkpoint directory not found: {ckpt_dir}")

    paths = sorted(ckpt_dir.glob(pattern), key=checkpoint_sort_key)
    if not paths:
        raise FileNotFoundError(
            f"No files matching '{pattern}' in {ckpt_dir}. "
            f"Use --unet-glob to match your checkpoint filenames."
        )

    if stride > 1:
        selected = paths[::stride]
        if selected[-1] != paths[-1]:  # always keep the final checkpoint
            selected.append(paths[-1])
        paths = selected
    return paths


# SSIM
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
    """Compute per-image SSIM between batches x and y of shape (B, C, H, W).

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

    ssim_map = num / den
    return ssim_map.mean(dim=(1, 2, 3))


# SSIM Evaluation: Generated Samples vs Test Dataset
@torch.no_grad()
def load_images(dataloader):
    """Collect every batch of a dataloader into one (N, C, H, W) CPU tensor in [0, 1]."""
    return torch.cat([batch.clamp(0, 1) for batch in dataloader], dim=0)


@torch.no_grad()
def compute_ssim_against_test(gen_images, test_images, device, chunk_size=32):
    """Compute SSIM metrics comparing generated samples against the test set.

    For each generated image, we find its best structural match (max SSIM)
    across the entire test dataset, as well as the average similarity.

    Parameters
    ----------
    gen_images : torch.Tensor
        Generated images (M, C, H, W) in range [0, 1].
    test_images : torch.Tensor
        Real test images (N, C, H, W) in range [0, 1], preloaded once.
    device : torch.device
        Device for computation.
    chunk_size : int
        Batch size of images processed at once to manage GPU memory.

    dict
        SSIM statistics (mean, std, min, max best-match SSIM, and mean pairwise SSIM).
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


# Meehan et al. Three-Sample Non-Parametric Test for Data-Copying
# (AISTATS 2020: "A Non-Parametric Test to Detect Data-Copying in Generative Models")
@torch.no_grad()
def extract_representations(dataloader, model=None, space="latent", max_samples=None, device="cpu"):
    """Extract representations for a dataset in either latent or pixel space.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader yielding image batches.
    model : StableDiffusion, optional
        Used when space == "latent" to encode images via the VQ-VAE.
    space : str
        "latent" (continuous scaled latents) or "pixel" (flattened raw pixels).
    max_samples : int, optional
        Maximum number of samples to collect.
    device : torch.device
        Compute device.

    torch.Tensor
        2D tensor (N, D) of extracted features.
    """
    features = []
    total = 0

    for batch in dataloader:
        batch = batch.to(device)
        if space == "latent":
            if model is None:
                raise ValueError("model must be provided when space='latent'")
            feats = model.encode(batch)  # (B, C, H, W)
            feats = feats.flatten(start_dim=1)  # (B, D)
        elif space == "pixel":
            feats = batch.flatten(start_dim=1)  # (B, C*H*W)
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
def compute_nearest_neighbor_distances(queries, targets, chunk_size=256, device="cpu"):
    """Compute Euclidean distance from each query to its nearest neighbor in targets.

    d(q, targets) = min_{t in targets} ||q - t||_2

    Parameters
    ----------
    queries : torch.Tensor
        Query vectors (Q, D).
    targets : torch.Tensor
        Target vectors (T, D).
    chunk_size : int
        Chunk size along queries to avoid O(Q * T) memory allocation.
    device : torch.device
        Device used for matrix distance calculations.

    torch.Tensor
        1D tensor (Q,) of min Euclidean distances.
    """
    q_count = queries.size(0)
    min_distances = torch.zeros(q_count, dtype=torch.float32)

    targets_dev = targets.to(device)
    target_norms_sq = (targets_dev ** 2).sum(dim=1)  # (T,)

    for i in range(0, q_count, chunk_size):
        end = min(i + chunk_size, q_count)
        q_chunk = queries[i:end].to(device)  # (B, D)

        # ||q - t||^2 = ||q||^2 - 2 q.t + ||t||^2
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

    The U statistic and Z-score are always computed from ranks, for sample x,
    so that their sign is well defined (the statistic scipy reports for the
    two-sided test has changed between versions). scipy, if installed, is used
    only for the p-values.
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
    """Execute the Meehan et al. Three-Sample Non-Parametric Test for Data-Copying.

    Datasets:
      - Train (T): samples used to train the generative model
      - Test  (P): held-out real validation samples from the true distribution
      - Gen   (Q): synthetic samples produced by the generative model

    Hypothesis:
      - H0: Q is not data-copying (its nearest-neighbour distances to T match
            those of independent test samples P).
      - H1: Q is data-copying (generated points are systematically closer to T
            than real test points are).

    d_test : torch.Tensor, optional
        Precomputed test->train nearest-neighbour distances. Does not depend on
        the UNet, so it is computed once and reused across checkpoints.

    dict
        Distance metrics, U-statistic, Z_U score, p-values, and verbal interpretation.
    """
    if d_test is None:
        d_test = compute_nearest_neighbor_distances(test_feats, train_feats, device=device)
    d_gen = compute_nearest_neighbor_distances(gen_feats, train_feats, device=device)

    d_test_np = d_test.numpy()
    d_gen_np = d_gen.numpy()

    # Mann-Whitney test comparing D_Q vs D_P
    test_res = mann_whitney_u_test(d_gen_np, d_test_np)

    z_u = test_res["z_score"]
    p_two = test_res["p_value_two_sided"]
    p_copy = test_res["p_value_copying_one_sided"]

    # Z_U < -1.96 means generated distances are significantly smaller than test distances
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


# Visualisation & Reporting
def plot_evaluation(ssim_res, meehan_res, gen_images, output_dir, title_prefix=""):
    """Generate diagnostic figures for SSIM and the Meehan Three-Sample test."""
    output_dir = Path(output_dir)

    # Meehan Three-Sample Test Distance Distribution Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    d_test = np.array(meehan_res["dist_test_to_train"])
    d_gen = np.array(meehan_res["dist_gen_to_train"])

    bins = np.histogram_bin_edges(np.concatenate([d_test, d_gen]), bins=30)
    ax1.hist(d_test, bins=bins, alpha=0.6, label=f"Test → Train (N={len(d_test)})", color="royalblue", density=True)
    ax1.hist(d_gen, bins=bins, alpha=0.6, label=f"Synthetic → Train (N={len(d_gen)})", color="crimson", density=True)
    ax1.axvline(d_test.mean(), color="darkblue", linestyle="--", label=f"Mean Test ({d_test.mean():.2f})")
    ax1.axvline(d_gen.mean(), color="darkred", linestyle="--", label=f"Mean Gen ({d_gen.mean():.2f})")
    ax1.set_xlabel("Nearest-Neighbor Distance to Training Set")
    ax1.set_ylabel("Density")
    ax1.set_title(f"{title_prefix}Meehan et al. Distance Distributions\n"
                  f"$Z_U = {meehan_res['z_u_score']:.3f}$ ({meehan_res['status']})")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.boxplot([d_test, d_gen], tick_labels=["Test → Train", "Synthetic → Train"], patch_artist=True,
                boxprops=dict(facecolor="lightblue", color="blue"),
                medianprops=dict(color="black", linewidth=1.5))
    ax2.set_ylabel("Nearest-Neighbor Euclidean Distance")
    ax2.set_title("Distance Comparison (Median & IQR)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "meehan_three_sample_test.png", dpi=150)
    plt.close(fig)

    # SSIM Distribution Plot
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

    # Grid of Generated Samples
    n_display = min(16, gen_images.size(0))
    grid = make_grid(gen_images[:n_display].clamp(0, 1), nrow=4, padding=4, normalize=False)
    save_image(grid, output_dir / "generated_samples_grid.png")


SUMMARY_FIELDS = [
    "checkpoint", "epoch",
    "best_match_ssim_mean", "best_match_ssim_std", "mean_pairwise_ssim",
    "z_u_score", "p_value_copying_one_sided", "status",
    "mean_dist_gen_to_train", "mean_dist_test_to_train",
    "generation_seconds", "error",
]


def write_summary(records, output_dir):
    """Write summary JSON, CSV and trend plot for all checkpoints evaluated so far."""
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

    if all(r["epoch"] is not None for r in ok):
        x, xlabel = [r["epoch"] for r in ok], "Checkpoint epoch"
    else:
        x, xlabel = list(range(len(ok))), "Checkpoint index"

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    ax[0, 0].plot(x, [r["best_match_ssim_mean"] for r in ok], marker="o")
    ax[0, 0].set_title("Mean Best-Match SSIM vs Test Set")
    ax[0, 0].set_ylabel("SSIM")

    ax[0, 1].plot(x, [r["mean_pairwise_ssim"] for r in ok], marker="o", color="darkorange")
    ax[0, 1].set_title("Mean Pairwise SSIM vs Test Set")
    ax[0, 1].set_ylabel("SSIM")

    ax[1, 0].plot(x, [r["z_u_score"] for r in ok], marker="o", color="crimson")
    ax[1, 0].axhline(-1.96, color="gray", linestyle="--", label="±1.96")
    ax[1, 0].axhline(1.96, color="gray", linestyle="--")
    ax[1, 0].axhline(0, color="black", linewidth=0.8)
    ax[1, 0].set_title("Meehan $Z_U$ (negative = copying)")
    ax[1, 0].set_ylabel("$Z_U$")
    ax[1, 0].legend()

    ax[1, 1].plot(x, [r["mean_dist_gen_to_train"] for r in ok], marker="o", color="green",
                  label="Synthetic → Train")
    ax[1, 1].axhline(ok[0]["mean_dist_test_to_train"], color="royalblue", linestyle="--",
                     label="Test → Train")
    ax[1, 1].set_title("Mean Nearest-Neighbor Distance to Train")
    ax[1, 1].set_ylabel("Distance")
    ax[1, 1].legend()

    for a in ax.flat:
        a.set_xlabel(xlabel)
        a.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "summary_trends.png", dpi=150)
    plt.close(fig)


# Per-checkpoint evaluation
def ensure_reference_cache(cache, model, train_loader, test_loader, args, device):
    """Compute everything that is independent of the UNet, once.

    Needs a loaded model only because latent-space features go through the
    (shared) VQ-VAE encoder.
    """
    if "test_images" not in cache:
        print("Loading test images for SSIM …")
        cache["test_images"] = load_images(test_loader)

    if "d_test" not in cache:
        print("Extracting training representations …")
        cache["train_feats"] = extract_representations(
            train_loader, model, args.metric_space, args.max_train_samples, device)
        print("Extracting test representations …")
        cache["test_feats"] = extract_representations(
            test_loader, model, args.metric_space, args.max_test_samples, device)
        print("Computing test → train nearest-neighbor distances …")
        cache["d_test"] = compute_nearest_neighbor_distances(
            cache["test_feats"], cache["train_feats"], device=device)


def evaluate_checkpoint(unet_ckpt, args, device, train_loader, test_loader, cache):
    """Generate samples from one UNet checkpoint and compute all metrics."""
    label = unet_ckpt.stem
    out_dir = args.output_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)

    model = StableDiffusion.from_checkpoints(
        vqvae_ckpt=args.vqvae_ckpt,
        unet_ckpt=unet_ckpt,
        image_size=args.image_size,
        device=device,
        train_vqvae=False,
        train_unet=False,
    )

    try:
        ensure_reference_cache(cache, model, train_loader, test_loader, args, device)

        # Generate (same seed for every checkpoint)
        print(f"Generating {args.num_samples} samples (DDIM steps={args.ddim_steps}) …")
        start = time.perf_counter()
        gen_images = model.sample(
            n=args.num_samples,
            ddim_steps=args.ddim_steps,
            batch_size=args.sample_batch_size,
            seed=args.seed,
        )
        gen_elapsed = time.perf_counter() - start
        gen_images = gen_images.clamp(0, 1).cpu()
        print(f"  done in {gen_elapsed:.1f}s ({args.num_samples / gen_elapsed:.2f} img/s)")

        if args.save_all_samples:
            samples_dir = out_dir / "samples"
            samples_dir.mkdir(exist_ok=True)
            for idx in range(gen_images.size(0)):
                save_image(gen_images[idx], samples_dir / f"sample_{idx:05d}.png")

        # Evaluation 1: SSIM
        ssim_results = compute_ssim_against_test(gen_images, cache["test_images"], device)
        print(f"  best-match SSIM: {ssim_results['best_match_ssim_mean']:.4f} "
              f"± {ssim_results['best_match_ssim_std']:.4f} | "
              f"pairwise: {ssim_results['mean_pairwise_ssim']:.4f}")

        # Evaluation 2: Meehan test
        if args.metric_space == "latent":
            with torch.no_grad():
                gen_feats = torch.cat([
                    model.encode(gen_images[i:i + args.batch_size].to(device)).flatten(start_dim=1).cpu()
                    for i in range(0, gen_images.size(0), args.batch_size)
                ], dim=0)
        else:
            gen_feats = gen_images.flatten(start_dim=1)

        meehan_results = meehan_three_sample_test(
            train_feats=cache["train_feats"],
            test_feats=cache["test_feats"],
            gen_feats=gen_feats,
            device=device,
            d_test=cache["d_test"],
        )
        print(f"  Z_U: {meehan_results['z_u_score']:+.3f} "
              f"(p_copy={meehan_results['p_value_copying_one_sided']:.3e}) "
              f"→ {meehan_results['status']}")

        # Persist
        plot_evaluation(ssim_results, meehan_results, gen_images, out_dir, title_prefix=f"{label}\n")

        record = {
            "checkpoint": label,
            "epoch": parse_epoch(unet_ckpt),
            "path": str(unet_ckpt),
            "generation_seconds": gen_elapsed,
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

        with open(out_dir / "diffusion_eval_metrics.json", "w") as f:
            json.dump({
                "config": {
                    "vqvae_ckpt": str(args.vqvae_ckpt),
                    "unet_ckpt": str(unet_ckpt),
                    "num_samples": args.num_samples,
                    "ddim_steps": args.ddim_steps,
                    "metric_space": args.metric_space,
                    "seed": args.seed,
                },
                "metrics": record,
            }, f, indent=2)

        return record

    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# CLI & Execution
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate many UNet checkpoints of a Latent Diffusion Model "
                    "(SSIM & Meehan et al. Data-Copying Test) for one VQ-VAE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Checkpoints
    parser.add_argument("--vqvae-ckpt", type=Path, required=True,
                        help="Path to the (single, shared) trained VQ-VAE checkpoint (.pt)")
    parser.add_argument("--unet-ckpt-dir", type=Path, required=True,
                        help="Directory containing UNet diffusion checkpoints (.pt)")
    parser.add_argument("--unet-glob", type=str, default="*unet*.pt",
                        help="Glob (relative to --unet-ckpt-dir) selecting UNet checkpoints. "
                             "Files are evaluated in order of the trailing number in their name.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Evaluate every k-th checkpoint (the last one is always included)")

    # Data directories
    parser.add_argument("--train-data-dir", type=Path, default=None,
                        help="Root directory of training images (default: <data-dir>/keras_png_slices_train)")
    parser.add_argument("--test-data-dir", type=Path, default=None,
                        help="Root directory of test images (default: <data-dir>/keras_png_slices_test)")
    parser.add_argument("--data-dir", type=Path, default=Path("./keras_png_slices_data"),
                        help="Parent dataset directory containing train and test folders")

    # Generation parameters
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of synthetic samples to generate per checkpoint")
    parser.add_argument("--sample-batch-size", type=int, default=16,
                        help="Batch size during diffusion generation")
    parser.add_argument("--ddim-steps", type=int, default=50,
                        help="Number of DDIM steps used for sampling")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed; identical for every checkpoint so initial noise matches")

    # Evaluation configuration
    parser.add_argument("--image-size", type=int, default=256,
                        help="Spatial resolution of images")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for dataloaders")
    parser.add_argument("--metric-space", type=str, default="latent",
                        choices=["latent", "pixel"],
                        help="Feature space for Meehan test ('latent' or 'pixel')")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Optional cap on training samples for nearest neighbor distance")
    parser.add_argument("--max-test-samples", type=int, default=None,
                        help="Optional cap on test samples for nearest neighbor distance")
    parser.add_argument("--output-dir", type=Path, default=Path("./eval_diffusion_results"),
                        help="Directory for saving metrics and diagnostic plots")
    parser.add_argument("--save-all-samples", action="store_true",
                        help="Save all individual generated images as PNGs (per checkpoint)")

    return parser.parse_args()


def resolve_data_dirs(args):
    """Resolve train and test data directories with sensible defaults."""
    train_dir = args.train_data_dir
    test_dir = args.test_data_dir

    if train_dir is None:
        candidate = args.data_dir / "keras_png_slices_train"
        train_dir = candidate if candidate.is_dir() else args.data_dir

    if test_dir is None:
        candidate = args.data_dir / "keras_png_slices_test"
        test_dir = candidate if candidate.is_dir() else args.data_dir

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training data directory not found: {train_dir}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Testing data directory not found: {test_dir}")

    return train_dir, test_dir


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dir, test_dir = resolve_data_dirs(args)

    unet_ckpts = find_unet_checkpoints(args.unet_ckpt_dir, args.unet_glob, args.stride)
    print(f"VQ-VAE: {args.vqvae_ckpt}")
    print(f"Found {len(unet_ckpts)} UNet checkpoint(s) in {args.unet_ckpt_dir}:")
    for p in unet_ckpts:
        print(f"  {p.name}")

    # Dataloaders (shared across checkpoints)
    train_loader = get_dataloader(
        str(train_dir), image_size=args.image_size, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(
        str(test_dir), image_size=args.image_size, batch_size=args.batch_size, shuffle=False)
    print(f"Training images: {len(train_loader.dataset)} | Testing images: {len(test_loader.dataset)}")

    cache = {}     # UNet-independent reference data, filled on first successful load
    records = []

    for i, ckpt in enumerate(unet_ckpts, start=1):
        print(f"\n{'═' * 66}")
        print(f"[{i}/{len(unet_ckpts)}] {ckpt.name}")
        print(f"{'═' * 66}")
        try:
            records.append(evaluate_checkpoint(ckpt, args, device, train_loader, test_loader, cache))
        except Exception as exc:  # keep going if one checkpoint is bad
            traceback.print_exc()
            records.append({"checkpoint": ckpt.stem, "epoch": parse_epoch(ckpt), "error": repr(exc)})

        write_summary(records, args.output_dir)  # persist progress after every checkpoint

    # Final summary table
    print(f"\n{'═' * 78}")
    print(f"{'EVALUATION SUMMARY':^78}")
    print(f"{'═' * 78}")
    print(f"{'checkpoint':<32}{'epoch':>6}{'best SSIM':>11}{'pair SSIM':>11}{'Z_U':>9}  status")
    for r in records:
        epoch = "-" if r["epoch"] is None else str(r["epoch"])
        if "error" in r:
            print(f"{r['checkpoint']:<32}{epoch:>6}  FAILED: {r['error'][:40]}")
        else:
            print(f"{r['checkpoint']:<32}{epoch:>6}{r['best_match_ssim_mean']:>11.4f}"
                  f"{r['mean_pairwise_ssim']:>11.4f}{r['z_u_score']:>+9.2f}  {r['status']}")
    print(f"{'═' * 78}")
    print(f"Results written to {args.output_dir}\n")


if __name__ == "__main__":
    main()