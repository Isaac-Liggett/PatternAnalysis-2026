"""Evaluate a Latent Diffusion Model (Stable Diffusion).

Computes:
1. **SSIM against Test Dataset** — Evaluates structural similarity of generated
   synthetic samples compared with held-out real test images (mean nearest-neighbor
   best-match SSIM and pairwise SSIM).
2. **Meehan et al. Three-Sample Non-Parametric Test** — Tests whether the
   diffusion model is merely copying / memorising training samples by comparing
   nearest-neighbour distance distributions from generated samples to the training
   set against test samples to the training set (Meehan et al., AISTATS 2020).
3. **Visualisations & Metrics Persistence** — Generates diagnostic plots (distance
   distributions, SSIM histograms, synthetic visual grid) and dumps all metrics
   to JSON.

Usage:
------
    python evaluate_diffusion.py \
        --vqvae-ckpt   ./data/checkpoints/vqvae_epoch_50.pt \
        --unet-ckpt    ./data/checkpoints/unet_epoch_100.pt \
        --train-data-dir ./keras_png_slices_data/keras_png_slices_train \
        --test-data-dir  ./keras_png_slices_data/keras_png_slices_test \
        --num-samples  100 \
        --output-dir   ./eval_diffusion_results
"""

from pathlib import Path
import argparse
import json
import math
import time

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

from diffusion_model import StableDiffusion
from dataset import get_dataloader, ImageFolderDataset


# ---------------------------------------------------------------------------
# SSIM (pure PyTorch, self-contained)
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


# ---------------------------------------------------------------------------
# SSIM Evaluation: Generated Samples vs Test Dataset
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_ssim_against_test(gen_images, test_loader, device, chunk_size=32):
    """Compute SSIM metrics comparing generated samples against the test set.

    For each generated image, we find its best structural match (max SSIM)
    across the entire test dataset, as well as the average similarity.

    Parameters
    ----------
    gen_images : torch.Tensor
        Generated images (M, C, H, W) in range [0, 1].
    test_loader : DataLoader
        DataLoader providing real test images in range [0, 1].
    device : torch.device
        Device for computation.
    chunk_size : int
        Batch size of generated images processed at once to manage GPU memory.

    Returns
    -------
    dict
        SSIM statistics (mean, std, min, max best-match SSIM, and mean pairwise SSIM).
    """
    print("Computing SSIM of generated images against test dataset …")
    gen_images = gen_images.clamp(0, 1)
    m = gen_images.size(0)

    # Collect all test images into a single tensor on CPU or device
    test_images_list = []
    for batch in test_loader:
        test_images_list.append(batch.clamp(0, 1))
    test_images = torch.cat(test_images_list, dim=0)  # (N, C, H, W)
    n = test_images.size(0)

    best_ssim_per_gen = torch.full((m,), -1.0, dtype=torch.float32)
    best_match_idx = torch.zeros((m,), dtype=torch.long)
    total_pairwise_ssim = 0.0

    # Compare in chunks over generated images and batches over test images
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

            # Pairwise broadcast: repeat g_chunk and t_chunk to compute SSIM
            # g_exp: (Bg * Bt, C, H, W)
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

    mean_best = best_ssim_per_gen.mean().item()
    std_best = best_ssim_per_gen.std().item()
    min_best = best_ssim_per_gen.min().item()
    max_best = best_ssim_per_gen.max().item()
    mean_pairwise = total_pairwise_ssim / (m * n)

    return {
        "best_match_ssim_mean": mean_best,
        "best_match_ssim_std": std_best,
        "best_match_ssim_min": min_best,
        "best_match_ssim_max": max_best,
        "mean_pairwise_ssim": mean_pairwise,
        "per_sample_best_ssim": best_ssim_per_gen.tolist(),
        "best_match_indices": best_match_idx.tolist(),
    }


# ---------------------------------------------------------------------------
# Meehan et al. Three-Sample Non-Parametric Test for Data-Copying
# (AISTATS 2020: "A Non-Parametric Test to Detect Data-Copying in Generative Models")
# ---------------------------------------------------------------------------

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

    Returns
    -------
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

    Returns
    -------
    torch.Tensor
        1D tensor (Q,) of min Euclidean distances.
    """
    q_count = queries.size(0)
    t_count = targets.size(0)
    min_distances = torch.zeros(q_count, dtype=torch.float32)

    targets_dev = targets.to(device)
    # Target norms squared: (T,)
    target_norms_sq = (targets_dev ** 2).sum(dim=1)

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


def mann_whitney_u_test(x, y):
    """Perform the two-sample Mann-Whitney U test between 1D arrays x and y.

    Computes U statistic, standard normal Z-score, and p-values without
    requiring external libraries, with exact tie handling.

    Tests whether x is stochastically smaller than y (data-copying alternative).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    m = len(y)

    if n == 0 or m == 0:
        raise ValueError("Inputs x and y must both be non-empty.")

    # Try scipy.stats first if available for high-precision p-values
    try:
        from scipy import stats
        res_two = stats.mannwhitneyu(x, y, alternative="two-sided")
        res_less = stats.mannwhitneyu(x, y, alternative="less")

        u_stat = res_two.statistic
        mean_u = (n * m) / 2.0
        std_u = math.sqrt((n * m * (n + m + 1)) / 12.0)
        z_score = (u_stat - mean_u) / std_u if std_u > 0 else 0.0

        return {
            "u_statistic": float(u_stat),
            "z_score": float(z_score),
            "p_value_two_sided": float(res_two.pvalue),
            "p_value_copying_one_sided": float(res_less.pvalue),
            "n_x": n,
            "n_y": m,
        }
    except ImportError:
        pass

    # Pure Python / NumPy implementation of Mann-Whitney U & Z-score
    # Combine and rank
    combined = np.concatenate([x, y])
    order = np.argsort(combined)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1)

    # Handle ties by assigning average rank
    sorted_comb = combined[order]
    unique_vals, inverse, counts = np.unique(sorted_comb, return_inverse=True, return_counts=True)
    if len(unique_vals) < len(combined):
        for val_idx, count in enumerate(counts):
            if count > 1:
                match_indices = np.where(sorted_comb == unique_vals[val_idx])[0]
                avg_rank = (match_indices + 1).mean()
                ranks[order[match_indices]] = avg_rank

    r_x = np.sum(ranks[:n])
    u_x = r_x - (n * (n + 1)) / 2.0
    mean_u = (n * m) / 2.0
    std_u = math.sqrt((n * m * (n + m + 1)) / 12.0)

    z_score = (u_x - mean_u) / std_u if std_u > 0 else 0.0

    # Normal CDF using erf
    p_less = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
    p_two = 2.0 * min(p_less, 1.0 - p_less)

    return {
        "u_statistic": float(u_x),
        "z_score": float(z_score),
        "p_value_two_sided": float(p_two),
        "p_value_copying_one_sided": float(p_less),
        "n_x": n,
        "n_y": m,
    }


def meehan_three_sample_test(train_feats, test_feats, gen_feats, device="cpu"):
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

    Returns
    -------
    dict
        Distance metrics, U-statistic, Z_U score, p-values, and verbal interpretation.
    """
    print("\nComputing nearest-neighbor distances to training set (Meehan et al. Test) …")

    # 1. Distances from Test samples P to nearest neighbour in Train T: D_P
    d_test = compute_nearest_neighbor_distances(test_feats, train_feats, device=device)

    # 2. Distances from Generated samples Q to nearest neighbour in Train T: D_Q
    d_gen = compute_nearest_neighbor_distances(gen_feats, train_feats, device=device)

    d_test_np = d_test.numpy()
    d_gen_np = d_gen.numpy()

    # 3. Mann-Whitney test comparing D_Q vs D_P
    test_res = mann_whitney_u_test(d_gen_np, d_test_np)

    z_u = test_res["z_score"]
    p_two = test_res["p_value_two_sided"]
    p_copy = test_res["p_value_copying_one_sided"]

    # Interpretation
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


# ---------------------------------------------------------------------------
# Visualisation & Reporting
# ---------------------------------------------------------------------------

def plot_evaluation(ssim_res, meehan_res, gen_images, output_dir):
    """Generate diagnostic figures for SSIM and the Meehan Three-Sample test."""
    output_dir = Path(output_dir)

    # 1. Meehan Three-Sample Test Distance Distribution Plot
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
    ax1.set_title(f"Meehan et al. Distance Distributions\n$Z_U = {meehan_res['z_u_score']:.3f}$ ({meehan_res['status']})")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Boxplot comparison
    ax2.boxplot([d_test, d_gen], tick_labels=["Test → Train", "Synthetic → Train"], patch_artist=True,
                boxprops=dict(facecolor="lightblue", color="blue"),
                medianprops=dict(color="black", linewidth=1.5))
    ax2.set_ylabel("Nearest-Neighbor Euclidean Distance")
    ax2.set_title("Distance Comparison (Median & IQR)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = output_dir / "meehan_three_sample_test.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved distance distribution plot → {fig_path}")

    # 2. SSIM Distribution Plot
    best_ssims = np.array(ssim_res["per_sample_best_ssim"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(best_ssims, bins=25, color="teal", edgecolor="black", alpha=0.7)
    ax.axvline(ssim_res["best_match_ssim_mean"], color="red", linestyle="--",
               label=f"Mean Best SSIM: {ssim_res['best_match_ssim_mean']:.4f}")
    ax.set_xlabel("Best-Match SSIM against Test Set")
    ax.set_ylabel("Count of Generated Images")
    ax.set_title("Synthetic Image SSIM against Test Dataset")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    ssim_fig_path = output_dir / "ssim_distribution.png"
    fig.savefig(ssim_fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved SSIM distribution plot → {ssim_fig_path}")

    # 3. Grid of Generated Samples
    n_display = min(16, gen_images.size(0))
    grid = make_grid(gen_images[:n_display].clamp(0, 1), nrow=4, padding=4, normalize=False)
    grid_path = output_dir / "generated_samples_grid.png"
    save_image(grid, grid_path)
    print(f"Saved generated sample grid → {grid_path}")


# ---------------------------------------------------------------------------
# CLI & Execution
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Latent Diffusion Model (SSIM & Meehan et al. Data-Copying Test).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Checkpoints
    parser.add_argument("--vqvae-ckpt", type=Path, required=True,
                        help="Path to trained VQ-VAE checkpoint (.pt)")
    parser.add_argument("--unet-ckpt", type=Path, required=True,
                        help="Path to trained UNet diffusion checkpoint (.pt)")

    # Data directories
    parser.add_argument("--train-data-dir", type=Path, default=None,
                        help="Root directory of training images (default: <data-dir>/keras_png_slices_train)")
    parser.add_argument("--test-data-dir", type=Path, default=None,
                        help="Root directory of test images (default: <data-dir>/keras_png_slices_test)")
    parser.add_argument("--data-dir", type=Path, default=Path("./keras_png_slices_data"),
                        help="Parent dataset directory containing train and test folders")

    # Generation parameters
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Arbitrary number of synthetic samples to generate")
    parser.add_argument("--sample-batch-size", type=int, default=16,
                        help="Batch size during diffusion generation")
    parser.add_argument("--ddim-steps", type=int, default=50,
                        help="Number of DDIM steps used for sampling")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for generation reproducibility")

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
                        help="Save all individual generated synthetic images as PNGs")

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

    print(f"Loading StableDiffusion from checkpoints:")
    print(f"  VQ-VAE: {args.vqvae_ckpt}")
    print(f"  UNet:   {args.unet_ckpt}")

    model = StableDiffusion.from_checkpoints(
        vqvae_ckpt=args.vqvae_ckpt,
        unet_ckpt=args.unet_ckpt,
        image_size=args.image_size,
        device=device,
        train_vqvae=False,
        train_unet=False,
    )

    # 1. Generate Synthetic Dataset
    print(f"\nGenerating {args.num_samples} synthetic images (DDIM steps={args.ddim_steps}) …")
    start_gen = time.perf_counter()
    gen_images = model.sample(
        n=args.num_samples,
        ddim_steps=args.ddim_steps,
        batch_size=args.sample_batch_size,
        seed=args.seed,
    )
    gen_elapsed = time.perf_counter() - start_gen
    gen_images = gen_images.clamp(0, 1)
    print(f"Generation completed in {gen_elapsed:.2f}s ({args.num_samples / gen_elapsed:.2f} images/s)")

    # Optionally save individual images
    if args.save_all_samples:
        samples_dir = args.output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        for idx in range(gen_images.size(0)):
            save_image(gen_images[idx], samples_dir / f"sample_{idx:05d}.png")
        print(f"Saved {gen_images.size(0)} individual images to {samples_dir}")

    # 2. Prepare Dataloaders for Train and Test sets
    print(f"\nLoading real datasets:")
    print(f"  Train directory: {train_dir}")
    print(f"  Test directory:  {test_dir}")

    train_loader = get_dataloader(
        str(train_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = get_dataloader(
        str(test_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f"  Training images: {len(train_loader.dataset)}")
    print(f"  Testing images:  {len(test_loader.dataset)}")

    # 3. Compute SSIM against Test Dataset
    print(f"\n{'─' * 60}")
    print("EVALUATION 1: SSIM AGAINST TEST DATASET")
    print(f"{'─' * 60}")
    ssim_results = compute_ssim_against_test(
        gen_images=gen_images,
        test_loader=test_loader,
        device=device,
    )

    print(f"Mean Best-Match SSIM: {ssim_results['best_match_ssim_mean']:.4f} ± {ssim_results['best_match_ssim_std']:.4f}")
    print(f"Min Best SSIM:        {ssim_results['best_match_ssim_min']:.4f}")
    print(f"Max Best SSIM:        {ssim_results['best_match_ssim_max']:.4f}")
    print(f"Mean Pairwise SSIM:   {ssim_results['mean_pairwise_ssim']:.4f}")

    # 4. Extract Representations for Meehan et al. Three-Sample Test
    print(f"\n{'─' * 60}")
    print(f"EVALUATION 2: MEEHAN ET AL. THREE-SAMPLE DATA-COPYING TEST ({args.metric_space.upper()} SPACE)")
    print(f"{'─' * 60}")

    print("Extracting training representations …")
    train_feats = extract_representations(
        dataloader=train_loader,
        model=model,
        space=args.metric_space,
        max_samples=args.max_train_samples,
        device=device,
    )

    print("Extracting test representations …")
    test_feats = extract_representations(
        dataloader=test_loader,
        model=model,
        space=args.metric_space,
        max_samples=args.max_test_samples,
        device=device,
    )

    print("Extracting synthetic representations …")
    if args.metric_space == "latent":
        gen_feats_list = []
        for i in range(0, gen_images.size(0), args.batch_size):
            batch = gen_images[i:i + args.batch_size].to(device)
            gen_feats_list.append(model.encode(batch).flatten(start_dim=1).cpu())
        gen_feats = torch.cat(gen_feats_list, dim=0)
    else:
        gen_feats = gen_images.flatten(start_dim=1)

    meehan_results = meehan_three_sample_test(
        train_feats=train_feats,
        test_feats=test_feats,
        gen_feats=gen_feats,
        device=device,
    )

    print(f"\nResults:")
    print(f"  Test → Train mean dist: {meehan_results['mean_dist_test_to_train']:.4f} ± {meehan_results['std_dist_test_to_train']:.4f}")
    print(f"  Gen  → Train mean dist: {meehan_results['mean_dist_gen_to_train']:.4f} ± {meehan_results['std_dist_gen_to_train']:.4f}")
    print(f"  Mann-Whitney U statistic: {meehan_results['u_statistic']:.1f}")
    print(f"  Meehan Z_U test statistic: {meehan_results['z_u_score']:+.4f}")
    print(f"  Two-sided p-value:         {meehan_results['p_value_two_sided']:.4e}")
    print(f"  Data-copying p-value:      {meehan_results['p_value_copying_one_sided']:.4e}")
    print(f"\nVERDICT: {meehan_results['verdict']}")

    # 5. Save Plots & Full Results
    plot_evaluation(ssim_results, meehan_results, gen_images, args.output_dir)

    metrics_dump = {
        "config": {
            "vqvae_ckpt": str(args.vqvae_ckpt),
            "unet_ckpt": str(args.unet_ckpt),
            "num_samples": args.num_samples,
            "ddim_steps": args.ddim_steps,
            "metric_space": args.metric_space,
            "seed": args.seed,
            "generation_runtime_seconds": gen_elapsed,
            "images_per_second": args.num_samples / gen_elapsed,
        },
        "ssim_metrics": {
            "best_match_ssim_mean": ssim_results["best_match_ssim_mean"],
            "best_match_ssim_std": ssim_results["best_match_ssim_std"],
            "best_match_ssim_min": ssim_results["best_match_ssim_min"],
            "best_match_ssim_max": ssim_results["best_match_ssim_max"],
            "mean_pairwise_ssim": ssim_results["mean_pairwise_ssim"],
        },
        "meehan_three_sample_test": {
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
        },
    }

    json_path = args.output_dir / "diffusion_eval_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_dump, f, indent=2)
    print(f"\nPersisted metrics JSON → {json_path}")

    # Summary table
    print(f"\n{'═' * 66}")
    print(f"{'EVALUATION SUMMARY':^66}")
    print(f"{'═' * 66}")
    print(f"  Generated Images:            {args.num_samples}")
    print(f"  Sampling Speed:              {args.num_samples / gen_elapsed:.1f} img/s")
    print(f"  Mean Best-Match SSIM:        {ssim_results['best_match_ssim_mean']:.4f}")
    print(f"  Mean Pairwise SSIM:          {ssim_results['mean_pairwise_ssim']:.4f}")
    print(f"  Meehan Z_U Score:            {meehan_results['z_u_score']:+.4f}")
    print(f"  Copying p-value:             {meehan_results['p_value_copying_one_sided']:.4e}")
    print(f"  Assessment:                  {meehan_results['status'].upper()}")
    print(f"{'═' * 66}\n")


if __name__ == "__main__":
    main()
