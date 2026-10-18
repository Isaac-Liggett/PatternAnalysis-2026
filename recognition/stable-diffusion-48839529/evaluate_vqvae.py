"""Evaluate one or more VQ-VAE checkpoints against a test dataset.

Metrics reported per checkpoint
-------------------------------
* **SSIM** — Structural Similarity Index between original and reconstructed
  images, averaged over the entire test set.
* **Training stability** — loss recorded in the checkpoint at save time,
  plotted across the supplied checkpoints to visualise convergence.
* **Generation runtime** — wall-clock time for a full encode → quantise →
  decode pass, reported as total seconds and per-image throughput.

Usage
-----
    python evaluate_vqvae.py \
        --checkpoint-dir data/checkpoints \
        --data-dir       ./keras_png_slices_data/keras_png_slices_test \
        --image-size     256 \
        --batch-size     16 \
        --output-dir     ./eval_results
"""

from pathlib import Path
import argparse
import json
import time

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from vqvae import VQModel, codebook_stats
from dataset import get_dataloader


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
    """Compute per-image SSIM between batches *x* and *y* (B, C, H, W).

    Both tensors are expected in [0, 1].  Returns a tensor of shape (B,).
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
    return ssim_map.mean(dim=(1, 2, 3))  # average over C, H, W → (B,)


# ---------------------------------------------------------------------------
# Evaluation for a single checkpoint
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_checkpoint(ckpt_path, dataloader, device):
    """Load a VQ-VAE checkpoint and evaluate it on the full dataloader.

    Returns a dict of scalar metrics.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]

    model = VQModel(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_embed = config.get("n_embed", 8192)

    total_ssim = 0.0
    total_recon = 0.0
    total_qloss = 0.0
    n_images = 0
    all_idx = []

    start_time = time.perf_counter()

    for batch in dataloader:
        batch = batch.to(device)
        x_rec, qloss, idx = model(batch)

        # SSIM expects [0, 1] — our images are already in [0, 1] via ToTensor.
        batch_ssim = ssim_batch(batch, x_rec.clamp(0, 1))
        total_ssim += batch_ssim.sum().item()
        total_recon += F.l1_loss(x_rec, batch, reduction="sum").item()
        total_qloss += qloss.item() * batch.size(0)
        n_images += batch.size(0)
        all_idx.append(idx)

    elapsed = time.perf_counter() - start_time

    # Codebook statistics over the full test set.
    all_idx = torch.cat([i.flatten() for i in all_idx])
    usage, perplexity = codebook_stats(all_idx, n_embed)

    return {
        "checkpoint": str(ckpt_path),
        "epoch": ckpt.get("epoch", -1),
        "train_loss": ckpt.get("loss", float("nan")),
        "mean_ssim": total_ssim / n_images,
        "mean_recon_l1": total_recon / n_images,
        "mean_qloss": total_qloss / n_images,
        "codebook_usage": usage,
        "codebook_perplexity": perplexity,
        "total_runtime_s": elapsed,
        "images_per_sec": n_images / elapsed,
        "n_images": n_images,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, output_dir):
    """Create summary plots from a list of per-checkpoint result dicts."""
    epochs = [r["epoch"] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes[1, 0].set_visible(False)  # unused panel

    # 1. SSIM across checkpoints
    ax = axes[0, 0]
    ax.plot(epochs, [r["mean_ssim"] for r in results], marker="o")
    ax.set_ylabel("Mean SSIM")
    ax.set_title("Reconstruction Quality (SSIM ↑)")

    # 2. Training loss (stability)
    ax = axes[0, 1]
    ax.plot(epochs, [r["train_loss"] for r in results], marker="o", color="darkorange")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Stability (Loss ↓)")

    # 3. Reconstruction L1
    ax = axes[0, 2]
    ax.plot(epochs, [r["mean_recon_l1"] for r in results], marker="o", color="green")
    ax.set_ylabel("Mean L1 Reconstruction")
    ax.set_title("Reconstruction L1 ↓")

    # 4. Throughput / runtime
    ax = axes[1, 1]
    ax.plot(epochs, [r["images_per_sec"] for r in results], marker="s", color="purple")
    ax.set_ylabel("Images / sec")
    ax.set_title("Inference Throughput")

    # 5. Codebook utilisation
    ax = axes[1, 2]
    ax.plot(epochs, [r["codebook_usage"] for r in results], marker="^",
            color="teal", label="Usage frac")
    ax.set_ylabel("Usage fraction")
    ax.set_ylim(0, 1.05)
    ax2 = ax.twinx()
    ax2.plot(epochs, [r["codebook_perplexity"] for r in results], marker="v",
             color="crimson", label="Perplexity")
    ax2.set_ylabel("Perplexity")
    ax.set_title("Codebook Health")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    for a in axes.flat:
        a.set_xlabel("Epoch")
        a.grid(True, alpha=0.3)

    fig.suptitle("VQ-VAE Evaluation Summary", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "vqvae_eval_summary.png", dpi=150)
    plt.close(fig)
    print(f"Saved plot → {output_dir / 'vqvae_eval_summary.png'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate VQ-VAE checkpoints on a test set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, required=True,
        help="Directory containing VQ-VAE checkpoint (.pt) files",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("./keras_png_slices_data/keras_png_slices_test"),
        help="Root directory of the test images",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./eval_results"),
        help="Directory for outputs (plots, JSON metrics)",
    )
    return parser.parse_args()


def scan_checkpoints(directory: Path):
    """Find valid VQ-VAE checkpoint files in *directory*.

    A file is considered a valid VQ-VAE checkpoint if it is a ``.pt`` file
    that can be loaded and contains a ``config`` and ``model_state_dict`` key.
    Files that fail to load or lack the expected keys are skipped with a
    warning.

    Returns a list of ``Path`` objects sorted by filename.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {directory}")

    candidates = sorted(directory.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt files found in {directory}")

    valid = []
    for p in candidates:
        try:
            ckpt = torch.load(p, map_location="cpu")
            if "config" in ckpt and "model_state_dict" in ckpt:
                valid.append(p)
            else:
                print(f"  ⚠ Skipping {p.name}: missing 'config' or "
                      f"'model_state_dict' keys (not a VQ-VAE checkpoint?)")
        except Exception as e:
            print(f"  ⚠ Skipping {p.name}: failed to load ({e})")

    if not valid:
        raise RuntimeError(
            f"No valid VQ-VAE checkpoints found in {directory}. "
            f"Expected .pt files with 'config' and 'model_state_dict' keys."
        )

    return valid


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataloader = get_dataloader(
        str(args.data_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    print(f"Test set: {len(dataloader.dataset)} images from {args.data_dir}")

    # Scan checkpoint directory for valid VQ-VAE checkpoints.
    print(f"\nScanning {args.checkpoint_dir} for VQ-VAE checkpoints …")
    ckpt_paths = scan_checkpoints(args.checkpoint_dir)
    print(f"Found {len(ckpt_paths)} valid checkpoint(s).")

    results = []
    for ckpt_path in ckpt_paths:
        print(f"\n{'─' * 50}")
        print(f"Evaluating: {ckpt_path}")
        metrics = evaluate_checkpoint(ckpt_path, dataloader, device)
        results.append(metrics)
        print(f"  Epoch {metrics['epoch']:>3d}  |  "
              f"SSIM {metrics['mean_ssim']:.4f}  |  "
              f"L1 {metrics['mean_recon_l1']:.4f}  |  "
              f"{metrics['images_per_sec']:.1f} img/s")

    # Persist raw metrics as JSON.
    json_path = args.output_dir / "vqvae_eval_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics → {json_path}")

    # Summary plots (only meaningful with ≥ 2 checkpoints).
    if len(results) >= 2:
        plot_results(results, args.output_dir)

    # Print final summary table.
    print(f"\n{'═' * 62}")
    print(f"{'Epoch':>6}  {'SSIM':>8}  {'L1':>8}  {'Q-Loss':>8}  "
          f"{'img/s':>7}  {'Usage':>7}")
    print(f"{'─' * 62}")
    for r in results:
        print(f"{r['epoch']:>6d}  {r['mean_ssim']:>8.4f}  "
              f"{r['mean_recon_l1']:>8.4f}  {r['mean_qloss']:>8.4f}  "
              f"{r['images_per_sec']:>7.1f}  {r['codebook_usage']:>6.1%}")
    print(f"{'═' * 62}")


if __name__ == "__main__":
    main()
