"""Sample n images from a dataset, run them through a trained VQ-VAE, and
display the originals alongside their reconstructions.

Usage:
    python sample_vqvae.py --checkpoint data/checkpoints/vqvae_epoch50.pt \
                           --data-dir ./keras_png_slices_data \
                           --n 8
"""

from pathlib import Path
import argparse
import random

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from vqvae import VQModel
from dataset import ImageFolderDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise VQ-VAE reconstructions on random dataset samples."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to a VQ-VAE checkpoint (.pt) saved by trainvqvae.py",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("./keras_png_slices_data"),
        help="Root directory of the image dataset",
    )
    parser.add_argument(
        "--n", type=int, default=8,
        help="Number of images to sample and reconstruct",
    )
    parser.add_argument(
        "--image-size", type=int, default=256,
        help="Resize images to this resolution (must match training)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--save", type=Path, default=None,
        help="If provided, save the figure to this path instead of displaying",
    )
    return parser.parse_args()


@torch.no_grad()
def reconstruct(model, images, device):
    """Run a batch through the VQ-VAE and return reconstructions + indices."""
    x = images.to(device)
    x_rec, qloss, idx = model(x)
    return x_rec.cpu(), qloss.item(), idx.cpu()


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint and rebuild model with saved config ──────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = VQModel(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  (loss={ckpt['loss']:.4f})")
    print(f"Model config: {config}")

    # ── Sample n random images from the dataset ─────────────────────────
    dataset = ImageFolderDataset(str(args.data_dir), image_size=args.image_size)
    n = min(args.n, len(dataset))
    indices = random.sample(range(len(dataset)), n)
    images = torch.stack([dataset[i] for i in indices])  # (n, C, H, W)

    # ── Reconstruct ─────────────────────────────────────────────────────
    x_rec, qloss, idx = reconstruct(model, images, device)
    per_image_l1 = F.l1_loss(x_rec, images, reduction="none").mean(dim=[1, 2, 3])

    print(f"Quantiser loss: {qloss:.4f}")
    print(f"Mean L1 reconstruction error: {per_image_l1.mean():.4f}")

    # ── Plot: top row = originals, bottom row = reconstructions ─────────
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5.5))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        orig = images[i].squeeze(0).numpy()       # (H, W) for greyscale
        recon = x_rec[i].squeeze(0).clamp(0, 1).numpy()

        axes[0, i].imshow(orig, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Original #{indices[i]}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(recon, cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"Recon  (L1={per_image_l1[i]:.4f})", fontsize=9)
        axes[1, i].axis("off")

    fig.suptitle(
        f"VQ-VAE Reconstructions  —  epoch {ckpt['epoch']}  |  "
        f"codebook K={config['n_embed']}  |  mean L1={per_image_l1.mean():.4f}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"Figure saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
