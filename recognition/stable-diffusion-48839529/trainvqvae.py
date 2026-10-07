from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from vqvae import VQModel, reset_dead_codes
from dataset import get_dataloader

BATCH_SIZE = 3
EPOCHS = 50
LR = 2e-4
IMAGE_SIZE = 256
IN_CHANNELS = 1
CH = 128
CH_MULT = (1, 2)   # m = 2 downsamplings -> f = 4 (256x256 input -> 64x64 latent)
N_RES = 1
N_EMBED = 1024     # codebook size K; my choice for a small single-channel dataset
EMBED_DIM = 3      # latent channels / codebook vector dimension
BETA = 0.25        # commitment weight (VQ-VAE paper), NOT a KL weight
LAM_Q = 1.0        # weight on the whole quantiser loss (codebook + commitment)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--ch", type=int, default=CH)
    parser.add_argument("--in-ch", type=float, default=IN_CHANNELS)
    parser.add_argument("--n-embed", type=int, default=N_EMBED)
    parser.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    parser.add_argument("--beta", type=float, default=BETA,
                        help="Commitment loss weight")
    parser.add_argument("--lam-q", type=float, default=LAM_Q)
    parser.add_argument(
        "--reset-dead", action="store_true",
        help="At each epoch end, re-initialise codes never used that epoch",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./keras_png_slices_data"),
        help="Path to the keras_png_slices_data folder",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data"),
        help="Path to the checkpoints and graphs parent folder",
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    return parser.parse_args()


def save_checkpoint(model, optimizer, epoch, loss, path, config):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config,  # lets stage 2 rebuild the exact architecture
    }, path)


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]


def plot_curves(recon_losses, q_losses, perplexities, usages, save_path):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    epochs = range(1, len(recon_losses) + 1)

    ax[0, 0].plot(epochs, recon_losses, marker="o")
    ax[0, 0].set_ylabel("Avg L1 reconstruction loss (per sample)")
    ax[0, 0].set_title("Reconstruction Loss")

    ax[0, 1].plot(epochs, q_losses, marker="o", color="darkorange")
    ax[0, 1].set_ylabel("Avg quantiser loss (codebook + commitment)")
    ax[0, 1].set_title("Quantiser Loss")

    ax[1, 0].plot(epochs, perplexities, marker="o", color="green")
    ax[1, 0].set_ylabel("Codebook perplexity (epoch)")
    ax[1, 0].set_title("Codebook Perplexity")

    ax[1, 1].plot(epochs, usages, marker="o", color="purple")
    ax[1, 1].set_ylabel("Fraction of codes used (epoch)")
    ax[1, 1].set_ylim(0, 1.05)
    ax[1, 1].set_title("Codebook Usage")

    for a in ax.flat:
        a.set_xlabel("Epoch")
        a.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def loss_function(x_rec, x, qloss, lam_q=1.0):
    """
    Returns (total_loss, recon_loss). Both are means (not sums); qloss from the
    quantiser is already codebook + beta * commitment. Log recon and qloss
    separately, and watch codebook perplexity/usage for codebook collapse
    (perplexity far below K, usage fraction near 0).
    """
    rec = F.l1_loss(x_rec, x)
    return rec + lam_q * qloss, rec


if __name__ == "__main__":
    print(f"Pytorch running on {device}")

    args = parse_args()

    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = OUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    loss_curve_path = args.output_dir / "loss_curve.png"

    dataloader = get_dataloader(
        str(args.data_dir / "keras_png_slices_train"),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=0
    )

    # Sanity check on the data: channel count and value range.
    sample = next(iter(dataloader))
    assert sample.shape[1] == IN_CHANNELS, f"expected {IN_CHANNELS} channels, got {sample.shape[1]}"
    print(f"Batch shape {tuple(sample.shape)}, value range "
          f"[{sample.min():.3f}, {sample.max():.3f}]")

    config = dict(ch=args.ch, ch_mult=CH_MULT, n_res=N_RES, n_embed=args.n_embed,
                  embed_dim=args.embed_dim, beta=args.beta, in_channel=args.in_ch)
    model = VQModel(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9))

    n_samples = len(dataloader.dataset)
    recon_losses, q_losses, perplexities, usages = [], [], [], []

    for epoch in tqdm(range(1, args.epochs + 1), desc="EPOCH"):
        model.train()
        epoch_recon, epoch_q = 0.0, 0.0
        counts = torch.zeros(args.n_embed, device=device)

        for data in tqdm(dataloader, desc="BATCH", leave=False):
            data = data.to(device)  # expects [N, C, H, W]

            x_rec, qloss, idx = model(data)
            loss, rec = loss_function(x_rec, data, qloss, args.lam_q)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = data.size(0)
            epoch_recon += rec.item() * bs
            epoch_q += qloss.item() * bs
            counts += torch.bincount(idx.flatten(), minlength=args.n_embed)

        avg_recon = epoch_recon / n_samples
        avg_q = epoch_q / n_samples
        p = counts / counts.sum()
        perplexity = torch.exp(-(p * torch.log(p + 1e-10)).sum()).item()
        usage = (counts > 0).float().mean().item()

        recon_losses.append(avg_recon)
        q_losses.append(avg_q)
        perplexities.append(perplexity)
        usages.append(usage)
        print(f"Epoch {epoch}/{args.epochs} | recon: {avg_recon:.4f}  q: {avg_q:.4f}  "
              f"perplexity: {perplexity:.1f}/{args.n_embed}  usage: {usage:.1%}")

        if args.reset_dead:
            with torch.no_grad():
                z = model.encode(data)  # last batch of the epoch
            n_dead = reset_dead_codes(model, z, counts)
            print(f"  reset {n_dead} dead codes")

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            save_checkpoint(
                model, optimizer, epoch, avg_recon + args.lam_q * avg_q,
                checkpoint_dir / f"vqvae_epoch{epoch}.pt", config,
            )

    plot_curves(recon_losses, q_losses, perplexities, usages, loss_curve_path)