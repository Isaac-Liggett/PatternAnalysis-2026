from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from vqvae import VQModel, reset_dead_codes
from unet import DenoisingUNet
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
N_STEPS = 50       # number of denoising steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae-ckpt", type=str)
    parser.add_argument("--n_steps", type=int, default=N_STEPS)
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


def plot_curves(losses, q_losses, perplexities, usages, save_path):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    epochs = range(1, len(losses) + 1)

    ax[0, 0].plot(epochs, losses, marker="o")
    ax[0, 0].set_ylabel("Avg L2 loss (per denoising step)")
    ax[0, 0].set_title("Denoising Loss")

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

def noise(z, t, noise):
    return t * noise + (1-t) * z

def loss_function(pred, true):
    return torch.norm(pred-true, 2)**2.0

if __name__ == "__main__":
    print(f"Pytorch running on {device}")

    args = parse_args()

    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = OUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    loss_curve_path = args.output_dir / "loss_curve.png"

    dataloader = get_dataloader(
        str(args.data_dir),
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
    vqvae = VQModel(**config).to(device)
    unet = DenoisingUNet(n_steps=args.n_steps).to(device)

    checkpoint = torch.load(args.vqvae_ckpt, map_location=device)
    vqvae.load_state_dict(checkpoint["model_state_dict"])
    vqvae.eval()

    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr, betas=(0.5, 0.9))

    n_samples = len(dataloader.dataset)

    losses = []

    for epoch in tqdm(range(1, args.epochs + 1), desc="EPOCH"):
        unet.train()
        epoch_loss = 0.0
        counts = torch.zeros(args.n_embed, device=device)

        for data in tqdm(dataloader, desc="BATCH", leave=False):
            data = data.to(device)

            with torch.no_grad():
                zs = vqvae.encode(data)
            
            z_noise = torch.randn_like(zs)
            
            noised = noise(zs, 1, z_noise)
            for step in range(args.n_steps-1, -1, -1):
                res = unet(noised)
                noised = noise(zs, step/args.n_steps, z_noise)

                loss = loss_function(res, noised)
                epoch_loss += loss.item()
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            
        losses.append(epoch_loss)

        print(f"Epoch {epoch}/{args.epochs} | loss: {losses[-1]:2f}")

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            save_checkpoint(
                unet, optimizer, epoch,
                checkpoint_dir / f"denoising_unet_epoch{epoch}.pt", config,
            )

    plot_curves(losses)