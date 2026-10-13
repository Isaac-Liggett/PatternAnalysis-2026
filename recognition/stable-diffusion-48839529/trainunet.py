from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from vqvae import VQModel
from unet import DenoisingUNet
from dataset import get_dataloader

BATCH_SIZE = 8
EPOCHS = 100
LR = 2e-4
IMAGE_SIZE = 256
T = 1000   # training timesteps (the DDIM step count is a sampling-time choice)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae-ckpt", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--data-dir", type=Path, default=Path("./keras_png_slices_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./data"))
    parser.add_argument("--checkpoint-every", type=int, default=5)
    return parser.parse_args()


def save_checkpoint(unet, optimizer, epoch, loss, path, unet_config, latent_std):
    torch.save({
        "epoch": epoch,
        "model_state_dict": unet.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "unet_config": unet_config,   # to rebuild the UNet when sampling
        "latent_std": latent_std,     # sampling must undo this scaling
        "T": T,
    }, path)


def plot_loss(losses, save_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(losses) + 1), losses, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean MSE (eps prediction)")
    ax.set_title("Denoising Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    print(f"Pytorch running on {device}")
    args = parse_args()

    # initial checks
    unet = DenoisingUNet(channels=3)
    x = torch.randn(2, 3, 64, 64)
    t = torch.tensor([0, 999])
    out = unet(x, t)
    assert out.shape == x.shape
    print(unet.abar[0].item(), unet.abar[-1].item())     # ~1.0 and ~1e-4
    print(unet.sample((2, 3, 64, 64), n_steps=5).shape)   # sampler runs end to end

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataloader = get_dataloader(
        str(args.data_dir), image_size=args.image_size,
        batch_size=args.batch_size, num_workers=0,
    )

    vq_ckpt = torch.load(args.vqvae_ckpt, map_location=device)
    vq_config = vq_ckpt["config"]
    vqvae = VQModel(**vq_config).to(device)
    vqvae.load_state_dict(vq_ckpt["model_state_dict"])
    vqvae.eval().requires_grad_(False)
    embed_dim = vq_config["embed_dim"]

    # estimating latent std
    with torch.no_grad():
        zs = torch.cat([vqvae.encode(b.to(device)) for _, b in zip(range(50), dataloader)])
    latent_std = zs.std().item()
    print(f"Latent shape {tuple(zs.shape[1:])}, mean {zs.mean():.3f}, std {latent_std:.3f}")
    del zs

    # Denoising UNET
    unet_config = dict(channels=embed_dim)
    unet = DenoisingUNet(**unet_config).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)
    print(f"UNet parameters: {sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M")

    # Noise schedule (DDPM, as used by the LDM authors)
    betas = torch.linspace(1.5e-3 ** 0.5, 1.95e-2 ** 0.5, T) ** 2
    abar = torch.cumprod(1.0 - betas, dim=0).to(device)

    losses = []
    for epoch in tqdm(range(1, args.epochs + 1), desc="EPOCH"):
        unet.train()
        running = 0.0

        for data in tqdm(dataloader, desc="BATCH", leave=False):
            data = data.to(device)
            with torch.no_grad():
                z = vqvae.encode(data) / latent_std

            eps = torch.randn_like(z)
            t = torch.randint(0, T, (z.size(0),), device=device)
            a = abar[t].view(-1, 1, 1, 1)
            z_t = a.sqrt() * z + (1 - a).sqrt() * eps

            loss = F.mse_loss(unet(z_t, t), eps)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()

        losses.append(running / len(dataloader))
        print(f"Epoch {epoch}/{args.epochs} | loss: {losses[-1]:.4f}")

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            save_checkpoint(
                unet, optimizer, epoch, losses[-1],
                checkpoint_dir / f"denoising_unet_epoch{epoch}.pt",
                unet_config, latent_std,
            )
            plot_loss(losses, args.output_dir / "loss_curve.png")