"""
Two-stage latent diffusion training.

Stage 1: train the VQ-VAE autoencoder (L1 recon + quantiser loss + Laplacian sharpness).
Stage 2: freeze it, encode images to latents, train a DDPM-style eps-prediction UNet on them.

Requires vqvae.py, unet.py, dataset.py in the same directory.

Outputs (under --output-dir):
    checkpoints/vqvae_epoch{N}.pt
    checkpoints/denoising_unet_epoch{N}.pt
    vqvae_curves.png      stage 1 curves (recon, quantiser, total, sharpness, perplexity, usage)
    denoising_loss.png    stage 2 curve

Usage:
    python train_ldm.py
    python train_ldm.py --vq-epochs 50 --ldm-epochs 100 --reset-dead
    python train_ldm.py --vqvae-ckpt data/checkpoints/vqvae_epoch50.pt   # skip stage 1
"""
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from modules import VQModel, reset_dead_codes, DenoisingUNet
from dataset import get_dataloader
from utils import save_vq_checkpoint, save_ldm_checkpoint, plot_vq_curves, plot_ldm_loss, LaplacianVarianceLoss

# Hyperparameters

# Shared
IMAGE_SIZE = 256

# VQ-VAE
VQ_BATCH_SIZE = 3
VQ_EPOCHS = 1
VQ_LR = 2e-4
IN_CHANNELS = 1
CH = 128
CH_MULT = (1, 2)   # 256x256 input -> 64x64 latent (f = 4)
N_RES = 1
N_EMBED = 1024
EMBED_DIM = 3
BETA = 0.25        # commitment weight, NOT a KL weight
LAM_Q = 1.0
LAM_SHARP = 0.01
LATENT_DOWNSAMPLE = 4   # keep consistent with CH_MULT above

# unet/diffusion
LDM_BATCH_SIZE = 8
LDM_EPOCHS = 1
LDM_LR = 2e-4
T = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    p = argparse.ArgumentParser()
    # shared
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--data-dir", type=Path, default=Path("./keras_png_slices_data"))
    p.add_argument("--output-dir", type=Path, default=Path("./data"))
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--vqvae-ckpt", type=str, default=None,
                   help="If given, skip stage 1 and load this VQ-VAE checkpoint")
    # VQ-VAE
    p.add_argument("--vq-batch-size", type=int, default=VQ_BATCH_SIZE)
    p.add_argument("--vq-epochs", type=int, default=VQ_EPOCHS)
    p.add_argument("--vq-lr", type=float, default=VQ_LR)
    p.add_argument("--ch", type=int, default=CH)
    p.add_argument("--in-ch", type=int, default=IN_CHANNELS)
    p.add_argument("--n-embed", type=int, default=N_EMBED)
    p.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    p.add_argument("--beta", type=float, default=BETA, help="Commitment loss weight")
    p.add_argument("--lam-q", type=float, default=LAM_Q)
    p.add_argument("--lam-sharp", type=float, default=LAM_SHARP)
    p.add_argument("--reset-dead", action="store_true",
                   help="At each epoch end, re-initialise codes never used that epoch")
    # UNET
    p.add_argument("--ldm-batch-size", type=int, default=LDM_BATCH_SIZE)
    p.add_argument("--ldm-epochs", type=int, default=LDM_EPOCHS)
    p.add_argument("--ldm-lr", type=float, default=LDM_LR)
    return p.parse_args()


def vq_loss_function(x_rec, x, qloss, sharpness_loss, lam_q=1.0, lam_sharp=0.01):
    """Returns (total, recon, sharp). All are batch means."""
    rec = F.l1_loss(x_rec, x)
    sharp = sharpness_loss(x_rec, x)
    total = rec + lam_q * qloss + lam_sharp * sharp
    return total, rec, sharp


def train_vqvae(args, config, checkpoint_dir):
    """
    VQ-VAE Training script. Loads the dataset according to the args

    params:
        args: arguments to be passed
        config: configuration to load into VQ-VAE if pretrained
        checkpoint_dir: where to save the checkpoint files per epochs

    returns the trained model to be used in the next stage 
    """

    dataloader = get_dataloader(
        str(args.data_dir), image_size=args.image_size,
        batch_size=args.vq_batch_size, num_workers=0,
    )

    sample = next(iter(dataloader))
    assert sample.shape[1] == args.in_ch, \
        f"expected {args.in_ch} channels, got {sample.shape[1]}"
    print(f"Batch shape {tuple(sample.shape)}, value range "
          f"[{sample.min():.3f}, {sample.max():.3f}]")

    model = VQModel(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.vq_lr, betas=(0.5, 0.9))
    sharp_loss = LaplacianVarianceLoss(args.in_ch).to(device)

    n_samples = len(dataloader.dataset)
    hist = {k: [] for k in ("recon", "q", "total", "sharp", "perplexity", "usage")}
    curve_path = args.output_dir / "vqvae_curves.png"

    # Main VQ-VAE training loop
    for epoch in tqdm(range(1, args.vq_epochs + 1), desc="VQ EPOCH"):
        model.train()
        e_rec = e_q = e_total = e_sharp = 0.0
        counts = torch.zeros(args.n_embed, device=device)

        for data in tqdm(dataloader, desc="BATCH", leave=False):
            data = data.to(device)

            x_rec, qloss, idx = model(data)
            loss, rec, sharp = vq_loss_function(
                x_rec, data, qloss, sharp_loss,
                lam_q=args.lam_q, lam_sharp=args.lam_sharp)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = data.size(0)
            e_rec += rec.item() * bs
            e_q += qloss.item() * bs
            e_total += loss.item() * bs
            e_sharp += sharp.item() * bs
            counts += torch.bincount(idx.flatten(), minlength=args.n_embed)

        p = counts / counts.sum()
        perplexity = torch.exp(-(p * torch.log(p + 1e-10)).sum()).item()
        usage = (counts > 0).float().mean().item()

        hist["recon"].append(e_rec / n_samples)
        hist["q"].append(e_q / n_samples)
        hist["total"].append(e_total / n_samples)
        hist["sharp"].append(e_sharp / n_samples)
        hist["perplexity"].append(perplexity)
        hist["usage"].append(usage)

        print(f"[VQ] Epoch {epoch}/{args.vq_epochs} | recon: {hist['recon'][-1]:.4f}  "
              f"q: {hist['q'][-1]:.4f}  sharp: {hist['sharp'][-1]:.4f}  "
              f"total: {hist['total'][-1]:.4f}  "
              f"perplexity: {perplexity:.1f}/{args.n_embed}  usage: {usage:.1%}")

        if args.reset_dead:
            with torch.no_grad():
                z = model.encode(data)  # last batch of the epoch only
            n_dead = reset_dead_codes(model, z, counts)
            print(f"  reset {n_dead} dead codes")

        if epoch % args.checkpoint_every == 0 or epoch == args.vq_epochs:
            save_vq_checkpoint(
                model, optimizer, epoch,
                hist["recon"][-1] + args.lam_q * hist["q"][-1],
                checkpoint_dir / f"vqvae_epoch{epoch}.pt", config)
            plot_vq_curves(hist, curve_path)

    return model


def train_diffusion(args, vqvae, embed_dim, checkpoint_dir):
    """
    Trains the UNET portion of the latent diffusion model.

    params
        args: arguments to be used for the training process
        vqvae: trained VQ-VAE model
        embed_dim: embedding dimension of the VQ-VAE model (latent dimension)
        checkpoint_dir: checkpoint directory for trained models
    
    returns the trained unet model to be used for inference.
    """
    vqvae.eval().requires_grad_(False)

    dataloader = get_dataloader(
        str(args.data_dir), image_size=args.image_size,
        batch_size=args.ldm_batch_size, num_workers=0,
    )

    # Estimate latent std (scalar) so diffusion sees roughly unit-variance inputs.
    with torch.no_grad():
        zs = torch.cat([vqvae.encode(b.to(device)) for _, b in zip(range(50), dataloader)])
    latent_std = zs.std().item()
    print(f"Latent shape {tuple(zs.shape[1:])}, mean {zs.mean():.3f}, std {latent_std:.3f}")
    del zs

    unet_config = dict(channels=embed_dim)
    unet = DenoisingUNet(**unet_config).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.ldm_lr)
    print(f"UNet parameters: {sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M")

    # Noise schedule (DDPM / LDM "scaled linear")
    betas = torch.linspace(1.5e-3 ** 0.5, 1.95e-2 ** 0.5, T) ** 2
    abar = torch.cumprod(1.0 - betas, dim=0).to(device)

    # The sampler inside the UNet keeps its own schedule; a mismatch with the
    # one used here would silently ruin sampling, so check it.
    if not torch.allclose(abar.cpu(), unet.abar.detach().cpu().float(), atol=1e-6):
        print("WARNING: training abar differs from unet.abar -- sampling will be inconsistent")

    losses = []
    loss_path = args.output_dir / "denoising_loss.png"

    for epoch in tqdm(range(1, args.ldm_epochs + 1), desc="LDM EPOCH"):
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
        print(f"[LDM] Epoch {epoch}/{args.ldm_epochs} | loss: {losses[-1]:.4f}")

        if epoch % args.checkpoint_every == 0 or epoch == args.ldm_epochs:
            save_ldm_checkpoint(
                unet, optimizer, epoch, losses[-1],
                checkpoint_dir / f"denoising_unet_epoch{epoch}.pt",
                unet_config, latent_std, T)
            plot_ldm_loss(losses, loss_path)

    return unet


def preflight_unet_check(embed_dim, image_size):
    """Fail fast: run the UNet/sampler sanity check first"""
    lat = image_size // LATENT_DOWNSAMPLE
    unet = DenoisingUNet(channels=embed_dim)
    x = torch.randn(2, embed_dim, lat, lat)
    t = torch.tensor([0, T - 1])
    assert unet(x, t).shape == x.shape
    print(f"UNet abar[0]={unet.abar[0].item():.4f}, abar[-1]={unet.abar[-1].item():.2e}  "
          f"(expect ~1.0 and ~1e-4)")
    assert unet.sample((2, embed_dim, lat, lat), n_steps=5).shape == x.shape
    del unet


if __name__ == "__main__":
    print(f"Pytorch running on {device}")
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 config (or the one stored in a supplied checkpoint)
    if args.vqvae_ckpt:
        ckpt = torch.load(args.vqvae_ckpt, map_location=device)
        config = ckpt["config"]
    else:
        config = dict(ch=args.ch, ch_mult=CH_MULT, n_res=N_RES, n_embed=args.n_embed,
                      embed_dim=args.embed_dim, beta=args.beta, in_channel=args.in_ch)
    embed_dim = config["embed_dim"]

    preflight_unet_check(embed_dim, args.image_size)

    if args.vqvae_ckpt:
        print(f"Skipping stage 1, loading {args.vqvae_ckpt}")
        vqvae = VQModel(**config).to(device)
        vqvae.load_state_dict(ckpt["model_state_dict"])
    else:
        print("=== Stage 1: VQ-VAE ===")
        vqvae = train_vqvae(args, config, checkpoint_dir)

    print("=== Stage 2: latent diffusion ===")
    train_diffusion(args, vqvae, embed_dim, checkpoint_dir)

    print(f"Done. Curves: {args.output_dir / 'vqvae_curves.png'}, "
          f"{args.output_dir / 'denoising_loss.png'}")