from diffusion_model import StableDiffusion
import argparse
from pathlib import Path
import json
import math
import time

import torch
from torchvision.utils import make_grid, save_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Latent Diffusion Model (SSIM & Meehan et al. Data-Copying Test).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Checkpoints
    parser.add_argument("--vqvae-ckpt", type=Path,
                        help="Path to trained VQ-VAE checkpoint (.pt)")
    parser.add_argument("--unet-ckpt", type=Path,
                        help="Path to trained UNet diffusion checkpoint (.pt)")
    parser.add_argument("--diffusion-ckpt", type=Path,
                        help="Path to trained Latent Diffusion model checkpoint (.pt)")

    # Generation parameters
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Arbitrary number of synthetic samples to generate")
    parser.add_argument("--ddim-steps", type=int, default=50,
                        help="Number of DDIM steps used for sampling")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for generation reproducibility")

    # Evaluation configuration
    parser.add_argument("--image-size", type=int, default=256,
                        help="Spatial resolution of images")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for dataloaders")
    parser.add_argument("--output-dir", type=Path, default=Path("./eval_diffusion_results"), required=True,
                        help="Directory for saving generated images")

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading StableDiffusion from checkpoints:")
    print(f"  VQ-VAE: {args.vqvae_ckpt}")
    print(f"  UNet:   {args.unet_ckpt}")

    if args.diffusion_ckpt:
        model = StableDiffusion.from_checkpoints(
            vqvae_ckpt=None,
            unet_ckpt=None,
            image_size=args.image_size,
            device=device,
            train_vqvae=False,
            train_unet=False,
        )

        model.load_state_dict(torch.load(args.diffusion_ckpt, weights_only=True))
    elif args.vqvae_ckpt and args.unet_ckpt:
        model = StableDiffusion.from_checkpoints(
            vqvae_ckpt=args.vqvae_ckpt,
            unet_ckpt=args.unet_ckpt,
            image_size=args.image_size,
            device=device,
            train_vqvae=False,
            train_unet=False,
        )
    else:
        raise ValueError("You must supply a model file (.pt)")

    # Generate Samples
    print(f"\nGenerating {args.num_samples} synthetic images (DDIM steps={args.ddim_steps}) …")
    start_gen = time.perf_counter()
    gen_images = model.sample(
        n=args.num_samples,
        ddim_steps=args.ddim_steps,
        batch_size=args.num_samples,
        seed=args.seed,
    )
    gen_elapsed = time.perf_counter() - start_gen
    gen_images = gen_images.clamp(0, 1)
    print(f"Generation completed in {gen_elapsed:.2f}s ({args.num_samples / gen_elapsed:.2f} images/s)")

    # Save individual images
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    for idx in range(gen_images.size(0)):
        save_image(gen_images[idx], samples_dir / f"sample_{idx:05d}.png")
    print(f"Saved {gen_images.size(0)} individual images to {samples_dir}")

if __name__ == "__main__":
    main()
