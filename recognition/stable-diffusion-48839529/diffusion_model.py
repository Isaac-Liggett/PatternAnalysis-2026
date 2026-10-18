"""Stable Diffusion model combining a VQ-VAE (first-stage autoencoder) and a
denoising UNet (second-stage latent diffusion model).

Provides:
    - ``from_checkpoints``: load both models from saved checkpoints.
    - ``load_vqvae`` / ``load_unet``: load individual sub-models from checkpoints.
    - ``freeze_vqvae`` / ``freeze_unet``: toggle gradient computation per sub-model.
"""

import torch

from vqvae import VQModel
from unet import DenoisingUNet


class StableDiffusion:
    """Combines a VQ-VAE autoencoder with a latent-space denoising UNet.

    Parameters
    ----------
    vqvae : VQModel
        First-stage autoencoder.
    unet : DenoisingUNet
        Second-stage denoising network.
    latent_std : float
        Standard deviation used to normalise latents before diffusion.
    latent_hw : int
        Spatial size of the latent map (height == width).
    latent_c : int
        Number of latent channels.
    device : torch.device
        Target device.
    train_vqvae : bool
        If ``False`` (default), the VQ-VAE is set to eval mode and its
        parameters are frozen.  Set ``True`` to keep it trainable.
    train_unet : bool
        If ``False`` (default), the UNet is set to eval mode and its
        parameters are frozen.  Set ``True`` to keep it trainable.
    """

    def __init__(self, vqvae, unet, latent_std, latent_hw, latent_c, device,
                 train_vqvae=False, train_unet=False):
        self.vqvae = vqvae
        self.unet = unet
        self.latent_std = latent_std
        self.latent_hw = latent_hw
        self.latent_c = latent_c
        self.device = device

        # Apply the requested freeze / unfreeze state.
        self.freeze_vqvae(not train_vqvae)
        self.freeze_unet(not train_unet)

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def freeze_vqvae(self, frozen=True):
        """Freeze or unfreeze the VQ-VAE parameters.

        Parameters
        ----------
        frozen : bool
            If ``True``, sets the VQ-VAE to eval mode and disables gradients.
            If ``False``, sets it to train mode and enables gradients.
        """
        if frozen:
            self.vqvae.eval().requires_grad_(False)
        else:
            self.vqvae.train().requires_grad_(True)

    def freeze_unet(self, frozen=True):
        """Freeze or unfreeze the UNet parameters.

        Parameters
        ----------
        frozen : bool
            If ``True``, sets the UNet to eval mode and disables gradients.
            If ``False``, sets it to train mode and enables gradients.
        """
        if frozen:
            self.unet.eval().requires_grad_(False)
        else:
            self.unet.train().requires_grad_(True)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoints(cls, vqvae_ckpt, unet_ckpt, image_size=256,
                         device=None, train_vqvae=False, train_unet=False):
        """Build a ``StableDiffusion`` from two checkpoint files.

        Parameters
        ----------
        vqvae_ckpt : str or Path
            Path to a VQ-VAE checkpoint (as saved by ``trainvqvae.py``).
        unet_ckpt : str or Path
            Path to a UNet checkpoint (as saved by ``trainunet.py``).
        image_size : int
            Spatial resolution of input images (assumed square).
        device : torch.device, optional
            Target device; defaults to CUDA if available.
        train_vqvae / train_unet : bool
            Passed through to the constructor to control which models are
            frozen vs trainable.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vq = torch.load(vqvae_ckpt, map_location=device)
        vqvae = VQModel(**vq["config"]).to(device)
        vqvae.load_state_dict(vq["model_state_dict"])

        un = torch.load(unet_ckpt, map_location=device)
        unet = DenoisingUNet(**un["unet_config"]).to(device)
        unet.load_state_dict(un["model_state_dict"])

        f = 2 ** len(vq["config"]["ch_mult"])         # downsampling factor
        return cls(vqvae, unet, un["latent_std"],
                   image_size // f, vq["config"]["embed_dim"], device,
                   train_vqvae=train_vqvae, train_unet=train_unet)

    # ------------------------------------------------------------------
    # Individual checkpoint loaders
    # ------------------------------------------------------------------

    def load_vqvae(self, ckpt_path, image_size=256, trainable=False):
        """Load VQ-VAE weights from a checkpoint into the existing model.

        Parameters
        ----------
        ckpt_path : str or Path
            Path to a VQ-VAE checkpoint.
        image_size : int
            Image resolution used to recompute ``latent_hw``.
        trainable : bool
            Whether the loaded model should be trainable (default frozen).
        """
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.vqvae = VQModel(**ckpt["config"]).to(self.device)
        self.vqvae.load_state_dict(ckpt["model_state_dict"])
        f = 2 ** len(ckpt["config"]["ch_mult"])
        self.latent_hw = image_size // f
        self.latent_c = ckpt["config"]["embed_dim"]
        self.freeze_vqvae(not trainable)

    def load_unet(self, ckpt_path, trainable=False):
        """Load UNet weights from a checkpoint into the existing model.

        Parameters
        ----------
        ckpt_path : str or Path
            Path to a UNet checkpoint.
        trainable : bool
            Whether the loaded model should be trainable (default frozen).
        """
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.unet = DenoisingUNet(**ckpt["unet_config"]).to(self.device)
        self.unet.load_state_dict(ckpt["model_state_dict"])
        self.latent_std = ckpt["latent_std"]
        self.freeze_unet(not trainable)

    # ------------------------------------------------------------------
    # Inference utilities
    # ------------------------------------------------------------------

    @staticmethod
    def to_unit(x):
        """[-1, 1] -> [0, 1]."""
        return (x.clamp(-1, 1) + 1) / 2

    @torch.no_grad()
    def encode(self, x):
        """Image -> scaled continuous latent (what the UNet was trained on)."""
        return self.vqvae.encode(x.to(self.device)) / self.latent_std

    @torch.no_grad()
    def decode(self, z):
        """Scaled latent -> image (undoes scaling, quantises, decodes)."""
        return self.vqvae.decode(z * self.latent_std)

    @torch.no_grad()
    def sample(self, n, ddim_steps=50, batch_size=8, seed=None):
        """Unconditional samples from pure noise. Returns (n, C, H, W) on CPU."""
        if seed is not None:
            torch.manual_seed(seed)
        out, remaining = [], n
        while remaining > 0:
            b = min(batch_size, remaining)
            z = self.unet.sample((b, self.latent_c, self.latent_hw, self.latent_hw),
                                 n_steps=ddim_steps)
            out.append(self.decode(z).cpu())
            remaining -= b
        return torch.cat(out)

    # Evaluation

    @torch.no_grad()
    def reconstruct(self, x):
        """decode(encode(x)): the quality ceiling set by the autoencoder."""
        return self.decode(self.encode(x)).cpu()

    @torch.no_grad()
    def denoise_from(self, x, t=500, ddim_steps=25, seed=None):
        """Noise real images to level t, then run the sampler from there.

        Should return something close to x (up to autoencoder error) if the
        UNet, schedule and timestep handling are correct.
        """
        if seed is not None:
            torch.manual_seed(seed)
        z0 = self.encode(x)
        t_batch = torch.full((z0.size(0),), t, device=self.device, dtype=torch.long)
        z_t = self.unet.add_noise(z0, t_batch, torch.randn_like(z0))
        z = self.unet.sample(z0.shape, n_steps=ddim_steps, z=z_t, t_start=t)
        return self.decode(z).cpu()