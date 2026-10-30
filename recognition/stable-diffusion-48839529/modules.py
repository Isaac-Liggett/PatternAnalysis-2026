"""VQ autoencoder following Table 7 of VQGAN (Esser et al., CVPR 2021).

Encoder: Conv -> m x {ResBlock, Downsample} -> Res, NonLocal, Res -> GN, Swish, Conv -> n_z
Decoder: Conv -> Res, NonLocal, Res -> m x {ResBlock, Upsample} -> GN, Swish, Conv -> C
No skip connections between encoder and decoder. f = 2^m, m = len(ch_mult).

The quantiser operates directly on the n_z-channel encoder output (n_z = embed_dim);
there are no extra 1x1 projection convs. Trained here with L1 + codebook + commitment
(plain VQ-VAE). Add LPIPS + PatchGAN for the full VQGAN-style objective.

Channel widths: widths = [ch, ch*ch_mult[0], ..., ch*ch_mult[-1]]. Encoder stage i maps
widths[i] -> widths[i+1]; decoder stage j maps widths[m-j] -> widths[m-j-1].
Table 7 does not specify widths, n_res, K or GroupNorm groups; these are chosen
hyperparameters, not sourced values.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu


def Norm(c):
    return nn.GroupNorm(num_groups=32, num_channels=c, eps=1e-6, affine=True)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out=None):
        super().__init__()
        c_out = c_out or c_in
        self.norm1 = Norm(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = Norm(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class NonLocalBlock(nn.Module):
    """Single-head spatial self-attention over the H*W positions."""
    def __init__(self, c):
        super().__init__()
        self.norm = Norm(c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).unbind(1)
        q, k, v = (t.transpose(1, 2).unsqueeze(1) for t in (q, k, v))  # B,1,HW,C
        h = F.scaled_dot_product_attention(q, k, v)                    # B,1,HW,C
        h = h.squeeze(1).transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(h)


class Downsample(nn.Module):
    """Stride-2 conv with asymmetric (right/bottom) padding. H, W must be even."""
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class Upsample(nn.Module):
    """Nearest-neighbour x2 followed by a 3x3 conv."""
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


def _stage(c_in, c_out, n_res):
    blocks = [ResBlock(c_in, c_out)] + [ResBlock(c_out) for _ in range(n_res - 1)]
    return nn.Sequential(*blocks)


class Encoder(nn.Module):
    def __init__(self, ch=128, ch_mult=(1, 2), n_res=1, n_z=3, in_ch=3):
        super().__init__()
        widths = [ch] + [ch * m for m in ch_mult]
        self.conv_in = nn.Conv2d(in_ch, widths[0], 3, padding=1)
        layers = []
        for i in range(len(ch_mult)):
            layers += [_stage(widths[i], widths[i + 1], n_res), Downsample(widths[i + 1])]
        self.down = nn.Sequential(*layers)
        c = widths[-1]
        self.mid = nn.Sequential(ResBlock(c), NonLocalBlock(c), ResBlock(c))
        self.norm_out = Norm(c)
        self.conv_out = nn.Conv2d(c, n_z, 3, padding=1)

    def forward(self, x):
        h = self.mid(self.down(self.conv_in(x)))
        return self.conv_out(F.silu(self.norm_out(h)))


class Decoder(nn.Module):
    def __init__(self, ch=128, ch_mult=(1, 2), n_res=1, n_z=3, out_ch=3):
        super().__init__()
        widths = [ch] + [ch * m for m in ch_mult]
        m_levels = len(ch_mult)
        c = widths[-1]
        self.conv_in = nn.Conv2d(n_z, c, 3, padding=1)
        self.mid = nn.Sequential(ResBlock(c), NonLocalBlock(c), ResBlock(c))
        layers = []
        for j in range(m_levels):
            c_in, c_out = widths[m_levels - j], widths[m_levels - j - 1]
            layers += [_stage(c_in, c_out, n_res), Upsample(c_out)]
        self.up = nn.Sequential(*layers)
        self.norm_out = Norm(widths[0])
        self.conv_out = nn.Conv2d(widths[0], out_ch, 3, padding=1)

    def forward(self, z):
        h = self.up(self.mid(self.conv_in(z)))
        return self.conv_out(F.silu(self.norm_out(h)))


class VectorQuantizer(nn.Module):
    """Nearest-neighbour codebook lookup with straight-through gradients.

    loss = ||sg[z_e] - e||^2  (moves codebook toward encoder outputs)
         + beta * ||z_e - sg[e]||^2  (commitment: keeps encoder near codebook)
    """
    def __init__(self, n_embed=8192, embed_dim=3, beta=0.25):
        super().__init__()
        self.n_embed, self.embed_dim, self.beta = n_embed, embed_dim, beta
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)

    def forward(self, z):
        B, C, H, W = z.shape
        z = z.float()  # quantisation in fp32 even under autocast
        zf = z.permute(0, 2, 3, 1).reshape(-1, C)              # (BHW, C)
        E = self.embedding.weight                                # (K, C)
        # ||z - e||^2 = ||z||^2 - 2 z.e + ||e||^2
        d = zf.pow(2).sum(1, keepdim=True) - 2 * zf @ E.t() + E.pow(2).sum(1)[None]
        idx = d.argmin(dim=1)                                    # (BHW,)
        zq = self.embedding(idx).view(B, H, W, C).permute(0, 3, 1, 2)

        loss = F.mse_loss(zq, z.detach()) + self.beta * F.mse_loss(z, zq.detach())
        zq = z + (zq - z).detach()                               # straight-through
        return zq, loss, idx.view(B, H, W)

    @torch.no_grad()
    def lookup(self, idx):
        return self.embedding(idx).permute(0, 3, 1, 2)


class VQModel(nn.Module):
    def __init__(self, ch=128, ch_mult=(1, 2), n_res=1,
                 n_embed=8192, embed_dim=3, beta=0.25, in_channel=3):
        super().__init__()
        self.encoder = Encoder(ch, ch_mult, n_res, n_z=embed_dim, in_ch=in_channel)
        self.quantizer = VectorQuantizer(n_embed, embed_dim, beta)
        self.decoder = Decoder(ch, ch_mult, n_res, n_z=embed_dim, out_ch=in_channel)

    def encode(self, x):
        """Continuous pre-quantisation latent. This is what the LDM diffuses over."""
        return self.encoder(x)

    def decode(self, z):
        """Quantise inside the decoder (as in LDM-VQ), then decode."""
        zq, _, _ = self.quantizer(z)
        return self.decoder(zq)

    def forward(self, x):
        z = self.encode(x)
        zq, qloss, idx = self.quantizer(z)
        return self.decoder(zq), qloss, idx


def codebook_stats(idx, n_embed):
    """Usage fraction and perplexity of code assignments in a batch."""
    p = torch.bincount(idx.flatten(), minlength=n_embed).float()
    p = p / p.sum()
    perplexity = torch.exp(-(p * torch.log(p + 1e-10)).sum())
    return (p > 0).float().mean().item(), perplexity.item()


@torch.no_grad()
def reset_dead_codes(model, z_batch, usage_counts, threshold=1):
    """Re-initialise codes unused over a window to random encoder outputs."""
    dead = usage_counts < threshold
    n_dead = int(dead.sum())
    if n_dead == 0:
        return 0
    zf = z_batch.permute(0, 2, 3, 1).reshape(-1, z_batch.shape[1]).float()
    pick = zf[torch.randint(0, zf.shape[0], (n_dead,), device=zf.device)]
    model.quantizer.embedding.weight.data[dead] = pick
    return n_dead


def train_step(model, opt, x, lam_q=1.0):
    x_rec, qloss, idx = model(x)
    rec = F.l1_loss(x_rec, x)
    loss = rec + lam_q * qloss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return dict(loss=loss.item(), rec=rec.item(), q=qloss.item(), idx=idx)

"""
Denoising UNet implementation adapted from Demo 2 implementation.

This implementation was based on the paper (Ronneberger et al, 2015)
"""

def Norm(c):
    return nn.GroupNorm(num_groups=32, num_channels=c, eps=1e-6)

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = max_period ** (-torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)   # (B, dim)

def make_schedule(T):
    betas = torch.linspace(1.5e-3 ** 0.5, 1.95e-2 ** 0.5, T) ** 2 
    return torch.cumprod(1.0 - betas, dim=0)

class Attn(nn.Module):
    """Single-head spatial self-attention."""
    def __init__(self, c):
        super().__init__()
        self.norm = Norm(c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).unbind(1)
        q, k, v = (t.transpose(1, 2).unsqueeze(1) for t in (q, k, v))
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.squeeze(1).transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(h)

class DenoisingUNet(nn.Module):
    def __init__(self, channels: int = 3, T: int = 1000):
        super().__init__()

        self.T = T
        betas = torch.linspace(1.5e-3 ** 0.5, 1.95e-2 ** 0.5, T) ** 2
        self.register_buffer("abar", torch.cumprod(1.0 - betas, dim=0), persistent=False)

        # --- Timestep embedding ---
        self.t_dim = 64
        temb_dim = 256
        self.time_mlp = nn.Sequential(
            nn.Linear(self.t_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim))

        # --- Encoder ---
        self.e11 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
        self.bn11 = Norm(64)
        self.t11 = nn.Linear(temb_dim, 64)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = Norm(64)
        self.pool1 = nn.MaxPool2d(2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = Norm(128)
        self.t21 = nn.Linear(temb_dim, 128)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = Norm(128)
        self.pool2 = nn.MaxPool2d(2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = Norm(256)
        self.t31 = nn.Linear(temb_dim, 256)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = Norm(256)
        self.pool3 = nn.MaxPool2d(2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = Norm(512)
        self.t41 = nn.Linear(temb_dim, 512)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = Norm(512)
        self.pool4 = nn.MaxPool2d(2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn51 = Norm(1024)
        self.t51 = nn.Linear(temb_dim, 1024)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn52 = Norm(1024)
        self.attn5 = Attn(1024)          # bottleneck (4x4 for a 64x64 latent)

        # --- Decoder ---
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d12 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bnd12 = Norm(512)
        self.td12 = nn.Linear(temb_dim, 512)
        self.d13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bnd13 = Norm(512)
        self.attn1 = Attn(512)           # 8x8

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d22 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bnd22 = Norm(256)
        self.td22 = nn.Linear(temb_dim, 256)
        self.d23 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bnd23 = Norm(256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d32 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bnd32 = Norm(128)
        self.td32 = nn.Linear(temb_dim, 128)
        self.d33 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bnd33 = Norm(128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d42 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bnd42 = Norm(64)
        self.td42 = nn.Linear(temb_dim, 64)
        self.d43 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bnd43 = Norm(64)

        # Output layer: predicts the noise eps, same shape as the input latent.
        self.outconv = nn.Conv2d(64, channels, kernel_size=1)

    def forward(self, x, t):
        temb = F.silu(self.time_mlp(timestep_embedding(t, self.t_dim)))   # (B, 256)

        # Encoder
        xe11 = relu(self.bn11(self.e11(x) + self.t11(temb)[:, :, None, None]))
        xe12 = relu(self.bn12(self.e12(xe11)))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.bn21(self.e21(xp1) + self.t21(temb)[:, :, None, None]))
        xe22 = relu(self.bn22(self.e22(xe21)))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.bn31(self.e31(xp2) + self.t31(temb)[:, :, None, None]))
        xe32 = relu(self.bn32(self.e32(xe31)))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.bn41(self.e41(xp3) + self.t41(temb)[:, :, None, None]))
        xe42 = relu(self.bn42(self.e42(xe41)))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.bn51(self.e51(xp4) + self.t51(temb)[:, :, None, None]))
        xe52 = relu(self.bn52(self.e52(xe51)))
        xe52 = self.attn5(xe52)

        # Decoder
        xd11 = self.up1(xe52)
        xu11 = torch.cat([xd11, xe42], dim=1)
        xd12 = relu(self.bnd12(self.d12(xu11) + self.td12(temb)[:, :, None, None]))
        xd13 = relu(self.bnd13(self.d13(xd12)))
        xd13 = self.attn1(xd13)

        xd21 = self.up2(xd13)
        xu22 = torch.cat([xd21, xe32], dim=1)
        xd22 = relu(self.bnd22(self.d22(xu22) + self.td22(temb)[:, :, None, None]))
        xd23 = relu(self.bnd23(self.d23(xd22)))

        xd31 = self.up3(xd23)
        xu32 = torch.cat([xd31, xe22], dim=1)
        xd32 = relu(self.bnd32(self.d32(xu32) + self.td32(temb)[:, :, None, None]))
        xd33 = relu(self.bnd33(self.d33(xd32)))

        xd41 = self.up4(xd33)
        xu42 = torch.cat([xd41, xe12], dim=1)
        xd42 = relu(self.bnd42(self.d42(xu42) + self.td42(temb)[:, :, None, None]))
        xd43 = relu(self.bnd43(self.d43(xd42)))

        return self.outconv(xd43)
    
    def add_noise(self, z, t, eps):
        a = self.abar[t].view(-1, 1, 1, 1)
        return a.sqrt() * z + (1 - a).sqrt() * eps
    
    @torch.no_grad()
    def sample(self, shape, n_steps=50):
        """Deterministic DDIM (eta = 0). Returns a latent in the scaled space;
        the caller multiplies by latent_std and decodes."""
        was_training = self.training
        self.eval()
        dev = self.abar.device

        ts = torch.linspace(self.T - 1, 0, n_steps).round().long().tolist()
        z = torch.randn(shape, device=dev)
        for i, t in enumerate(ts):
            a_t = self.abar[t]
            a_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.ones_like(a_t)

            eps = self(z, torch.full((shape[0],), t, device=dev, dtype=torch.long))
            z0_hat = (z - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            z = a_prev.sqrt() * z0_hat + (1 - a_prev).sqrt() * eps

        self.train(was_training)
        return z

"""Stable Diffusion model combining a VQ-VAE (first-stage autoencoder) and a
denoising UNet (second-stage latent diffusion model).

Provides:
    - ``from_checkpoints``: load both models from saved checkpoints.
    - ``load_vqvae`` / ``load_unet``: load individual sub-models from checkpoints.
    - ``freeze_vqvae`` / ``freeze_unet``: toggle gradient computation per sub-model.
"""

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

    def freeze_vqvae(self, frozen=True):
        """Freeze or unfreeze the VQ-VAE parameters.

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

        frozen : bool
            If ``True``, sets the UNet to eval mode and disables gradients.
            If ``False``, sets it to train mode and enables gradients.
        """
        if frozen:
            self.unet.eval().requires_grad_(False)
        else:
            self.unet.train().requires_grad_(True)

    @classmethod
    def from_checkpoints(cls, vqvae_ckpt, unet_ckpt, image_size=256,
                         device=None, train_vqvae=False, train_unet=False):
        """Build a ``StableDiffusion`` from two checkpoint files.

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

    def load_vqvae(self, ckpt_path, image_size=256, trainable=False):
        """Load VQ-VAE weights from a checkpoint into the existing model.

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