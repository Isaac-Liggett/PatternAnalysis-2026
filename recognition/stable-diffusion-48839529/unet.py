import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu

def Norm(c):
    return nn.GroupNorm(num_groups=32, num_channels=c, eps=1e-6)

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
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