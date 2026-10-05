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
                 n_embed=8192, embed_dim=3, beta=0.25):
        super().__init__()
        self.encoder = Encoder(ch, ch_mult, n_res, n_z=embed_dim)
        self.quantizer = VectorQuantizer(n_embed, embed_dim, beta)
        self.decoder = Decoder(ch, ch_mult, n_res, n_z=embed_dim)

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


# if __name__ == "__main__":
#     torch.manual_seed(0)
#     model = VQModel(ch=64, ch_mult=(1, 2), n_res=1, n_embed=512, embed_dim=3)
#     opt = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.9))
#     x = torch.randn(4, 3, 64, 64).clamp(-1, 1)
#     z = model.encode(x)
#     print("latent shape:", tuple(z.shape), "(expect (4, 3, 16, 16): m=2, f=4)")
#     assert z.shape == (4, 3, 16, 16)
#     assert model(x)[0].shape == x.shape
#     for i in range(30):
#         out = train_step(model, opt, x)
#     use, ppl = codebook_stats(out["idx"], 512)
#     print(f"loss {out['loss']:.4f} rec {out['rec']:.4f} q {out['q']:.4f} "
#           f"usage {use:.2%} perplexity {ppl:.1f}")
#     print("decode(encode(x)) shape:", tuple(model.decode(model.encode(x)).shape))