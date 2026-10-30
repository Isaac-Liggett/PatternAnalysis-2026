import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Utilities 
def save_vq_checkpoint(model, optimizer, epoch, loss, path, config):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config,
    }, path)


def save_ldm_checkpoint(unet, optimizer, epoch, loss, path, unet_config, latent_std, T):
    torch.save({
        "epoch": epoch,
        "model_state_dict": unet.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "unet_config": unet_config,
        "latent_std": latent_std,   # sampling must undo this scaling
        "T": T,
    }, path)


def plot_vq_curves(h, save_path):
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    epochs = range(1, len(h["recon"]) + 1)

    panels = [
        (ax[0, 0], h["recon"], "Reconstruction Loss", "Avg L1 reconstruction loss", None),
        (ax[0, 1], h["q"], "Quantiser Loss", "Avg quantiser loss (codebook + commitment)", "darkorange"),
        (ax[0, 2], h["total"], "Total Loss", "Avg total loss", "crimson"),
        (ax[1, 0], h["sharp"], "Laplacian Sharpness Loss", "Avg L1 of Laplacian variances", "teal"),
        (ax[1, 1], h["perplexity"], "Codebook Perplexity", "Perplexity (epoch)", "green"),
        (ax[1, 2], h["usage"], "Codebook Usage", "Fraction of codes used (epoch)", "purple"),
    ]
    for a, y, title, ylabel, color in panels:
        a.plot(epochs, y, marker="o", color=color)
        a.set_title(title)
        a.set_ylabel(ylabel)
        a.set_xlabel("Epoch")
        a.grid(True, alpha=0.3)
    ax[1, 2].set_ylim(0, 1.05)

    fig.suptitle("Stage 1: VQ-VAE")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_ldm_loss(losses, save_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(losses) + 1), losses, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean MSE (eps prediction)")
    ax.set_title("Stage 2: Denoising Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


class LaplacianVarianceLoss(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels
        kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer("laplacian_kernel", kernel)

    def _laplacian_variance(self, x):
        edges = F.conv2d(x, self.laplacian_kernel, padding=1, groups=self.channels)
        mean_edges = torch.mean(edges, dim=[2, 3], keepdim=True)
        variance = torch.mean((edges - mean_edges) ** 2, dim=[2, 3])
        return torch.mean(variance, dim=1)

    def forward(self, y_pred, y_true):
        return F.l1_loss(self._laplacian_variance(y_pred),
                         self._laplacian_variance(y_true))