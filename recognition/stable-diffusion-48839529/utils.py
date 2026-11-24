import time

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


class GPUTracker:
    """Lightweight per-epoch GPU resource tracker using torch.cuda internals.

    Records VRAM allocation (current and peak), caching-allocator reservation,
    GPU compute utilisation, and wall-clock epoch time.  Gracefully no-ops when
    running on CPU.
    """

    def __init__(self, device):
        self.enabled = device.type == "cuda"
        self.history = {
            "epoch": [],
            "allocated_mb": [],
            "peak_allocated_mb": [],
            "reserved_mb": [],
            "peak_reserved_mb": [],
            "utilization_pct": [],
            "epoch_time_s": [],
        }
        self._epoch_start = None

    def epoch_start(self):
        """Call at the very beginning of each epoch."""
        if self.enabled:
            torch.cuda.reset_peak_memory_stats()
        self._epoch_start = time.perf_counter()

    def epoch_end(self, epoch):
        """Call at the end of each epoch to record a snapshot."""
        epoch_time = time.perf_counter() - self._epoch_start
        self.history["epoch"].append(epoch)
        self.history["epoch_time_s"].append(epoch_time)

        if not self.enabled:
            for key in ("allocated_mb", "peak_allocated_mb",
                        "reserved_mb", "peak_reserved_mb",
                        "utilization_pct"):
                self.history[key].append(None)
            return

        self.history["allocated_mb"].append(
            torch.cuda.memory_allocated() / 1024 ** 2)
        self.history["peak_allocated_mb"].append(
            torch.cuda.max_memory_allocated() / 1024 ** 2)
        self.history["reserved_mb"].append(
            torch.cuda.memory_reserved() / 1024 ** 2)
        self.history["peak_reserved_mb"].append(
            torch.cuda.max_memory_reserved() / 1024 ** 2)
        try:
            self.history["utilization_pct"].append(torch.cuda.utilization())
        except AttributeError:
            # torch.cuda.utilization() requires PyTorch >= 2.1
            self.history["utilization_pct"].append(None)

    def summary(self):
        """Print a one-line summary of peak VRAM usage across all epochs."""
        if not self.enabled or not self.history["epoch"]:
            return
        peaks = [v for v in self.history["peak_allocated_mb"] if v is not None]
        if not peaks:
            return
        peak = max(peaks)
        total = torch.cuda.get_device_properties(0).total_mem / 1024 ** 2
        print(f"  GPU peak VRAM: {peak:.0f} MB / {total:.0f} MB "
              f"({peak / total:.1%})")


def plot_gpu_usage(history, save_path, stage_name=""):
    """Plot GPU VRAM, utilisation and epoch duration over training."""
    epochs = history["epoch"]
    if not epochs:
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # ── VRAM ──
    peak_alloc = history["peak_allocated_mb"]
    cur_alloc = history["allocated_mb"]
    if any(v is not None for v in peak_alloc):
        valid_e = [e for e, v in zip(epochs, peak_alloc) if v is not None]
        valid_peak = [v for v in peak_alloc if v is not None]
        valid_cur = [v for e, v in zip(epochs, cur_alloc) if v is not None]
        ax1.plot(valid_e, valid_peak, "o-", color="crimson",
                 label="Peak Allocated")
        ax1.plot(valid_e, valid_cur, "o-", color="royalblue",
                 label="Current Allocated")
        ax1.fill_between(valid_e, valid_cur, valid_peak,
                         alpha=0.15, color="crimson")
    ax1.set_ylabel("VRAM (MB)")
    ax1.set_title(f"{stage_name} GPU Memory Usage")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── GPU utilisation ──
    utils = history["utilization_pct"]
    if any(u is not None for u in utils):
        valid = [(e, u) for e, u in zip(epochs, utils) if u is not None]
        ax2.plot(*zip(*valid), "o-", color="forestgreen")
    ax2.set_ylabel("GPU Utilization (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title(f"{stage_name} GPU Compute Utilization")
    ax2.grid(True, alpha=0.3)

    # ── Epoch wall time ──
    ax3.bar(epochs, history["epoch_time_s"], color="slategray", alpha=0.7)
    ax3.set_ylabel("Time (s)")
    ax3.set_xlabel("Epoch")
    ax3.set_title(f"{stage_name} Epoch Duration")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)