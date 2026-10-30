from typing import List

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class VAE(nn.Module):
    """
    VAE adapted from https://github.com/AntixK/PyTorch-VAE
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: List[int] = None,
        image_size: int = 64,
        output_activation: str = "sigmoid",
        logvar_clamp=(-10.0, 10.0),
    ) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.image_size = image_size
        self.logvar_clamp = logvar_clamp

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = list(hidden_dims)  # keep an unreversed copy

        # ---- Build Encoder ----
        modules = []
        c_in = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(c_in, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            c_in = h_dim
        self.encoder = nn.Sequential(*modules)

        # Determine actual bottleneck shape by running a dummy input through
        # the encoder, rather than assuming a fixed 2x2 spatial size.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            enc_out = self.encoder(dummy)
        _, self._enc_channels, self._enc_h, self._enc_w = enc_out.shape
        flat_dim = self._enc_channels * self._enc_h * self._enc_w

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_var = nn.Linear(flat_dim, latent_dim)

        # ---- Build Decoder ----
        self.decoder_input = nn.Linear(latent_dim, flat_dim)

        rev_dims = list(hidden_dims)
        rev_dims.reverse()

        modules = []
        for i in range(len(rev_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        rev_dims[i], rev_dims[i + 1],
                        kernel_size=3, stride=2, padding=1, output_padding=1,
                    ),
                    nn.BatchNorm2d(rev_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )
        self.decoder = nn.Sequential(*modules)

        if output_activation == "sigmoid":
            final_act = nn.Sigmoid()
        elif output_activation == "tanh":
            final_act = nn.Tanh()
        else:
            raise ValueError(f"Unknown output_activation: {output_activation!r}")

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                rev_dims[-1], rev_dims[-1],
                kernel_size=3, stride=2, padding=1, output_padding=1,
            ),
            nn.BatchNorm2d(rev_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(rev_dims[-1], out_channels=in_channels, kernel_size=3, padding=1),
            final_act,
        )

        if image_size % (2 ** len(hidden_dims)) != 0:
            raise ValueError(
                f"image_size={image_size} is not divisible by 2**len(hidden_dims)="
                f"{2 ** len(hidden_dims)}. Encoder/decoder spatial shapes will not "
                f"match. Choose an image_size that is a multiple of this value, or "
                f"change hidden_dims."
            )

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        log_var = torch.clamp(log_var, self.logvar_clamp[0], self.logvar_clamp[1])
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self._enc_channels, self._enc_h, self._enc_w)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def sample(self, num_samples: int, current_device) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        return self.decode(z)

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]