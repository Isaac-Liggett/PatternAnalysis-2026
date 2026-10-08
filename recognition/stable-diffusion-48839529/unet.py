import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, ConvTranspose2d
from torch.nn.functional import relu, softmax, one_hot
from torchvision import models

class DenoisingUNet(nn.Module):
    def __init__(self, channels:int=1, n_steps:int=50):
        super().__init__()

        self.register_buffer('denoising_steps', n_steps)
 
        # --- Encoder ---
        self.e11 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
 
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
 
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2)
 
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2)
 
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(1024)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(1024)
 
        # --- Decoder ---
        # NOTE: up1-up4 (ConvTranspose2d) are left without norm/activation,
        # consistent with standard practice — see explanation above.
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d12 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bnd12 = nn.BatchNorm2d(512)
        self.d13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bnd13 = nn.BatchNorm2d(512)
 
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d22 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bnd22 = nn.BatchNorm2d(256)
        self.d23 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bnd23 = nn.BatchNorm2d(256)
 
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d32 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bnd32 = nn.BatchNorm2d(128)
        self.d33 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bnd33 = nn.BatchNorm2d(128)
 
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d42 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bnd42 = nn.BatchNorm2d(64)
        self.d43 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bnd43 = nn.BatchNorm2d(64)
 
        # Output layer: no norm, no activation — raw logits for the loss.
        self.outconv = nn.Conv2d(64, channels, kernel_size=1)
 
    def forward(self, x):
        # Encoder
        xe11 = relu(self.bn11(self.e11(x)))
        xe12 = relu(self.bn12(self.e12(xe11)))
        xp1 = self.pool1(xe12)
 
        xe21 = relu(self.bn21(self.e21(xp1)))
        xe22 = relu(self.bn22(self.e22(xe21)))
        xp2 = self.pool2(xe22)
 
        xe31 = relu(self.bn31(self.e31(xp2)))
        xe32 = relu(self.bn32(self.e32(xe31)))
        xp3 = self.pool3(xe32)
 
        xe41 = relu(self.bn41(self.e41(xp3)))
        xe42 = relu(self.bn42(self.e42(xe41)))
        xp4 = self.pool4(xe42)
 
        xe51 = relu(self.bn51(self.e51(xp4)))
        xe52 = relu(self.bn52(self.e52(xe51)))
 
        # Decoder
        xd11 = self.up1(xe52)
        xu11 = torch.cat([xd11, xe42], dim=1)
        xd12 = relu(self.bnd12(self.d12(xu11)))
        xd13 = relu(self.bnd13(self.d13(xd12)))
 
        xd21 = self.up2(xd13)
        xu22 = torch.cat([xd21, xe32], dim=1)
        xd22 = relu(self.bnd22(self.d22(xu22)))
        xd23 = relu(self.bnd23(self.d23(xd22)))
 
        xd31 = self.up3(xd23)
        xu32 = torch.cat([xd31, xe22], dim=1)
        xd32 = relu(self.bnd32(self.d32(xu32)))
        xd33 = relu(self.bnd33(self.d33(xd32)))
 
        xd41 = self.up4(xd33)
        xu42 = torch.cat([xd41, xe12], dim=1)
        xd42 = relu(self.bnd42(self.d42(xu42)))
        xd43 = relu(self.bnd43(self.d43(xd42)))
 
        return self.outconv(xd43)

    def denoise(self, z):
        x = z
        for step in range(self.denoising_steps):
            x = self(x)
        
        return x