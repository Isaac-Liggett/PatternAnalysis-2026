import torch
import torch.nn as nn

class StableDiffusion(nn.Module):
    def __init__(self, n_steps=50, model_dict=None):
        super().__init__()

        if model_dict is not None:
            self.load_state_dict(model_dict)

    def forward(self, x):
        pass

class DenoisingUNET(nn.Module):
    def __init__(self, n_steps=50):
        super().__init__()
    
    def forward(self, x):
        pass   