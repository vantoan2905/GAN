import os
import torch
import torch.nn as nn
import math

class AdaptiveGenerator(nn.Module):
    def __init__(self, latent_dim, output_size, channels=3):
        super(AdaptiveGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size
        # Calculate the number of layers
        self.num_layers = int(math.log2(output_size)) - 2
        # Create the layers
        layers = []
        
        # Calculate number of channels for the first layer
        initial_channels = min(2048, 1024 * (2 ** (self.num_layers - 6)))
        initial_channels = int(initial_channels)
        # First layer from latent vector
        layers.extend([
            nn.ConvTranspose2d(latent_dim, initial_channels, 4, 1, 0, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(True)
        ])
        
        # Intermediate layers
        in_channels = initial_channels
        for i in range(self.num_layers):
            out_channels = in_channels // 2
            if out_channels < 64:
                out_channels = 64
            
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            ])
            in_channels = out_channels
        
        # Final layer
        layers.extend([
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        ])
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z):
        z = z.view(-1, self.latent_dim, 1, 1)
        return self.model(z)
