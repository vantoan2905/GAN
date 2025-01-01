import torch
import torch.nn as nn
import math

class AdaptiveDiscriminator(nn.Module):
    def __init__(self, image_size, channels=3, features_d=64):
        super(AdaptiveDiscriminator, self).__init__()
        
        # Calculate the number of layers needed
        self.num_layers = int(math.log2(image_size)) - 2
        # Create the layers
        layers = []
        
        # First layer
        layers.append(nn.Conv2d(channels, features_d, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Intermediate layers
        in_channels = features_d
        for i in range(self.num_layers - 1):
            out_channels = min(in_channels * 2, 1024)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_channels = out_channels
            
        # Final layer
        layers.append(nn.Conv2d(in_channels, 1, 4, 1, 0, bias=False))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x).view(-1, 1)
