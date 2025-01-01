import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

class GANVisualizer:
    def __init__(self, output_dir="generated_images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def denormalize(self, img):
        return (img + 1) / 2
    
    def show_single_image(self, img, title=None, save_path=None):
        if torch.is_tensor(img):
            img = self.denormalize(img)
            img = img.detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        if title:
            plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()
    
    def show_image_grid(self, imgs, nrow=8, title=None, save_path=None):
        img_grid = make_grid(self.denormalize(imgs), nrow=nrow, padding=2, normalize=False)
        img_grid = img_grid.detach().cpu().numpy()
        img_grid = np.transpose(img_grid, (1, 2, 0))
        
        plt.figure(figsize=(15, 15))
        plt.imshow(img_grid)
        if title:
            plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()
