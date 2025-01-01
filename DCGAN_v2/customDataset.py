import torch
import os
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None, image_size=(128, 128)):
        self.image_dir = image_dir
        self.transform = transform
        self.image_size = image_size
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {self.image_files[idx]}: {str(e)}")
            return torch.zeros((3, *self.image_size))
