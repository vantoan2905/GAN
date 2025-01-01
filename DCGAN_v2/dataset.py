import os
from PIL import Image
import torch
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
        
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return image
