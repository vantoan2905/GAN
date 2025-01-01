import argparse
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CustomDataset
from generator import AdaptiveGenerator
from discriminator import AdaptiveDiscriminator
from train import GANTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data", help="Path to dataset directory")
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimensionality of latent space")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--image_size", type=int, default=128, help="Image size (128 or 256)")
    parser.add_argument("--channels", type=int, default=3, help="Number of channels in images")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--save_interval", type=int, default=50, help="Interval for saving checkpoints")
    parser.add_argument("--sample_interval", type=int, default=10, help="Interval for generating and saving samples")
    return parser

def main():
    # Parse arguments
    args = parse_args().parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * args.channels, (0.5,) * args.channels)
    ])
    
    # Setup dataset and dataloader
    dataset = CustomDataset(image_dir=args.path, transform=transform, image_size=(int(args.image_size), int(args.image_size)))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    
    # Initialize models
    generator = AdaptiveGenerator(latent_dim=args.latent_dim, output_size=int(args.image_size), channels=args.channels)
    discriminator = AdaptiveDiscriminator(image_size=int(args.image_size), channels=args.channels)
    
    # Setup trainer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"gan_training_{timestamp}"
    
    trainer = GANTrainer(generator=generator, discriminator=discriminator, dataloader=dataloader, latent_dim=args.latent_dim, save_dir=save_dir, learning_rate=args.learning_rate)
    
    # Start training
    trainer.train(num_epochs=args.num_epochs, save_interval=args.save_interval, sample_interval=args.sample_interval)

if __name__ == "__main__":
    main()
