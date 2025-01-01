import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils as vutils
from datetime import datetime
import os
import argparse
from generator import Generator, Generator_256
from discriminator import Discriminator, Discriminator_256
from customDataset import CustomDataset


# Utility functions
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(generator_class, latent_dim, channels, path):
    model = generator_class(latent_dim, channels)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def generate_images(generator, num_images, latent_dim, output_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(device)
        fake_images = generator(z)
        fake_images = (fake_images + 1) / 2
        if output_path:
            vutils.save_image(fake_images, output_path, normalize=True, nrow=4)
    return fake_images


# Training function
def train_gan(generator, discriminator, dataloader, num_epochs, latent_dim, save_interval, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, discriminator = generator.to(device), discriminator.to(device)
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    os.makedirs(save_dir, exist_ok=True)

    try:
        for epoch in range(num_epochs):
            for real_images in dataloader:
                batch_size = real_images.size(0)
                real_images = real_images.to(device)
                labels_real, labels_fake = torch.ones(batch_size, 1).to(device), torch.zeros(batch_size, 1).to(device)

                # Train Discriminator
                d_optimizer.zero_grad()
                d_loss_real = criterion(discriminator(real_images), labels_real)
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_images = generator(z)
                d_loss_fake = criterion(discriminator(fake_images.detach()), labels_fake)
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()

                # Train Generator
                g_optimizer.zero_grad()
                g_loss = criterion(discriminator(fake_images), labels_real)
                g_loss.backward()
                g_optimizer.step()

            if (epoch + 1) % save_interval == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")
                generate_images(generator, 1, latent_dim, os.path.join(save_dir, f"samples_epoch_{epoch+1}.png"))

        save_model(generator, os.path.join(save_dir, "generator_final.pth"))
    except Exception as e:
        print(f"Training interrupted: {str(e)}")
        save_model(generator, os.path.join(save_dir, "generator_error_state.pth"))
    return generator


# Main function
def main(path, latent_dim, batch_size, num_epochs, image_size, channels, generator_class, discriminator_class):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CustomDataset(image_dir=path, transform=transform, image_size=(image_size, image_size))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                            pin_memory=torch.cuda.is_available())
    generator = generator_class(latent_dim, channels)
    discriminator = discriminator_class(channels)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"gan_training_{timestamp}"

    trained_generator = train_gan(generator, discriminator, dataloader, num_epochs, latent_dim, 50, save_dir)
    generate_images(trained_generator, 4, latent_dim, "final_samples.png")


# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the directory containing images")
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimensionality of latent vector")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--image_size", type=int, default=128, help="Image size (128 or 256)")
    parser.add_argument("--channels", type=int, default=3, help="Number of channels in images")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    generator_class = Generator if args.image_size == 128 else Generator_256
    discriminator_class = Discriminator if args.image_size == 128 else Discriminator_256
    main(args.path, args.latent_dim, args.batch_size, args.num_epochs, args.image_size, args.channels,
         generator_class, discriminator_class)
