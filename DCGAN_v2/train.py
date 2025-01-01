import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision

class GANVisualizer:
    def __init__(self, device):
        self.device = device
    
    def show_image_grid(self, images, save_path=None, normalize=True):
        if normalize:
            images = (images + 1) / 2
        images = images.detach().cpu()
        grid = torchvision.utils.make_grid(images, nrow=4, padding=2, normalize=False)
        grid = grid.numpy().transpose((1, 2, 0))
        grid = grid.clip(0, 1)
        
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.imshow(grid)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

class GANTrainer:
    def __init__(self, generator, discriminator, dataloader, latent_dim, 
                 save_dir, device=None, learning_rate=0.0002):
        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader
        self.latent_dim = latent_dim
        self.save_dir = Path(save_dir)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.visualizer = GANVisualizer(self.device)
        
        self.setup()
    
    def setup(self):
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        # lr_generator = 0.0002
        # lr_discriminator = 0.0002
        # beta1 = 0.3 if 
        # beta2 = 0.999
        self.criterion = nn.BCELoss()
        self.g_optimizer = optim.Adam(self.generator.parameters(), 
                                    lr = self.learning_rate, betas=(0.3, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), 
                                    lr = self.learning_rate, betas=(0.3, 0.999))
        
        self.setup_directories()
        
        self.g_losses = []
        self.d_losses = []
    
    def setup_directories(self):
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir / "samples", exist_ok=True)
        os.makedirs(self.save_dir / "checkpoints", exist_ok=True)

    def train_epoch(self, epoch, num_epochs):
        self.generator.train()
        self.discriminator.train()

        progress_bar = tqdm(self.dataloader, desc=f'Epoch {epoch}/{num_epochs}')
        epoch_g_loss = 0
        epoch_d_loss = 0

        for batch_idx, real_images in enumerate(progress_bar):
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)

            # Train Discriminator
            self.d_optimizer.zero_grad()


            ##############################################################################################
            ##############################################################################################
            # Labeling

            label_real = torch.ones(batch_size, 1).to(self.device)
            label_fake = torch.zeros(batch_size, 1).to(self.device)

            # Generate noise and create fake images
            noise = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
            fake_images = self.generator(noise)

            # Ensure fake images have the same dimensions as real images
            if fake_images.size() != real_images.size():
                fake_images = nn.functional.interpolate(fake_images, size=real_images.size()[2:])

            output_real = self.discriminator(real_images)
            d_loss_real = self.criterion(output_real, label_real)

            output_fake = self.discriminator(fake_images.detach())
            d_loss_fake = self.criterion(output_fake, label_fake)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            
            ###################################################################################################
            ###################################################################################################
            # Discriminator optimization
            self.d_optimizer.step()

            # Train Generator
            self.g_optimizer.zero_grad()
            output_fake = self.discriminator(fake_images)
            g_loss = self.criterion(output_fake, label_real)
            g_loss.backward()


            ###################################################################################################
            ###################################################################################################
            # Generator optimization
            self.g_optimizer.step()
            self.g_optimizer.step()




            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            # G_z =
            # D_x = 
            # D_G_z = 
            progress_bar.set_postfix({'D_loss': f'{d_loss.item():.4f}', 'G_loss': f'{g_loss.item():.4f}'})

        epoch_g_loss /= len(self.dataloader)
        epoch_d_loss /= len(self.dataloader)

        return epoch_g_loss, epoch_d_loss

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses
        }
        
        filename = self.save_dir / "checkpoints" / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, filename)
        
        if is_best:
            best_filename = self.save_dir / "checkpoints" / 'best_model.pth'
            torch.save(checkpoint, best_filename)

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.g_losses, label='Generator Loss')
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.save_dir / 'loss_plot.png')
        plt.close()

    def train(self, num_epochs, save_interval=50, sample_interval=10):
        """
        Main training loop for the GAN
        """
        for epoch in range(num_epochs):
            # Train for one epoch
            g_loss, d_loss = self.train_epoch(epoch + 1, num_epochs)
            self.g_losses.append(g_loss)
            self.d_losses.append(d_loss)
            
            # Generate and save sample images
            if (epoch + 1) % sample_interval == 0:
                with torch.no_grad():
                    sample_noise = torch.randn(16, self.latent_dim, 1, 1).to(self.device)
                    fake_samples = self.generator(sample_noise)
                    self.visualizer.show_image_grid(
                        fake_samples, 
                        save_path=self.save_dir / f'samples/epoch_{epoch+1}.png'
                    )
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1)
                self.plot_losses()
        
        # Save final model
        self.save_checkpoint(num_epochs, is_best=True)
        self.plot_losses()
        print("Training completed!")