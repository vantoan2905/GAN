# models/generator.py
import torch
import torch.nn as nn
import math

class AdaptiveGenerator(nn.Module):
    def __init__(self, latent_dim, output_size, channels=3):
        super(AdaptiveGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size
        
        # Tính số lớp deconv cần thiết
        self.num_layers = int(math.log2(output_size)) - 2
        
        # Tạo các layers
        layers = []
        
        # Tính số channels cho layer đầu tiên
        initial_channels = min(2048, 1024 * (2 ** (self.num_layers - 6)))
        
        # Layer đầu tiên từ latent vector
        layers.extend([
            nn.ConvTranspose2d(latent_dim, initial_channels, 4, 1, 0, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(True)
        ])
        
        # Các layer trung gian
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
        
        # Layer cuối cùng
        layers.extend([
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        ])
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z):
        z = z.view(-1, self.latent_dim, 1, 1)
        return self.model(z)

# models/discriminator.py
class AdaptiveDiscriminator(nn.Module):
    def __init__(self, image_size, channels=3, features_d=64):
        super(AdaptiveDiscriminator, self).__init__()
        
        # Tính số lớp conv cần thiết
        self.num_layers = int(math.log2(image_size)) - 2
        
        # Tạo các layers
        layers = []
        
        # Layer đầu tiên
        layers.append(nn.Conv2d(channels, features_d, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Số channels cho mỗi layer
        in_channels = features_d
        
        # Các layer trung gian
        for i in range(self.num_layers - 1):
            out_channels = min(in_channels * 2, 1024)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_channels = out_channels
            
        # Layer cuối cùng
        layers.append(nn.Conv2d(in_channels, 1, 4, 1, 0, bias=False))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x).view(-1, 1)

# utils/visualizer.py
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

# dataset/custom_dataset.py
class CustomDataset(torch.utils.data.Dataset):
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

# trainer/gan_trainer.py
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
        
        # Setup
        self.setup()
        
    def setup(self):
        # Move models to device
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        
        # Initialize optimizers and criterion
        self.criterion = nn.BCELoss()
        self.g_optimizer = optim.Adam(self.generator.parameters(), 
                                    lr=self.learning_rate, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), 
                                    lr=self.learning_rate, betas=(0.5, 0.999))
        
        # Setup directories and logging
        self.setup_directories()
        self.setup_logging()
        
        # Initialize metrics tracking
        self.g_losses = []
        self.d_losses = []
        self.fid_scores = []
        self.inception_scores = []
        
        # Initialize visualizer
        self.visualizer = GANVisualizer(self.save_dir / 'samples')
        
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
            label_real = torch.ones(batch_size, 1).to(self.device)
            label_fake = torch.zeros(batch_size, 1).to(self.device)
            
            output_real = self.discriminator(real_images)
            d_loss_real = self.criterion(output_real, label_real)
            
            noise = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_images = self.generator(noise)
            output_fake = self.discriminator(fake_images.detach())
            d_loss_fake = self.criterion(output_fake, label_fake)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.d_optimizer.step()
            
            # Train Generator
            self.g_optimizer.zero_grad()
            output_fake = self.discriminator(fake_images)
            g_loss = self.criterion(output_fake, label_real)
            g_loss.backward()
            self.g_optimizer.step()
            
            # Update metrics
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}'
            })
        
        # Calculate average epoch losses
        epoch_g_loss /= len(self.dataloader)
        epoch_d_loss /= len(self.dataloader)
        
        return epoch_g_loss, epoch_d_loss
    
    def train(self, num_epochs, save_interval=50, sample_interval=10):
        self.logger.info(f"Starting training on device: {self.device}")
        
        try:
            for epoch in range(num_epochs):
                g_loss, d_loss = self.train_epoch(epoch + 1, num_epochs)
                
                # Save metrics
                self.g_losses.append(g_loss)
                self.d_losses.append(d_loss)
                
                # Log progress
                self.logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}"
                )
                
                # Generate and save samples
                if (epoch + 1) % sample_interval == 0:
                    with torch.no_grad():
                        fake_samples = self.generator(
                            torch.randn(16, self.latent_dim).to(self.device)
                        )
                        self.visualizer.show_image_grid(
                            fake_samples,
                            save_path=self.save_dir / f'samples/epoch_{epoch+1}.png'
                        )
                
                # Save checkpoint
                if (epoch + 1) % save_interval == 0:
                    self.save_checkpoint(epoch + 1)
                    self.plot_losses()
                    self.save_metrics()
                    
        except Exception as e:
            self.logger.error(f"Training interrupted: {str(e)}")
            self.save_checkpoint(epoch + 1)
            raise e
        
        # Final saves
        self.save_checkpoint(num_epochs, is_best=True)
        self.plot_losses()
        self.save_metrics()
        self.logger.info("Training completed successfully!")

# main.py
def main():
    # Parse arguments
    args = parse_args()
    
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
    dataset = CustomDataset(
        image_dir=args.path,
        transform=transform,
        image_size=(args.image_size, args.image_size)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Initialize models
    generator = AdaptiveGenerator(
        latent_dim=args.latent_dim,
        output_size=args.image_size,
        channels=args.channels
    )
    discriminator = AdaptiveDiscriminator(
        image_size=args.image_size,
        channels=args.channels
    )
    
    # Setup trainer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"gan_training_{timestamp}"
    
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        latent_dim=args.latent_dim,
        save_dir=save_dir,
        learning_rate=args.learning_rate
    )
    
    # Start training
    trainer.train(
        num_epochs=args.num_epochs,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval
    )

if __name__ == "__main__":
    main()