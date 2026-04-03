import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import numpy as np
import time
from tqdm import tqdm
import sys
from pathlib import Path

# Fix path to import from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))
from path_utils import DATA_RAW, MODELS, CHARTS
from dashboard_core import Generator, Discriminator, save_training_losses, save_epoch_grid

# DCGAN Weight Initialization
def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def stop_4_dataset(batch_size=128):
    print("\n[STOP 4] Loading Fashion-MNIST Dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1] for Tanh
    ])
    
    dataset = datasets.FashionMNIST(root=DATA_RAW, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print(f"Dataset Size: {len(dataset)} samples | Batch Size: {batch_size}")
    return dataloader

def stop_5_architecture(device):
    print("\n[STOP 5] Initializing DCGAN Components...")
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    print("Networks Initialized: Generator & Discriminator")
    return netG, netD

def stop_6_optimizers():
    # Standard DCGAN params: lr=0.0002, betas=(0.5, 0.999)
    print("\n[STOP 6] Configured Adam Optimizers (DCGAN Standard)...")
    return 0.0002, (0.5, 0.999)

def train_gan(num_epochs=10, batch_size=128, device="cuda" if torch.cuda.is_available() else "cpu"):
    dataloader = stop_4_dataset(batch_size)
    netG, netD = stop_5_architecture(device)
    lr, betas = stop_6_optimizers()
    
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, 100, device=device) # For progress monitoring
    
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=betas)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=betas)
    
    img_list = []
    G_losses = []
    D_losses = []
    
    print(f"\n[STOP 7-9] Starting Adversarial Training Loop on {device}...")
    
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for i, (data, _) in enumerate(progress_bar):
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            
            # --- Label Smoothing (0.9 instead of 1.0) ---
            label = torch.full((b_size,), 0.9, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            noise = torch.randn(b_size, 100, device=device)
            fake = netG(noise)
            label.fill_(0.0) # Fake label = 0
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # (2) Update G network: maximize log(D(G(z))) [Non-saturating]
            netG.zero_grad()
            label.fill_(1.0) # Fake labels are real for generator cost
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            # Stats tracking
            if i % 100 == 0:
                progress_bar.set_postfix({"Loss_D": f"{errD.item():.3f}", "Loss_G": f"{errG.item():.3f}"})
            
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        # Save progress visualization every 5 epochs or last epoch
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            save_epoch_grid(netG, epoch + 1, fixed_z=fixed_noise)
            
    # Save Final Models
    torch.save(netG.state_dict(), MODELS / "generator.pkl")
    torch.save(netD.state_dict(), MODELS / "discriminator.pkl")
    save_training_losses(G_losses, D_losses)
    
    print("\n[TRAINING COMPLETE] Artifacts saved to models/ and charts/")
    return G_losses, D_losses

if __name__ == "__main__":
    train_gan(num_epochs=1) # Fast run for testing
