import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
import json
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from path_utils import MODELS, CHARTS


# Generator and Discriminator classes (from DCGAN architecture in project_problem.md)
class Generator(nn.Module):
    def __init__(self, noise_dim=100, img_channels=1, img_size=28):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim

        self.model = nn.Sequential(
            # Input: Z latent vector, shape: [batch, noise_dim]
            nn.Linear(noise_dim, 7 * 7 * 256),
            nn.BatchNorm1d(7 * 7 * 256),
            nn.ReLU(True),
            # Reshape to [batch, 256, 7, 7]
            nn.Unflatten(1, (256, 7, 7)),
            # State: [batch, 256, 7, 7]
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State: [batch, 128, 14, 14]
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State: [batch, 64, 28, 28]
            nn.ConvTranspose2d(64, img_channels, 3, 1, 1, bias=False),
            nn.Tanh(),
            # Output: [batch, img_channels, 28, 28]
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, img_channels=1, img_size=28):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Input: [batch, img_channels, 28, 28]
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: [batch, 64, 14, 14]
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State: [batch, 128, 7, 7]
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State: [batch, 256, 7, 7]
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        return self.model(img)


def load_generator(model_path=None):
    """Load the trained generator model"""
    if model_path is None:
        model_path = MODELS / "generator.pkl"

    generator = Generator()
    if model_path.exists():
        generator.load_state_dict(torch.load(model_path, map_location="cpu"))
        generator.eval()
    else:
        # Initialize with weights if no saved model exists (DCGAN standard)
        def weights_init(m):
            classname = m.__class__.__name__
            if "Conv" in classname:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif "BatchNorm" in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        generator.apply(weights_init)

    return generator


def load_discriminator(model_path=None):
    """Load the trained discriminator model"""
    if model_path is None:
        model_path = MODELS / "discriminator.pkl"

    discriminator = Discriminator()
    if model_path.exists():
        discriminator.load_state_dict(torch.load(model_path, map_location="cpu"))
        discriminator.eval()
    else:
        # Initialize with weights if no saved model exists (DCGAN standard)
        def weights_init(m):
            classname = m.__class__.__name__
            if "Conv" in classname:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif "BatchNorm" in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        discriminator.apply(weights_init)

    return discriminator


def generate_images(generator, n=64, noise_dim=100, fixed_z=None):
    """Generate n images using the generator"""
    if fixed_z is not None:
        z = fixed_z
    else:
        z = torch.randn(n, noise_dim)

    with torch.no_grad():
        fake_images = generator(z)

    # Convert from [-1, 1] to [0, 1] for display
    fake_images = (fake_images + 1) / 2
    fake_images = fake_images.clamp(0, 1)

    # Convert to PIL Image grid
    fake_images = fake_images.cpu().numpy()

    # Create a grid of images (8x8 for 64 images)
    grid_size = int(np.ceil(np.sqrt(n)))
    img_grid = np.ones((grid_size * 28, grid_size * 28), dtype=np.float32)

    for i in range(min(n, grid_size * grid_size)):
        row = (i // grid_size) * 28
        col = (i % grid_size) * 28
        img_grid[row : row + 28, col : col + 28] = fake_images[i, 0]

    # Convert to PIL Image
    img_grid = (img_grid * 255).astype(np.uint8)
    return Image.fromarray(img_grid, mode="L")


def get_training_curves():
    """Get training loss curves from saved files"""
    g_loss_path = MODELS / "g_losses.json"
    d_loss_path = MODELS / "d_losses.json"

    g_losses = []
    d_losses = []

    if g_loss_path.exists():
        with open(g_loss_path, "r") as f:
            g_losses = json.load(f)

    if d_loss_path.exists():
        with open(d_loss_path, "r") as f:
            d_losses = json.load(f)

    return {"g_losses": g_losses, "d_losses": d_losses}


def get_epoch_grids():
    """Get list of saved epoch grid image paths"""
    grids_dir = CHARTS / "generated_grids"
    if not grids_dir.exists():
        return []

    grid_files = sorted(grids_dir.glob("epoch_*.png"))
    return [str(f) for f in grid_files]


def get_mode_coverage():
    """Get mode coverage analysis (simplified for demo)"""
    # In a real implementation, this would use a classifier to analyze generated images
    # For now, return dummy data showing uniform distribution across 10 classes
    return {
        "0": 10,
        "1": 10,
        "2": 10,
        "3": 10,
        "4": 10,
        "5": 10,
        "6": 10,
        "7": 10,
        "8": 10,
        "9": 10,
    }


def save_training_losses(g_losses, d_losses):
    """Save training losses to files"""
    with open(MODELS / "g_losses.json", "w") as f:
        json.dump(g_losses, f)

    with open(MODELS / "d_losses.json", "w") as f:
        json.dump(d_losses, f)


def save_epoch_grid(generator, epoch, fixed_z=None, noise_dim=100):
    """Save a grid of generated images for an epoch"""
    grids_dir = CHARTS / "generated_grids"
    grids_dir.mkdir(parents=True, exist_ok=True)

    # Generate images using fixed z for consistent monitoring
    grid_img = generate_images(generator, n=64, fixed_z=fixed_z, noise_dim=noise_dim)
    grid_img.save(grids_dir / f"epoch_{epoch:03d}.png")
