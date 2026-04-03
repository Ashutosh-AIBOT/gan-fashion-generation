import time
from tqdm import tqdm

def print_stop(stop_num, title, explanation, auto_mode=False):
    print(f"\n{'='*60}")
    print(f"STOP {stop_num}: {title}")
    print(f"{'='*60}")
    print(f"\nEXPLANATION:\n{explanation}")
    print(f"\n{'='*60}\n")
    if not auto_mode:
        input("Press Enter to continue to the next STOP...")

def stop_1_gan_core(auto_mode=False):
    explanation = (
        "The GAN (Generative Adversarial Network) consists of two neural networks:\n"
        "1. The Generator (G): Learns to create 'fake' images from random noise.\n"
        "2. The Discriminator (D): Learns to distinguish between 'real' images and 'fake' ones.\n\n"
        "The objective is a Minimax game: min_G max_D [log D(x) + log(1 - D(G(z)))]\n"
        "D tries to maximize its accuracy, while G tries to minimize it (by fooling D)."
    )
    print_stop(1, "GAN Core Concept", explanation, auto_mode)

def stop_2_training_dynamics(auto_mode=False):
    explanation = (
        "GAN training is highly unstable. We use two tricks for stability:\n"
        "1. Real labels = 1.0, Fake labels = 0.0 for D.\n"
        "2. For G, we use 'Non-saturating Loss': Instead of minimizing log(1-D(G(z))),\n"
        "   we maximize log D(G(z)). This provides stronger gradients early on.\n"
        "3. Adam optimizer with specific betas (0.5, 0.999) is standard for DCGAN."
    )
    print_stop(2, "Training Dynamics", explanation, auto_mode)

def stop_3_mode_collapse(auto_mode=False):
    explanation = (
        "Mode Collapse occurs when G finds a 'shortcut'—generating a single type of image\n"
        "that consistently fools D. For Fashion-MNIST, if G only generates sneakers\n"
        "and ignores shirts/bags, that is Mode Collapse.\n\n"
        "We monitor this by saving a grid of 64 images every few epochs to check diversity."
    )
    print_stop(3, "Mode Collapse & Stability", explanation, auto_mode)

def run_theory(auto_mode=False):
    print("Initializing GAN Theory Module...")
    stop_1_gan_core(auto_mode)
    stop_2_training_dynamics(auto_mode)
    stop_3_mode_collapse(auto_mode)
    print("\n[THEORY MODULE COMPLETE]")

if __name__ == "__main__":
    run_theory(auto_mode=False)
