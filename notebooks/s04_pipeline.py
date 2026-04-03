import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from notebooks.s01_theory import run_theory
from notebooks.s02_train import train_gan
from notebooks.s03_evaluation import run_evaluation

def run_full_pipeline(num_epochs=10, auto_mode=False):
    print("\n" + "="*60)
    print("      DCGAN IMAGE GENERATION MASTER PIPELINE      ")
    print("="*60 + "\n")
    
    # 1. Theory Module
    run_theory(auto_mode=auto_mode)
    
    # 2. Training Module
    # num_epochs default to 10 for decent variety in Fashion-MNIST
    print(f"\n[PIPELINE] Starting DCGAN Training ({num_epochs} epochs)...")
    train_gan(num_epochs=num_epochs)
    
    # 3. Evaluation Module
    print("\n[PIPELINE] Starting Evaluation & Latent Interpolation...")
    run_evaluation()
    
    print("\n" + "="*60)
    print("          GAN PIPELINE EXECUTION COMPLETE          ")
    print("="*60 + "\n")
    print("Next Step: Launch Dashboard via 'streamlit run app.py'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCGAN Pipeline Runner")
    parser.add_argument("--auto", action="store_true", help="Run in non-interactive mode")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    
    args = parser.parse_args()
    run_full_pipeline(num_epochs=args.epochs, auto_mode=args.auto)
