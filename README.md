---
title: GAN Fashion Generation
emoji: 👗
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# 👗 GAN Fashion Image Generation (DCGAN)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)

A professional-grade Generative Adversarial Network (GAN) implementation for synthesizing fashion images from the Fashion-MNIST dataset. This project demonstrates a complete DCGAN (Deep Convolutional GAN) pipeline with real-time visualization and inference capabilities.

---

## 📋 Project Overview

### 1. The Problem
Generative models are at the core of modern AI. This project addresses the challenge of generating new fashion items (T-shirts, Trousers, Pullovers, etc.) by learning the underlying distribution of the Fashion-MNIST dataset through adversarial training.

### 2. The Dataset
* **Source**: Fashion-MNIST ( Zalando's article images)
* **Classes**: 10 categories (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
* **Dimensions**: 28×28 grayscale images
* **Training Samples**: 60,000

### 3. The Solution (DCGAN Architecture)

#### Generator Architecture
```
Input (100-dim noise) → Linear(100→12544) → BatchNorm → ReLU
→ ConvTranspose2D(256, 7×7) → BatchNorm → ReLU
→ ConvTranspose2D(128, 14×14) → BatchNorm → ReLU
→ ConvTranspose2D(1, 28×28) → Tanh → Output (28×28)
```

#### Discriminator Architecture
```
Input (28×28) → Conv2D(1→64) → LeakyReLU(0.2)
→ Conv2D(64→128) → BatchNorm → LeakyReLU(0.2)
→ Conv2D(128→256) → BatchNorm → LeakyReLU(0.2)
→ Flatten → Linear(12544→1) → Sigmoid → Output
```

#### Training Configuration
* **Optimizer**: Adam (lr=0.0002, β₁=0.5, β₂=0.999)
* **Batch Size**: 128
* **Epochs**: 50+
* **Loss**: Binary Cross Entropy (BCE)

---

## 📂 Project Structure

```text
GAN Image Generation/
├── app.py                    # Streamlit Dashboard (UI)
├── dashboard_core.py         # Generator & Discriminator models
├── path_utils.py             # Dynamic path management
├── Dockerfile                # Containerization
├── requirements.txt          # Dependencies
├── notebooks/                # Training pipelines
│   ├── s01_theory.py         # GAN theory & concepts
│   ├── s02_train.py          # DCGAN training loop
│   ├── s03_evaluation.py     # Evaluation metrics
│   └── s04_pipeline.py       # Master pipeline
├── models/                   # Trained weights
│   ├── generator.pkl         # Generator state dict
│   ├── discriminator.pkl     # Discriminator state dict
│   ├── g_losses.json         # Generator loss history
│   └── d_losses.json         # Discriminator loss history
├── charts/                   # Visualizations
│   ├── generated_grids/      # Epoch-wise generated images
│   └── latent_interpolation.png
└── data/                     # Fashion-MNIST dataset
    └── raw/FashionMNIST/
```

---

## 📈 Numerical Results & Performance

The GAN demonstrates successful convergence with balanced Generator and Discriminator losses:

| Metric | Value | Interpretation |
|:-------|:------|:---------------|
| **Final G Loss** | ~0.5-1.0 | Stable generator training |
| **Final D Loss** | ~0.5-1.0 | Discriminator not overpowering |
| **Epochs Trained** | 50 | Sufficient for diverse samples |

### Generated Samples
The model successfully generates all 10 fashion categories with increasing quality over training epochs (epoch_005 → epoch_050).

---

## 🚀 Live Deployment

### Dockerized Hosting (HF Spaces / Cloud)
This project is fully containerized for deployment:

1. **Build the Image**:
   ```bash
   docker build -t gan-fashion-generator .
   ```
2. **Run the Dashboard**:
   ```bash
   docker run -p 8501:7860 gan-fashion-generator
   ```

### Standard Local Execution
1. **Activate Environment**:
   ```bash
   conda activate ml-env
   ```
2. **Launch Streamlit**:
   ```bash
   streamlit run app.py
   ```

---

## 🛠️ GitHub Configuration & Workflow

```bash
# 1. Initialize & track with LFS
git init
git lfs install
git lfs track "models/*.pkl" "charts/**/*.png"

# 2. Deploy to GitHub
git add .
git commit -m "feat: Professional DCGAN implementation for Fashion-MNIST image generation"
git branch -M main
git remote add origin git@github.com:Ashutosh-AIBOT/gan-fashion-generation.git
git push -u origin main
```

---

## 🎨 Dashboard Features

The Streamlit dashboard provides:
- **📊 Training Analytics**: Real-time loss curves for G and D
- **🖼️ Live Generation**: Generate new fashion images from random noise
- **🔄 Latent Space Interpolation**: Visualize smooth transitions between generated samples
- **📈 Epoch Visualization**: Browse generated images from different training stages

---

**Developed by [Ashutosh-AIBOT](https://github.com/Ashutosh-AIBOT)**
