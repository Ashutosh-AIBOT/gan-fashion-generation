import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os
import sys
from pathlib import Path

# Fix path to import from project root
sys.path.append(str(Path(__file__).resolve().parent))
from dashboard_core import (
    load_generator, generate_images, get_training_curves,
    get_epoch_grids, get_mode_coverage
)

# --- 1. APP CONFIG ---
st.set_page_config(
    page_title="GAN Fashion Generation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PREMIUM CSS ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden}
    footer {visibility: hidden}
    header {visibility: hidden}
    
    [data-testid="stSidebar"] {background-color: #0f1116; border-right: 1px solid #333}
    [data-testid="stSidebar"] * {color: #ccc; font-family: 'Inter', sans-serif}
    
    .block-container {padding-top: 2rem; padding-bottom: 2rem; background-color: #0e1117}
    h1, h2, h3 {font-weight: 700; letter-spacing: -0.03em; color: #ffffff}
    .muted-text {color: #888; font-size: 0.9rem}

    /* Generator Card */
    .gen-card {
        padding: 1.5rem;
        background: #1e1e26;
        border-radius: 8px;
        border: 1px solid #333;
        text-align: center;
        transition: transform 0.2s;
    }
    .gen-card:hover {transform: translateY(-2px); border-color: #7F77DD}

    /* KPI Cards */
    .kpi-row {display: flex; gap: 1rem; margin-bottom: 2rem}
    .kpi-card {
        flex: 1;
        padding: 1rem;
        background: #161a21;
        border-radius: 4px;
        border: 1px solid #222;
        border-bottom: 3px solid #7F77DD;
    }
    .kpi-label {font-size: 11px; color: #888; text-transform: uppercase; font-weight: 600}
    .kpi-value {font-size: 22px; font-weight: 600; color: #fff; margin-top: 4px}
    
    .stTabs [data-baseweb="tab-list"] {gap: 10px}
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        background: transparent;
        border-radius: 4px;
        color: #888;
    }
    .stTabs [aria-selected="true"] {background: #1e1e26 !important; color: #7F77DD !important}
</style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.markdown("### GAN CONSOLE v1.0")
    st.caption("Deep Convolutional Generative Adversarial Network")
    st.markdown("---")
    
    noise_dim = 100
    n_images = st.slider("Samples to Generate", 16, 64, 64, step=16)
    
    st.markdown("---")
    st.markdown("#### Hyperparameters")
    st.code("LR: 0.0002\nBetas: (0.5, 0.999)\nBatch: 128\nArch: DCGAN")
    
    st.markdown("---")
    st.button("Reset Latent", use_container_width=True)

# --- 4. HEADER ---
st.markdown("# Generative Fashion Synthesis")
st.info("**Mission**: Building a creative AI that learns to synthesize original fashion assets from random noise using Adversarial Minimax dynamics.")

st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-card">
        <div class="kpi-label">Model Resolution</div>
        <div class="kpi-value">28 x 28 px</div>
    </div>
    <div class="kpi-card" style="border-bottom-color: #1D9E75">
        <div class="kpi-label">Architecture</div>
        <div class="kpi-value">DCGAN (TransposeConv)</div>
    </div>
    <div class="kpi-card" style="border-bottom-color: #FFB347">
        <div class="kpi-label">Latent Space</div>
        <div class="kpi-value">100-Dim Gaussian</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 5. TABS ---
tab1, tab2, tab3 = st.tabs(["Active Synthesis", "Training Gallery", "Engineering Audit"])

with tab1:
    col_gen, col_info = st.columns([2, 1], gap="large")
    
    with col_gen:
        st.markdown("#### Adversarial Synthesis Reservoir")
        generator = load_generator()
        
        c1, c2 = st.columns([1, 2])
        seed = c1.number_input("Latent Seed", value=42, step=1)
        
        # Generator Button
        if c2.button("🚀 Trigger Synthesis", type="primary", use_container_width=True):
            torch.manual_seed(seed)
            grid_img = generate_images(generator, n=n_images, noise_dim=noise_dim)
            st.image(grid_img, use_container_width=True, caption=f"Generated {n_images} items from Seed {seed}")
        else:
            st.info("Click 'Trigger Generation' to synthesize fresh fashion assets from the latent space.")
            
    with col_info:
        st.markdown("#### Synthesis Logic")
        st.caption("The Generator maps a 100-dimensional normal distribution to the image manifold. Each pixel is a probability in the Fashion-MNIST domain.")
        
        st.markdown("""
        <div class="gen-card">
            <h3 style='margin:0'>Z-Vector Mapping</h3>
            <p style='color:#888; font-size: 13px'>Latent Point ➔ TransposeConv(x3) ➔ Tanh Output</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Status**: GPU Weights Loaded")
        st.markdown(f"**Seed**: {seed}")

with tab2:
    st.markdown("#### Training Progression (Epoch Grids)")
    grids = get_epoch_grids()
    if grids:
        idx = st.select_slider("Review Training History (Epochs)", options=list(range(len(grids))), value=len(grids)-1)
        st.image(grids[idx], use_container_width=True, caption=f"Fixed-Z Generator Progression at Sample Index {idx}")
    else:
        st.warning("No progression grids found. Run 's04_pipeline.py' to generate history.")
    
    st.markdown("---")
    st.markdown("#### Minimax Loss Curves")
    curves = get_training_curves()
    if curves["g_losses"]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=curves["g_losses"], name="Gen Loss", line=dict(color="#1D9E75", width=1.5)))
        fig.add_trace(go.Scatter(y=curves["d_losses"], name="Disc Loss", line=dict(color="#7F77DD", width=1.5)))
        fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No loss history found.")

with tab3:
    st.markdown("#### Adversarial Engineering Audit")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Mode Coverage Strategy")
        coverage = get_mode_coverage()
        fig_cov = px.bar(x=list(coverage.keys()), y=list(coverage.values()), labels={'x':'Category', 'y':'Samples'})
        fig_cov.update_layout(template="plotly_dark", height=250, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_cov, use_container_width=True)
        st.caption("Monitoring entropy across predicted classes to detect partial mode collapse.")

    with c2:
        st.markdown("#### Stability Components")
        st.image("charts/latent_interpolation.png", use_container_width=True) if os.path.exists("charts/latent_interpolation.png") else st.caption("Latent interpolation chart available after evaluation.")
        st.markdown("""
        - **Label Smoothing**: Soft labels (0.9) to prevent Disc overconfidence.
        - **Non-Saturating Loss**: log(D) trick for gradient health.
        - **DCGAN Init**: Normal(0, 0.02) setup for Conv layers.
        """)

# --- 6. FOOTER ---
st.markdown("---")
st.caption("Project 12: GAN Image Generation · Fashion-MNIST Deep Synthesis")
