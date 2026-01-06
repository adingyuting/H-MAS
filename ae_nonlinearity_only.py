
# -*- coding: utf-8 -*-
"""
Autoencoder-only visualization: showing it captures *nonlinear* structure
— no baseline comparisons.

What this script does:
1) Load AMI series from "normal user.xlsx" (sheet with 'Date' + user columns).
2) Build sliding windows of length W (e.g., 96 samples) and standardize per-window.
3) Train a tiny MLP Autoencoder with 2D latent (PyTorch).
4) Plot three figures:
   (A) Input vs AE reconstruction (several random windows) + residual curves.
   (B) Latent scatter (2D) color-coded by a "peakiness" statistic -> curved manifold.
   (C) Latent traversal: decode points along a curved path in latent space to show
       nonlinear morphing of waveform shapes.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===== User settings =====
INPUT_XLSX = "normal user.xlsx"
DATE_COL   = "Date"
USER_COL   = None
W         = 96
STRIDE    = 8
EPOCHS    = 80
BATCH     = 128
LR        = 1e-3
LATENT_D  = 2
OUTDIR    = Path("ae_nonlinearity_only")

# ====== Load data ======
df = pd.read_excel(INPUT_XLSX)
if DATE_COL in df.columns:
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.sort_values(DATE_COL)
num_cols = [c for c in df.columns if c != DATE_COL and pd.api.types.is_numeric_dtype(df[c])]
if USER_COL is None:
    assert len(num_cols) > 0, "No numeric user column found."
    USER_COL = num_cols[0]
x = df[USER_COL].astype(float).to_numpy()
x = pd.Series(x).interpolate().bfill().ffill().to_numpy()

# ====== Build sliding windows ======
def sliding_windows(arr, win, stride):
    out = []
    idxs = []
    i = 0
    while i + win <= len(arr):
        out.append(arr[i:i+win])
        idxs.append(i)
        i += stride
    return np.stack(out, axis=0), np.array(idxs)

X, start_idx = sliding_windows(x, W, STRIDE)

# standardize per-window
mu = X.mean(axis=1, keepdims=True)
sd = X.std(axis=1, keepdims=True) + 1e-8
Xn = (X - mu) / sd

# ====== Tiny MLP Autoencoder (PyTorch) ======
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AE(nn.Module):
    def __init__(self, in_dim=W, z_dim=LATENT_D):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 32),    nn.ReLU(inplace=True),
            nn.Linear(32, z_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 32),   nn.ReLU(inplace=True),
            nn.Linear(32, 128),     nn.ReLU(inplace=True),
            nn.Linear(128, in_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        xh = self.decoder(z)
        return xh, z

ae = AE(in_dim=W, z_dim=LATENT_D).to(device)
opt = torch.optim.Adam(ae.parameters(), lr=LR)
crit = nn.MSELoss()

X_t = torch.from_numpy(Xn).float()
loader = DataLoader(TensorDataset(X_t), batch_size=BATCH, shuffle=True, drop_last=False)

ae.train()
for ep in range(1, EPOCHS+1):
    total = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        opt.zero_grad()
        recon, _ = ae(batch)
        loss = crit(recon, batch)
        loss.backward()
        opt.step()
        total += loss.item()*len(batch)
    if ep % max(1, EPOCHS//10) == 0:
        print(f"[AE] epoch {ep:03d}/{EPOCHS}  loss={total/len(X_t):.6f}")

OUTDIR.mkdir(parents=True, exist_ok=True)

# ====== Encode & Reconstruct ======
ae.eval()
with torch.no_grad():
    X_rec, Z = ae(X_t.to(device))
X_rec = X_rec.cpu().numpy()
Z = Z.cpu().numpy()
X_rec_real = X_rec * sd + mu

# ====== (A) Input vs AE reconstruction + residuals ======
np.random.seed(0)
sel = np.random.choice(len(X), size=min(6, len(X)), replace=False)
fig, axes = plt.subplots(len(sel), 1, figsize=(10, 2.6*len(sel)), sharex=True)
if len(sel) == 1: axes = [axes]
t = np.arange(W)
for ax, idx in zip(axes, sel):
    ax.plot(t, X[idx], linewidth=1.8, label="Input window")
    ax.plot(t, X_rec_real[idx], linewidth=1.8, linestyle="--", label="AE reconstruction")
    res = X[idx] - X_rec_real[idx]
    ax2 = ax.twinx()
    ax2.plot(t, res, alpha=0.5, linewidth=1.0)
    ax.set_ylabel("kWh")
    ax.grid(True, alpha=0.3)
axes[0].legend(loc="upper right")
axes[-1].set_xlabel("Samples within window")
fig.suptitle("(A) AE reconstruction preserves nonlinear shapes while denoising")
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig(OUTDIR/"A_input_vs_ae_reconstruction.png", dpi=300); plt.close(fig)

# ====== (B) 2D latent scatter — color by 'peakiness' ======
rng = np.ptp(X, axis=1) / (np.std(X, axis=1)+1e-8)
c = (rng - rng.min()) / (rng.max() - rng.min() + 1e-8)
plt.figure(figsize=(6.6,5.6))
plt.scatter(Z[:,0], Z[:,1], s=10, c=c, cmap="viridis")
plt.colorbar(label="Peakiness (normalized range)")
plt.title("(B) AE 2D latent organizes windows along a curved manifold")
plt.xlabel("z1"); plt.ylabel("z2"); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(OUTDIR/"B_latent_scatter_peakiness.png", dpi=300); plt.close()

# ====== (C) Latent traversal: decode along a curved path ======
z0 = Z.mean(axis=0)
scale = np.std(Z, axis=0) + 1e-6
R = 1.2 * np.max(scale)
theta = np.linspace(-1.1*np.pi/2, 1.1*np.pi/2, 7)
Z_path = np.stack([z0[0] + R*np.cos(theta), z0[1] + R*np.sin(theta)], axis=1).astype(np.float32)

import torch
with torch.no_grad():
    dec = ae.decoder(torch.from_numpy(Z_path).to(device)).cpu().numpy()
dec_real = dec * sd.mean() + mu.mean()

plt.figure(figsize=(10,6))
for i in range(len(Z_path)):
    plt.plot(dec_real[i], linewidth=1.8, label=f"decode {i+1}")
plt.title("(C) Decoded shapes along a curved latent path — nonlinear morphing")
plt.xlabel("Samples within window"); plt.ylabel("kWh"); plt.grid(True, alpha=0.3)
plt.legend(ncol=3)
plt.tight_layout(); plt.savefig(OUTDIR/"C_latent_traversal_decoded_shapes.png", dpi=300); plt.close()

print("[Done] Saved figures to", OUTDIR.resolve())
