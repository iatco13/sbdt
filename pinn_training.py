"""
=========================================================================================
Title: Advanced PINN for Sucrose Degradation
Related Paper: "Building Cyber-Physical Resilient Regional Supply Chains: 
                Physics-Informed Neural Networks and Model Predictive Control in Stochastic Environments"
Description: 
    This script incorporates non-linear, time-dependent microbiological decay 
    and strict temperature-dependent respiration rates according to the 
    Instruction on rationing of beet mass and sugar losses
    
    The PDE embedded in the neural network now models the transition from 
    smooth early-stage respiration to exponential late-stage microbiological rotting.
=========================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# 1. SYNTHETIC DATA GENERATION (90-Day Horizon, Decreasing Autumn-Winter Temp)
# =============================================================================
print("=====================================================================")
print(" ADVANCED PINN: DATA GENERATION (USSR 1983 NORMS)")
print("=====================================================================")
data_start = time.time()

days = 90
t_full = np.linspace(0, days, 200).reshape(-1, 1)

# Temperature drops from ~12°C (285K) in Month 1 to ~2°C (275K) in Month 3
T_full = 285.0 - 10.0 * (t_full / days) + 1.5 * np.sin(0.2 * t_full)

# Thermodynamic & Microbiological Constants (Calibrated to USSR 1983 Norms)
A = 4.3e13       # Pre-exponential factor
Ea = 82000.0     # Activation energy (J/mol) calibrated for 0.01% at 1°C and 0.04% at 12°C
R = 8.314        # Universal gas constant
alpha = 0.0005   # Microbiological acceleration factor (quadratic time dependence)

# Ground Truth Generation (Euler method)
S_true = np.zeros_like(t_full)   # создаём массив той же формы
S_true[0] = 18.0                 # начальная сахаристость 18%

for i in range(1, len(t_full)):
    dt = t_full[i] - t_full[i-1]
    T_i = T_full[i-1]
    t_i = t_full[i-1]
    
    # Combined Respiration + Microbiological Decay Rate
    rate = A * np.exp(-Ea / (R * T_i)) * (1.0 + alpha * (t_i**2))
    S_true[i] = S_true[i-1] - rate * dt

# Laboratory Blind Spots (sparse samples over 3 months)
idx_lab = np.array([6, 12, 20, 30, 45, 60, 75])   # 7 точек на 90 дней
t_lab = t_full[idx_lab]
S_lab = S_true[idx_lab] + np.random.normal(0, 0.05, size=(len(idx_lab), 1))

# Convert to PyTorch Tensors
t_tensor = torch.tensor(t_full, dtype=torch.float32, requires_grad=True)
T_tensor = torch.tensor(T_full, dtype=torch.float32)
t_lab_tensor = torch.tensor(t_lab, dtype=torch.float32)
S_lab_tensor = torch.tensor(S_lab, dtype=torch.float32)

data_end = time.time()
print(f"    Time points: {len(t_full)}")
print(f"    Lab samples: {len(idx_lab)} (sporadic)")
print(f"    Generation time: {data_end - data_start:.4f} sec\n")

# =============================================================================
# 2. PHYSICS-INFORMED NEURAL NETWORK ARCHITECTURE
# =============================================================================
class PINN_AdvancedDecay(nn.Module):
    def __init__(self):
        super(PINN_AdvancedDecay, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, t):
        return self.net(t)

model = PINN_AdvancedDecay()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

# =============================================================================
# 3. TRAINING LOOP WITH ADVANCED NON-LINEAR PDE EMBEDDING
# =============================================================================
epochs = 8000
lambda_data = 1.0
lambda_physics = 1.0

print("=====================================================================")
print(" PINN TRAINING: NON-LINEAR MICROBIOLOGICAL ACCELERATION MODEL")
print("=====================================================================")
train_start = time.time()
epoch_timer = time.time()

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # A. Data Loss (L_data)
    S_pred_lab = model(t_lab_tensor)
    loss_data = torch.mean((S_pred_lab - S_lab_tensor)**2)
    
    # B. Physics Loss (L_PDE)
    S_pred_full = model(t_tensor)
    dS_dt = torch.autograd.grad(
        S_pred_full, t_tensor, 
        grad_outputs=torch.ones_like(S_pred_full), 
        create_graph=True
    )[0]   # извлекаем тензор
    
    # Advanced Decay Equation based on USSR Norms
    decay_rate = A * torch.exp(-Ea / (R * T_tensor)) * (1.0 + alpha * (t_tensor**2))
    pde_residual = dS_dt + decay_rate * S_pred_full   # Внимание: decay_rate уже включает S? Нет, это k(t), поэтому умножаем на S
    loss_physics = torch.mean(pde_residual**2)
    
    # Global Loss
    loss_total = lambda_data * loss_data + lambda_physics * loss_physics
    loss_total.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        elapsed = time.time() - epoch_timer
        print(f"    Epoch [{epoch+1}/{epochs}] | Time: {elapsed:.1f}s | "
              f"Total: {loss_total.item():.5f} | Data: {loss_data.item():.5f} | Physics: {loss_physics.item():.5f}")
        epoch_timer = time.time()

train_end = time.time()
print(f"\n[*] Training completed in {train_end - train_start:.2f} seconds.\n")

# =============================================================================
# 4. SMART TRIAGE VALIDATION (EARLY VS LATE STORAGE)
# =============================================================================
inference_start = time.time()
S_predicted = model(t_tensor).detach().numpy()
inference_end = time.time()

# Test 1: First Month (Day 15 to Day 16) - High Temp, but NO Microbial bloom
day_15_idx = 33   # индекс примерно для дня 15
day_16_idx = 36   # день 16
loss_early = S_predicted[day_15_idx].item() - S_predicted[day_16_idx].item()

# Test 2: Third Month (Day 80 to Day 81) - Low Temp, but MASSIVE Microbial bloom
day_80_idx = 177  # индекс дня 80
day_81_idx = 180  # день 81
loss_late = S_predicted[day_80_idx].item() - S_predicted[day_81_idx].item()

print("=====================================================================")
print(" SMART TRIAGE & MILP PENALTY GENERATION (USSR NORM COMPLIANCE)")
print("=====================================================================")
print(f"    [Month 1] Daily Loss (Temp ~10°C, Resp. Dominant): {loss_early:.4f} %")
print(f"    [Month 3] Daily Loss (Temp  ~3°C, Microbes Active): {loss_late:.4f} %")
print(f"    Inference time: {inference_end - inference_start:.4f} sec")
print(f"    Total script time: {train_end - data_start + (inference_end - inference_start):.2f} sec")
print("=====================================================================")

if loss_late > loss_early:
    print("    [!] ALERT: Avalanche microbiological decay detected in Month 3!")
    print("    [!] Smart Triage shifts evacuation priority to older piles.")
else:
    print("    [i] Standard respiration dominates, no emergency rerouting.")