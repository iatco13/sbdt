"""
PoC for MILP solver
Related Paper: Building Cyber-Physical Resilient Regional Supply Chains
Author: Corneliu Iatco
License: MIT

MILP Router with PINN-based Degradation, Factory Shortage Penalty, and Weighting Factors
Supports weights: w1 (transport), w2 (sugar loss), w3 (factory shortage)
"""

import pulp
import time
import numpy as np
import torch
import torch.nn as nn
import json
import warnings
warnings.filterwarnings("ignore")

# ========== USER ADJUSTABLE WEIGHTS (from paper w1...w3) ==========
w_transport = 1.0      # weight for transport cost (ton·km)
w_degradation = 10.0   # weight for sugar loss penalty (increase to prioritize sugar preservation)
w_shortage = 1.0       # weight for factory idle penalty

# ========== ECONOMIC PARAMETERS ==========
SUGAR_PRICE_PER_TON = 500.0     # USD / ton of sugar
SHORTAGE_PENALTY_PER_TON = 20.0 # USD / ton of missing supply at factory
TRANSPORT_RATE_PER_TON_KM = 0.5 # USD / (ton·km)

# ========== LOAD PINN MODEL ==========
class PINN_Sucrose(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, t):
        return self.net(t)

device = torch.device("cpu")
model = PINN_Sucrose()
model.load_state_dict(torch.load("pinn_sucrose_model.pth", map_location=device))
model.eval()

with open("pinn_physics_params.json", "r") as f:
    params = json.load(f)
INITIAL_SUCROSE = params.get("initial_sucrose", 18.0)

def sugar_loss_percent(t_days):
    if t_days < 0:
        return 0.0
    with torch.no_grad():
        t_tensor = torch.tensor([[t_days]], dtype=torch.float32)
        S_pred = model(t_tensor).item()
        loss = INITIAL_SUCROSE - S_pred
        return max(0.0, loss)

def sugar_loss_tons(biomass_tons, t_days):
    return biomass_tons * (sugar_loss_percent(t_days) / 100.0)

# ========== DATA GENERATION ==========
SPEED_KM_PER_DAY = 24.0

def generate_data_with_storage():
    np.random.seed(42)
    M = [f"m_{i}" for i in range(1, 11)]          # 10 MAUS
    F = [f"f_{i}" for i in range(1, 351)]         # 350 fields
    Z = [f"z_{i}" for i in range(1, 3)]           # 2 factories

    # Distances (km)
    D_mf = {m: {f: np.random.uniform(5.0, 50.0) for f in F} for m in M}
    D_fz = {f: {z: np.random.uniform(10.0, 100.0) for z in Z} for f in F}
    
    # Biomass (tons)
    V_f = {f: np.random.uniform(200, 800) for f in F}
    
    # MAUS daily capacity (tons)
    C_m = {m: 25000 for m in M}
    
    # Storage time at field (days) – 20% of fields are old
    storage_days = {}
    for f in F:
        if np.random.random() < 0.1:
            storage_days[f] = np.random.uniform(15, 45)
        else:
            storage_days[f] = np.random.uniform(0, 5)
    
    # Travel time (days)
    travel_time = {f: {z: D_fz[f][z] / SPEED_KM_PER_DAY for z in Z} for f in F}
    total_time = {f: {z: storage_days[f] + travel_time[f][z] for z in Z} for f in F}
    
    return M, F, Z, D_mf, D_fz, V_f, C_m, storage_days, travel_time, total_time

def compute_penalties(V_f, total_time):
    """Return penalty for each (field, factory) = sugar loss (tons) * sugar price"""
    penalties = {}
    for f in V_f.keys():
        for z in total_time[f].keys():
            t_total = total_time[f][z]
            loss_tons = sugar_loss_tons(V_f[f], t_total)
            penalties[(f, z)] = loss_tons * SUGAR_PRICE_PER_TON
    return penalties

# ========== MILP WITH WEIGHTS ==========
def main():
    print("="*70)
    print("MILP ROUTING WITH WEIGHTS (w_transport={}, w_degradation={}, w_shortage={})".format(
        w_transport, w_degradation, w_shortage))
    print("="*70)

    M, F, Z, D_mf, D_fz, V_f, C_m, storage_days, travel_time, total_time = generate_data_with_storage()

    old_fields = [f for f in F if storage_days[f] > 15]
    print(f"Fields with storage >15 days: {len(old_fields)} / {len(F)}")
    if old_fields:
        print(f"  Example old field: {old_fields[0]} stored {storage_days[old_fields[0]]:.1f} days")

    print("Computing degradation penalties using PINN...")
    start = time.time()
    penalties = compute_penalties(V_f, total_time)
    print(f"  Done in {time.time()-start:.2f} sec for {len(penalties)} pairs")

    # -------------------------------
    # MILP Problem
    # -------------------------------
    prob = pulp.LpProblem("BeetRouting_Weighted", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (M, F, Z), cat=pulp.LpBinary)
    
    # Transport cost (realistic: mass * distance * rate)
    transport = pulp.lpSum(x[m][f][z] * V_f[f] * D_fz[f][z] * TRANSPORT_RATE_PER_TON_KM
                           for m in M for f in F for z in Z)
    
    # Degradation penalty
    degradation = pulp.lpSum(x[m][f][z] * penalties[(f, z)] for m in M for f in F for z in Z)
    
    # Shortage variables and penalty
    target_per_factory = 50000.0   # desired tons per day per factory
    shortage = pulp.LpVariable.dicts("shortage", Z, lowBound=0, cat=pulp.LpContinuous)
    shortage_penalty = pulp.lpSum(shortage[z] * SHORTAGE_PENALTY_PER_TON for z in Z)
    
    # Weighted objective
    prob += (w_transport * transport +
             w_degradation * degradation +
             w_shortage * shortage_penalty), "TotalWeightedCost"
    
    # Constraints
    for f in F:
        prob += pulp.lpSum(x[m][f][z] for m in M for z in Z) == 1, f"Unique_{f}"
    for m in M:
        prob += pulp.lpSum(x[m][f][z] * V_f[f] for f in F for z in Z) <= C_m[m], f"Cap_{m}"
    for z in Z:
        delivered = pulp.lpSum(x[m][f][z] * V_f[f] for m in M for f in F)
        prob += delivered + shortage[z] >= target_per_factory, f"Target_{z}"
    
    print(f"MILP built: {len(prob.variables())} vars, {len(prob.constraints)} constraints")
    
    try:
        solver = pulp.getSolver('HiGHS', timeLimit=60, threads=1, msg=False)
    except:
        solver = pulp.getSolver('HiGHS_CMD', path='highs.exe', timeLimit=60, threads=1, msg=False)
    
    solve_start = time.time()
    prob.solve(solver)
    solve_time = time.time() - solve_start
    
    # ----- Collect assignments -----
    assignments = []
    for m in M:
        for f in F:
            for z in Z:
                if pulp.value(x[m][f][z]) == 1.0:
                    t_total = total_time[f][z]
                    loss_pct = sugar_loss_percent(t_total)
                    loss_tons = sugar_loss_tons(V_f[f], t_total)
                    penalty = loss_tons * SUGAR_PRICE_PER_TON
                    transport_cost = V_f[f] * (D_mf[m][f] + D_fz[f][z]) * TRANSPORT_RATE_PER_TON_KM
                    total = (w_transport * transport_cost +
                             w_degradation * penalty)
                    assignments.append({
                        'maus': m, 'field': f, 'factory': z,
                        'biomass': V_f[f], 'total_days': t_total,
                        'loss_pct': loss_pct, 'loss_tons': loss_tons,
                        'penalty': penalty, 'transport_cost': transport_cost,
                        'weighted_total': total
                    })
    
    # ----- Output -----
    print("\n" + "="*70)
    print(" DETAILED LOAD DISTRIBUTION (with weighting)")
    print("="*70)
    
    # By MAUS (sorted by weighted total cost)
    print("\n--- BY MAUS (sorted by weighted total, first 3 per MAUS) ---")
    for m in M:
        m_assign = [a for a in assignments if a['maus'] == m]
        m_assign_sorted = sorted(m_assign, key=lambda x: x['weighted_total'], reverse=True)
        total_mass = sum(a['biomass'] for a in m_assign)
        total_transport = sum(a['transport_cost'] for a in m_assign)
        total_penalty = sum(a['penalty'] for a in m_assign)
        total_weighted = w_transport * total_transport + w_degradation * total_penalty
        print(f"\n{m} → {len(m_assign)} fields, mass: {total_mass:.0f} t, "
              f"transp: {total_transport:.0f}$, penalty: {total_penalty:.0f}$, weighted: {total_weighted:.0f}")
        for i, a in enumerate(m_assign_sorted[:3]):
            print(f"    {i+1}. {a['field']} → {a['factory']} | {a['biomass']:.0f}t | "
                  f"{a['total_days']:.1f}d | loss {a['loss_pct']:.2f}% ({a['loss_tons']:.1f}t) | "
                  f"transp: {a['transport_cost']:.0f}$ | penalty: {a['penalty']:.0f}$ | weighted: {a['weighted_total']:.0f}")
        if len(m_assign) > 3:
            print(f"    ... and {len(m_assign)-3} more")
    
    # By Factory (with shortage info)
    print("\n--- BY FACTORY (sorted by weighted total, first 5 deliveries) ---")
    for z in Z:
        z_assign = [a for a in assignments if a['factory'] == z]
        z_assign_sorted = sorted(z_assign, key=lambda x: x['weighted_total'], reverse=True)
        total_mass = sum(a['biomass'] for a in z_assign)
        total_transport = sum(a['transport_cost'] for a in z_assign)
        total_penalty = sum(a['penalty'] for a in z_assign)
        total_weighted = w_transport * total_transport + w_degradation * total_penalty
        shortage_val = pulp.value(shortage[z])
        shortage_penalty_val = shortage_val * SHORTAGE_PENALTY_PER_TON
        print(f"\nFactory {z} ← {len(z_assign)} fields, mass: {total_mass:.0f} t, "
              f"transp: {total_transport:.0f}$, penalty: {total_penalty:.0f}$, weighted: {total_weighted:.0f}, "
              f"shortage: {shortage_val:.0f} t (penalty: {shortage_penalty_val:.0f}$)")
        for i, a in enumerate(z_assign_sorted[:5]):
            print(f"    {i+1}. {a['maus']} from {a['field']} | {a['biomass']:.0f}t | "
                  f"{a['total_days']:.1f}d | loss {a['loss_pct']:.2f}% ({a['loss_tons']:.1f}t) | "
                  f"transp: {a['transport_cost']:.0f}$ | penalty: {a['penalty']:.0f}$ | weighted: {a['weighted_total']:.0f}")
        if len(z_assign) > 5:
            print(f"    ... and {len(z_assign)-5} more")
    
    print("="*70)
    print(f"Weighting used: w_transport={w_transport}, w_degradation={w_degradation}, w_shortage={w_shortage}")
    print("="*70)

if __name__ == "__main__":
    main()