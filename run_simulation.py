"""
=========================================================================================
Title: Extreme Stress-Test MILP Benchmark for Regional Sugar Beet Shuttle Routing
Related Paper: "Building Cyber-Physical Resilient Regional Supply Chains: 
                Physics-Informed Neural Networks and Model Predictive Control in Stochastic Environments"
Description: 
    This script provides an isolated, algorithmic Proof-of-Concept (PoC) for the 
    Mixed-Integer Linear Programming (MILP) core under EXTREME peak workloads 
    (10 MAUS, 350 Fields, 2 Factories).
    
    It demonstrates the raw mathematical convergence speed of the spatial routing matrix
    (yielding 7,000 binary variables and 360 structural constraints).
    
Execution: 
    Strictly single-threaded (threads=1) using the open-source HiGHS solver to 
    ensure hardware-agnostic reproducibility and fair academic benchmarking.
=========================================================================================
"""

import pulp
import time
import numpy as np
import warnings

# Suppress minor warnings for a clean benchmark output
warnings.filterwarnings("ignore")

def generate_synthetic_benchmark_data():
    """
    Generates a synthetic dataset that statistically mirrors the topological 
    variance of the regional supply chain during an EXTREME peak harvesting day.
    
    Returns:
        M (list): Set of mobile loaders (MAUS).
        F (list): Set of harvest-ready fields.
        Z (list): Set of processing factories.
        D_mf (dict): Distance/Cost matrix from MAUS to Fields.
        D_fz (dict): Distance/Cost matrix from Fields to Factories.
        V_f (dict): Available biomass volume at each field (tons).
        C_m (dict): Daily operational capacity of each MAUS (tons).
    """
    np.random.seed(42) # Fixed seed for strict scientific reproducibility
    
    # 1. Sets (Indices)
    M = [f"m_{i}" for i in range(1, 11)]       # Set of MAUS loaders (|M| = 10)
    F = [f"f_{i}" for i in range(1, 351)]      # Set of Fields (|F| = 350) - EXTREME SCENARIO
    Z = [f"z_{i}" for i in range(1, 3)]        # Set of Factories (|Z| = 2)
    
    # 2. Parameters
    # D_{m,f}: Relocation cost/distance from MAUS m to Field f
    D_mf = {m: {f: np.random.uniform(5.0, 50.0) for f in F} for m in M}
    
    # D_{f,z}: Transit cost/distance from Field f to Factory z
    D_fz = {f: {z: np.random.uniform(10.0, 100.0) for z in Z} for f in F}
    
    # V_{f}: Biological raw material volume at generation node f
    V_f = {f: np.random.uniform(200, 800) for f in F}
    
    # C_{m}: Daily throughput capacity constraint for MAUS m
    # Increased to 25,000 tons to mathematically absorb the massive biomass from 350 fields
    C_m = {m: 25000 for m in M} 
    
    return M, F, Z, D_mf, D_fz, V_f, C_m


def execute_milp_core():
    """
    Constructs and solves the spatial routing optimization matrix.
    Formulation maps directly to the mathematical equations presented in the paper.
    """
    print("=====================================================================")
    print(" EXTREME STOCHASTIC ROUTING BENCHMARK: INITIALIZING MATHEMATICAL CORE")
    print("=====================================================================\n")
    
    # Load sets and parameters
    M, F, Z, D_mf, D_fz, V_f, C_m = generate_synthetic_benchmark_data()
    
    # Initialize the MILP Problem (Minimization)
    prob = pulp.LpProblem("SugarBeet_Routing_MILP_Extreme", pulp.LpMinimize)
    
    print("[*] Generating EXTREME active search space and variables...")
    build_start = time.time()
    
    # -------------------------------------------------------------------------
    # DECISION VARIABLES
    # x_{m,f,z} ∈ {0, 1}: 1 if MAUS m serves field f and dispatches to factory z
    # -------------------------------------------------------------------------
    x = pulp.LpVariable.dicts("x", (M, F, Z), cat=pulp.LpBinary)
    
    # -------------------------------------------------------------------------
    # OBJECTIVE FUNCTION
    # Min Z = Σ_m Σ_f Σ_z x_{m,f,z} * (D_{m,f} + D_{f,z})
    # Minimizing macro-systemic transportation and relocation expenditures.
    # -------------------------------------------------------------------------
    prob += pulp.lpSum(
        x[m][f][z] * (D_mf[m][f] + D_fz[f][z])
        for m in M for f in F for z in Z
    ), "Total_Logistics_Cost"
    
    # -------------------------------------------------------------------------
    # STRUCTURAL CONSTRAINTS
    # -------------------------------------------------------------------------
    
    # Constraint 1: Exact Field Servicing (Demand Satisfaction)
    # Σ_m Σ_z x_{m,f,z} = 1, ∀ f ∈ F
    # Ensures every harvest-ready field is processed exactly once per planning step.
    for f in F:
        prob += pulp.lpSum(x[m][f][z] for m in M for z in Z) == 1, f"Service_Req_{f}"
        
    # Constraint 2: MAUS Throughput Capacity Limit
    # Σ_f Σ_z (V_f * x_{m,f,z}) ≤ C_m, ∀ m ∈ M
    # Ensures the allocated biomass volume does not exceed the daily loader capacity.
    for m in M:
        prob += pulp.lpSum(
            x[m][f][z] * V_f[f] 
            for f in F for z in Z
        ) <= C_m[m], f"Capacity_Limit_{m}"
        
    build_end = time.time()
    
    print(f"    Matrix built in:       {build_end - build_start:.4f} seconds")
    print(f"    Binary Variables:      {len(prob.variables())} (x_m,f,z)")
    print(f"    Structural Constraints: {len(prob.constraints)}\n")
    
    # -------------------------------------------------------------------------
    # SOLVER EXECUTION
    # Using HiGHS solver. 'threads=1' enforces single-threaded execution 
    # to demonstrate algorithmic compactness without hardware acceleration.
    # -------------------------------------------------------------------------
    print("[*] Launching HiGHS Exact Solver (Single-Threaded Mode)...")
    
    try:
        # Attempting to use the Python binding first (Method A)
        solver = pulp.getSolver('HiGHS', timeLimit=60, threads=1, msg=False)
    except pulp.PulpSolverError:
        # Falling back to the standalone executable in the current folder (Method B)
        solver = pulp.getSolver('HiGHS_CMD', path='highs.exe', timeLimit=60, threads=1, msg=False)
    
    solve_start = time.time()
    prob.solve(solver)
    solve_end = time.time()
    
    # -------------------------------------------------------------------------
    # BENCHMARK RESULTS & METRICS
    # -------------------------------------------------------------------------
    print("=====================================================================")
    print(" BENCHMARK RESULTS")
    print("=====================================================================")
    print(f"    Optimization Status:  {pulp.LpStatus[prob.status]}")
    print(f"    Convergence Time:     {solve_end - solve_start:.4f} seconds")
    print(f"    Objective Value (Z):  {pulp.value(prob.objective):.2f} units\n")
    
    print("[*] Optimal Macro-Routing Sample (Top 5 Dispatch Directives):")
    assignments_shown = 0
    for m in M:
        for f in F:
            for z in Z:
                # If the decision variable is active (== 1)
                if pulp.value(x[m][f][z]) == 1.0:
                    assignments_shown += 1
                    if assignments_shown <= 5:
                        route_cost = D_mf[m][f] + D_fz[f][z]
                        print(f"    [+] {m} -> {f} -> {z} | Biomass: {V_f[f]:.1f}t | Cost: {route_cost:.2f}")

# Execute the main function
if __name__ == "__main__":
    execute_milp_core()