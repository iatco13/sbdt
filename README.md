# 🚜 Sugar Beet Digital Twin: Cyber-Physical Supply Chain PoC

**Related Paper:** *Building Cyber-Physical Resilient Regional Supply Chains: Physics-Informed Neural Networks and Model Predictive Control in Stochastic Environments*

## 📌 Overview
This repository contains a **Standalone Algorithmic Proof-of-Concept (PoC)** to accompany the research paper. To ensure open scientific reproducibility without violating Corporate Non-Disclosure Agreements (NDA), we have extracted the core mathematical engines of the Digital Twin and isolated them from the enterprise Event-Driven Architecture (EDA).

This sandbox environment focuses strictly on the mathematical viability, matrix compactness, and computational convergence speed of the proposed algorithms.

## 🗂️ Repository Structure

The repository is divided into two synergistic computational modules:

### 1. Macro-Routing & Scheduling Core (`run_simulation.ipynb` / `run_simulation.py`)
This script contains the core Mixed-Integer Linear Programming (MILP) heuristic (Forward-Backward Scheduling) described in **Section 3.2** of the paper.
* **Objective:** Minimization of macro-systemic transportation and relocation expenditures.
* **Scale:** Simulates an extreme peak-day scenario with 10 MAUS loaders, 350 fields, and 2 factories (yielding exactly 7,000 binary variables and 360 structural constraints).
* **Execution:** Strictly single-threaded (`threads=1`) using the open-source HiGHS solver to ensure hardware-agnostic reproducibility and fair academic benchmarking. Expected convergence time is under 0.5 seconds on a standard laptop.

### 2. Physics-Informed Neural Network (PINN) Sandbox (`pinn_training.ipynb` / `pinn_training.py`)
This script provides a standalone PyTorch implementation of the PINN module (as defined in **Eq. 28** and **Eq. 29** of the manuscript).
* **Objective:** Predicts non-linear biochemical sucrose degradation in field piles when empirical laboratory data is sporadic or heavily delayed (Laboratory Blind Spots).
* **Mechanism:** The neural network embeds a complex Ordinary Differential Equation (ODE) via `torch.autograd`. It models the transition from smooth early-stage cellular respiration (Arrhenius kinetics) to exponential late-stage microbiological decay ($\alpha \cdot t^2$).
* **Integration:** Calculates the daily degradation penalty coefficient ($\Delta S_j^t$), which is dynamically forwarded to the MILP core for Smart Triage and priority evacuation.

---

## ⚠️ Architecture Disclaimer (What this PoC is NOT)
As detailed in the manuscript, the actual production-grade Digital Twin operates within a complex multi-tier architecture. To make this code easily executable for peer reviewers, the following enterprise-grade middleware has been deliberately stripped out:
* **Event-Driven Automation:** Message brokers and asynchronous triggers have been replaced by sequential script execution.
* **Database Integration:** Direct queries to corporate ERPs and Azure Data Lakes have been replaced with static, synthetically generated datasets.
* **Hardware Acceleration:** GPU-clusters are disabled to demonstrate raw algorithmic efficiency on standard CPUs.

*Note: In a full simulated production environment, the entire pipeline (including PINN fine-tuning, neural inference, and MILP optimization) combined with Azure Data Lake synchronization executes reliably within a 15-minute window.*

---

## 🛠 Prerequisites and Required Libraries

To run the scripts or Jupyter Notebooks, you need Python 3.8+ installed on your system. Install all required mathematical, optimization, and deep learning libraries using `pip`:

```bash
pip install pulp pandas numpy jupyter highspy torch matplotlib
```

*Note: The `highspy` package provides pre-compiled Python bindings for the HiGHS solver, which allows `PuLP` to run HiGHS directly out-of-the-box on most systems.*

---

## ▶️ How to Run the Simulations

1. Clone or download this repository to your local machine.
2. Open your terminal or command prompt in the repository directory.
3. Launch Jupyter Notebook (or run the `.py` scripts directly via terminal):
   ```bash
   jupyter notebook
   ```

### Running the PINN Model:
* Open `pinn_training.ipynb` and execute the cells.
* Observe the `Total Loss` dropping as the Physics Loss ($\mathcal{L}_{PDE}$) guides the network.
* At the end of the execution, the script will output the calculated Daily Sucrose Loss ($\Delta S_j^t$), triggering the Smart Triage logic.

### Running the MILP Core:
* Open `run_simulation.ipynb` and execute the cells.
* Observe the structural generation of 7,000 variables and 360 constraints.
* Check the **Convergence Time** (expected ~0.45 seconds) and the optimal macro-routing dispatch directives.

---

## ✒️ Authorship & Citation

**Author:** Corneliu Iatco, Academy of Economic Studies of Moldova (ASEM).

This codebase is released as supplementary material for the academic paper. If you use these algorithms, logic, or the synthetic datasets in your own research and simulations, please cite the original manuscript:

> **Iatco, C. (2026).** *Building Cyber-Physical Resilient Regional Supply Chains: Physics-Informed Neural Networks and Model Predictive Control in Stochastic Environments.* Smart Cities and Regional Development (SCRD) Journal. DOI: [To be added upon publication]

---

## 📄 License

This Proof-of-Concept is open-sourced under the **MIT License**. 

You are free to use, modify, distribute, and run this academic sandbox for your own research, educational, and testing purposes. The software is provided "as is", without warranty of any kind. For more details, see the standard MIT License terms.
