# Robust Risk-Aware Offline Neural Contextual Bandits

This repository contains the JAX implementation for **Robust Risk-Aware Offline Contextual Bandits**, extending the NeuraLCB framework to handle **outliers (heavy-tails)** and **risk-sensitivity (CVaR)**.

## Key Features
* **Risk-Aware Learning**: Optimize Conditional Value at Risk (CVaR), Entropic Risk, and Mean-Variance measures.
* **Robustness (Tofu Loss)**: Reward truncation to mitigate the impact of black-swan outliers and heavy-tailed noise.
* **Medical Simulation Support**: Integration with `simglucose` for Type-1 Diabetes bolus estimation research featuring multi-patient modeling and a 4-hour post-prandial risk assessment.
* **Efficient Covariance Matrix**: Memory-optimized chunked calculation for the exact neural tangent kernel (NTK) covariance.

## Dependencies 
Recommended environment: Python 3.10
Install dependencies via `pip`:
```bash
pip install -r requirements.txt
```
*Note: `setuptools<70` is required for `simglucose` compatibility.*

## Data Generation
Before running medical simulations, generate the offline dataset:
```bash
# Generate Simglucose data (UVA/Padova model)
# This simulates 4 patients with 5% sensor failure-induced outliers
PYTHONPATH=. python data/simglucose_offline_gen.py
```

## Running the Experiments

### 1. Manual Execution
Run the `RobustOfflineBatchNeuraLCB` algorithm:
```bash
# Medical Data (Simglucose)
# Requires --context_dim 4 and --num_actions 11
PYTHONPATH=. python realworld_main.py \
    --data_type simglucose --context_dim 4 --num_actions 11 \
    --algo_group robust-offline --risk_measure cvar --beta 0.1

# Synthetic Data
PYTHONPATH=. python realworld_main.py \
    --data_type robust_syn --function_type cosine --noise_type student-t \
    --algo_group robust-offline --num_sim 100
```

### 2. Automated Sweeps (Benchmarking)
To generate results for multiple sample sizes ($N$) and compute confidence intervals (as shown in standard papers):

```bash
# Run sweep for Synthetic data (Varying N from 1k to 20k)
python run_robust_sweep.py

# Run sweep for Medical data (Varying N for Simglucose)
python run_simglucose_sweep.py
```

## Evaluation Metrics
The results are saved in `results/` as `.npz` files containing:
* **Regret**: Difference between optimal reward (Oracle) and agent's reward.
* **GT CVaR**: The actual ground-truth risk encountered in the simulator (bottom 5% of rewards).
* **Oracle CVaR**: The theoretical best risk achievable by the optimal policy.
* **Err**: Error rate (1 - Accuracy) in selecting the optimal arm.

## Available Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `--data_type` | Dataset (`simglucose`, `robust_syn`, `mushroom`, etc.) | `mushroom` |
| `--algo_group` | Algorithm group (`robust-offline`, `approx-neural`, `baseline`) | `approx-neural` |
| `--risk_measure` | Risk functional (`cvar`, `mean`, `entropic`, `mean_variance`) | `cvar` |
| `--alpha` | CVaR level (tail probability) | `0.05` |
| `--tau_n` | Truncation threshold for Tofu Loss | `1.0` |
| `--beta` | Confidence parameter (pessimism level) | `0.1` |
| `--layer_sizes` | Neural network structure (use comma for multiple layers, e.g., `32,32`) | `100,100` |
| `--num_sim` | Number of independent simulations per N (for error bars) | `10` |

## Repository Structure
* `/algorithms`: Implementation of `RobustOfflineBatchNeuraLCB` and baseline models.
* `/core`: Core JAX neural network utilities and bandit runners.
* `/data`: Data loaders and generators (`simglucose`, `robust_synthetic_data`).
* `run_..._sweep.py`: Automation scripts for large-scale experiments.
