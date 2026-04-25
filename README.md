# Robust Risk-Aware Offline Neural Contextual Bandits

This repository contains the JAX implementation for **Robust Risk-Aware Offline Contextual Bandits**, extending the NeuraLCB framework to handle **outliers (heavy-tails)** and **risk-sensitivity (CVaR)**.

## Key Features
* **Risk-Aware Learning**: Optimize Conditional Value at Risk (CVaR), Entropic Risk, and Mean-Variance measures.
* **Robustness (Tofu Loss)**: Reward truncation to mitigate the impact of black-swan outliers and heavy-tailed noise.
* **Medical Simulation Support**: Integration with `simglucose` for Type-1 Diabetes bolus estimation research.
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
# 1. Generate Simglucose data (UVA/Padova model)
PYTHONPATH=. python data/simglucose_offline_gen.py

# 2. (Optional) Generate Robust Synthetic data
# Automatically generated when running synthetic_main.py or realworld_main.py with --data_type robust_syn
```

## Running the Experiments

### 1. Robust Risk-Aware OPL (Medical Data)
Run the `RobustOfflineBatchNeuraLCB` algorithm on the generated Simglucose dataset:
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=. python realworld_main.py \
    --data_type simglucose \
    --algo_group robust-offline \
    --risk_measure cvar \
    --alpha 0.05 \
    --tau_n 1.0 \
    --beta 0.1 \
    --num_sim 1 \
    --num_steps 1000 \
    --layer_sizes 16,16 \
    --nouse_wandb
```

### 2. Robust Synthetic Experiments
```bash
PYTHONPATH=. python realworld_main.py \
    --data_type robust_syn \
    --function_type cosine \
    --noise_type student-t \
    --algo_group robust-offline \
    --num_steps 500
```

## Available Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `--data_type` | Dataset (`simglucose`, `robust_syn`, `mushroom`, `mnist`, etc.) | `mushroom` |
| `--algo_group` | Algorithm group (`robust-offline`, `approx-neural`, `baseline`) | `approx-neural` |
| `--risk_measure` | Risk functional (`cvar`, `mean`, `entropic`, `mean_variance`) | `cvar` |
| `--alpha` | CVaR level (tail probability) | `0.05` |
| `--tau_n` | Truncation threshold for Tofu Loss | `1.0` |
| `--beta` | Confidence parameter (pessimism level) | `0.1` |
| `--layer_sizes` | Neural network structure (e.g., `16,16` or `64,64`) | `100,100` |
| `--num_steps` | Number of gradient updates for training | `100` |
| `--num_sim` | Number of independent simulations to run | `10` |

## Repository Structure
* `/algorithms`: Implementation of `RobustOfflineBatchNeuraLCB` and baseline bandit models.
* `/core`: Core JAX neural network utilities and bandit runners.
* `/data`: Data loaders and synthetic generators (`simglucose`, `robust_synthetic_data`).
* `realworld_main.py`: Entry point for experiments.
