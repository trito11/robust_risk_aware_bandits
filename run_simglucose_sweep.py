import subprocess
import os

# Configuration for Simglucose
DATA_TYPE = "simglucose"
CONTEXT_DIM = 4
NUM_ACTIONS = 11
ALGO_GROUP = "robust-offline"
NUM_SIM = 100        # Number of simulations per N for confidence intervals
LAYER_SIZES = "32,32"  

# Sweep over num_contexts (N)
# Note: Simglucose generator by default produces 8000 training samples
N_VALUES = [200, 400, 1000, 4000]

for n in N_VALUES:
    print(f"\n" + "="*60)
    print(f"RUNNING SIMGUCOSE SWEEP: N = {n} (Simulating {NUM_SIM} times)")
    print("="*60)
    
    cmd = [
        "python", "realworld_main.py",
        "--data_type", DATA_TYPE,
        "--context_dim", str(CONTEXT_DIM),
        "--num_actions", str(NUM_ACTIONS),
        "--num_contexts", str(n),
        "--algo_group", ALGO_GROUP,
        "--num_sim", str(NUM_SIM),
        "--beta", "0.1",
        "--layer_sizes", LAYER_SIZES,
        "--nouse_wandb"
    ]
    
    # Execute the experiment
    process = subprocess.run(cmd)
    
    if process.returncode != 0:
        print(f"ERROR: Experiment failed for N={n}. Skipping...")

print("\nSimglucose sweep completed! Check the results/ directory.")
