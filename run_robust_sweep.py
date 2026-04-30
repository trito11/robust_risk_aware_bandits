import subprocess
import os

# 1. Configuration from user request
DATA_TYPE = "robust_syn"
FUNCTION_TYPE = "cosine"
NOISE_TYPE = "student-t"
CONTEXT_DIM = 20
NUM_ACTIONS = 30
LAYER_SIZES = "10"
BETA = 0.1
RISK_MEASURE = "cvar"
ALPHA = 0.05
TAU_N = 1.0
NUM_SIM = 10      # 500 runs to get stable statistics for the plot
NUM_STEPS = 1000
NUM_TEST = 50     # Number of test contexts (keep small for MILP)

# Options added: ALGO_GROUP and POLICY_EVAL_METHOD
ALGO_GROUP = "risk-exact" # "robust-offline", "approx-neural", "baseline", "risk-exact"
POLICY_EVAL_METHOD = "local"  # "local" (point-wise argmax) or "global" (MILP)

# 2. Sweep over num_contexts (N)
N_VALUES = [100,200]

for n in N_VALUES:
    print(f"\n" + "="*60)
    print(f"RUNNING SWEEP: N = {n} (Simulating {NUM_SIM} times)")
    print("="*60)
    
    cmd = [
        "python", "realworld_main.py",
        "--data_type", DATA_TYPE,
        "--function_type", FUNCTION_TYPE,
        "--noise_type", NOISE_TYPE,
        "--context_dim", str(CONTEXT_DIM),
        "--num_actions", str(NUM_ACTIONS),
        "--num_contexts", str(n),
        "--num_test_contexts", str(NUM_TEST),
        "--layer_sizes", LAYER_SIZES,
        "--beta", str(BETA),
        "--algo_group", ALGO_GROUP,
        "--policy_eval_method", POLICY_EVAL_METHOD,
        "--risk_measure", RISK_MEASURE,
        "--alpha", str(ALPHA),
        "--tau_n", str(TAU_N),
        "--num_sim", str(NUM_SIM),
        "--num_steps", str(NUM_STEPS),
        "--nouse_wandb"
    ]
    
    # Execute the experiment
    process = subprocess.run(cmd)
    
    if process.returncode != 0:
        print(f"ERROR: Experiment failed for N={n}. Stopping sweep.")
        break

print("\nSweep completed! You can now use the notebook to plot the results.")
