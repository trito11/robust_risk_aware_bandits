import subprocess
import os

# 1. Configuration from user request
DATA_TYPE = "robust_syn"
FUNCTION_TYPE = "cosine"
NOISE_TYPE = "student-t"
CONTEXT_DIM = 20
NUM_ACTIONS = 30
LAYER_SIZES = "32,32"
BETA = 0.1
RISK_MEASURE = "cvar"
ALPHA = 0.05
TAU_N = 1.0
NUM_SIM = 10      # 500 runs to get stable statistics for the plot
NUM_STEPS = 1000
NUM_TEST = 500    # Number of test contexts (keep small for MILP)

# Options added: ALGO_GROUP, AGENT_EVAL_METHOD, ORACLE_EVAL_METHOD
ALGO_GROUP = "risk-lin-lcb" # "robust-offline", "neural-regression", "risk-exact", "quantile-risk", "risk-lin-lcb"
AGENT_EVAL_METHOD = "local"   # "local" (point-wise argmax) or "global" (MILP)
ORACLE_EVAL_METHOD = "global" # "local" or "global" (Marginal CVaR via MILP)

# 2. Sweep over num_contexts (N)
N_VALUES = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]

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
        "--num_sim", str(NUM_SIM),
        "--beta", str(BETA),
        "--risk_measure", RISK_MEASURE,
        "--alpha", str(ALPHA),
        "--tau_n", str(TAU_N),
        "--layer_sizes", LAYER_SIZES,
        "--num_steps", str(NUM_STEPS),
        "--algo_group", ALGO_GROUP,
        "--agent_eval_method", AGENT_EVAL_METHOD,
        "--oracle_eval_method", ORACLE_EVAL_METHOD,
        "--nouse_wandb"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

print("\nSweep completed! You can now use the notebook to plot the results.")
