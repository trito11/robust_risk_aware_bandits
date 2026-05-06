import subprocess
import os
import shutil

# 1. Configuration for Simglucose (Referenced from run_robust_sweep.py)
DATA_TYPE = "simglucose"
CONTEXT_DIM = 4
NUM_ACTIONS = 11
NUM_SIM = 10        # Number of simulations per N
LAYER_SIZES = "32,32"  
DATA_PATH = "data/simglucose_offline.npz"
NUM_TEST = 500    

# Algorithm parameters from run_robust_sweep.py
RISK_MEASURE = "cvar"
ALPHA = 0.05
TAU_N = 50.0       # Reduced to enable clipping for better robustness in Simglucose
LAMBDA0 = 10.0
POLICY_TYPE = "risk-aware"
AGENT_EVAL_METHOD = "local"
ORACLE_EVAL_METHOD = "global"

# List of all algorithm groups to compare
ALGO_GROUPS = [
    "robust-offline",    # RobustOfflineBatchNeuraLCB
    "risk-exact",        # RiskExactNeuraLCBV2
    "quantile-risk",     # QuantileRiskNeuralBandit
    "risk-lin-lcb",      # RiskLinLCB
    "neural-regression"  # NeuralRegressionOffline
]

# Sweep over num_contexts (N)
N_VALUES = [0, 100, 200, 400]

# 2. Experimental Loop
for algo in ALGO_GROUPS:
    # Fine-tune parameters based on algorithm
    if algo == "quantile-risk":
        BETA = 0.01      # Increased slightly for stability
        NUM_STEPS = 3000
        LR = 5e-4
        LAMBDA = 1e-3
        TRUNC_MODE = "none"
    elif algo == "robust-offline":
        BETA = 0.05      # Reduced to avoid over-pessimism and focus on Action 3 accuracy
        NUM_STEPS = 5000 # Increased training steps for better Tofu loss convergence
        LR = 1e-3
        LAMBDA = 1e-2    # Increased regularization to stabilize learning
        TRUNC_MODE = "clip" # Re-enabled for heavy-tail robustness
    else:
        BETA = 0.5       # Standard Neural LCB also uses 0.5
        NUM_STEPS = 2000
        LR = 1e-3
        LAMBDA = 1e-4
        TRUNC_MODE = "none"

    for n in N_VALUES:
        print(f"\n" + "="*60)
        print(f"RUNNING SIMGUCOSE: Algo={algo} | N={n} | Trunc={TRUNC_MODE}")
        print("="*60)
        
        cmd = [
            "python", "realworld_main.py",
            "--data_type", DATA_TYPE,
            "--context_dim", str(CONTEXT_DIM),
            "--num_actions", str(NUM_ACTIONS),
            "--num_contexts", str(n),
            "--algo_group", algo,
            "--num_sim", str(NUM_SIM),
            "--layer_sizes", LAYER_SIZES,
            "--beta", str(BETA),
            "--risk_measure", RISK_MEASURE,
            "--alpha", str(ALPHA),
            "--tau_n", str(TAU_N),
            "--num_steps", str(NUM_STEPS),
            "--lambd0", str(LAMBDA0),
            "--lambd", str(LAMBDA),
            "--lr", str(LR),
            "--policy_type", POLICY_TYPE,
            "--truncation_mode", TRUNC_MODE,
            "--agent_eval_method", AGENT_EVAL_METHOD,
            "--oracle_eval_method", ORACLE_EVAL_METHOD,
            "--nouse_wandb",
            "--num_steps", str(NUM_STEPS),
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Error running simulation for {algo} at N={n}: {e}")
            continue

print("\nToàn bộ thực nghiệm Simglucose đã hoàn tất!")
