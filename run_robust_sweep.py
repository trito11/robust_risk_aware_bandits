import subprocess
import os

# 1. Configuration from user request
DATA_TYPE = "robust_syn"
FUNCTION_TYPE = "cosine"
NOISE_TYPE = "student-t"
CONTEXT_DIM = 20
NUM_ACTIONS = 30
LAYER_SIZES = "32,32"

# Chọn nhóm thuật toán tại đây
ALGO_GROUP = "quantile-risk" # "robust-offline", "neural-regression", "risk-exact", "quantile-risk", "risk-lin-lcb"

# Tinh chỉnh tham số dựa trên thuật toán
if ALGO_GROUP == "quantile-risk":
    BETA = 0.005        # Giảm mạnh Beta vì có hệ số L_rho = 20 (1/0.05)
    NUM_STEPS = 3000    # Tăng bước huấn luyện cho học phân phối
    LR = 5e-4           # Learning rate thấp hơn cho ổn định
    LAMBDA = 1e-3       # Tăng regularization để tránh bùng nổ đặc trưng phi
else:
    BETA = 0.1
    NUM_STEPS = 1000
    LR = 1e-3
    LAMBDA = 1e-4

RISK_MEASURE = "cvar"
ALPHA = 0.05
TAU_N = 1.0
NUM_SIM = 10      
NUM_TEST = 500    

# Options added: ALGO_GROUP, AGENT_EVAL_METHOD, ORACLE_EVAL_METHOD
AGENT_EVAL_METHOD = "local"   
ORACLE_EVAL_METHOD = "global" 
LAMBDA0 = 10.0                
POLICY_TYPE = "risk-aware"    

# 2. Sweep over num_contexts (N)
N_VALUES = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]

for n in N_VALUES:
    print(f"\n" + "="*60)
    print(f"RUNNING SWEEP: {ALGO_GROUP} | N = {n} (Simulating {NUM_SIM} times)")
    print("="*60)
    
    cmd = [
        "python", "realworld_main.py",
        "--data_type", DATA_TYPE,
        "--function_type", FUNCTION_TYPE,
        "--noise_type", NOISE_TYPE,
        "--context_dim", str(CONTEXT_DIM),
        "--num_actions", str(NUM_ACTIONS),
        "--num_contexts", str(n),
        "--beta", str(BETA),
        "--risk_measure", RISK_MEASURE,
        "--alpha", str(ALPHA),
        "--tau_n", str(TAU_N),
        "--num_sim", str(NUM_SIM),
        "--num_steps", str(NUM_STEPS),
        "--num_test_contexts", str(NUM_TEST),
        "--algo_group", ALGO_GROUP,
        "--agent_eval_method", AGENT_EVAL_METHOD,
        "--oracle_eval_method", ORACLE_EVAL_METHOD,
        "--lambd0", str(LAMBDA0),
        "--lambd", str(LAMBDA),
        "--lr", str(LR),
        "--policy_type", POLICY_TYPE,
        "--nouse_wandb"
    ]
    
    subprocess.run(cmd)

print(f"\nSweep completed for {ALGO_GROUP}! Bạn có thể dùng notebook để vẽ lại đồ thị.")
