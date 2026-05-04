import subprocess
import os
import sys

# 1. Danh sách cấu hình thực nghiệm theo yêu cầu mới
ALGO_GROUPS = ["risk-exact"]
FUNCTION_TYPES = ["cosine"]
N_VALUES = [100, 500, 1000, 5000, 10000, 20000, 50000]

# 2. Cấu hình cố định
DATA_TYPE = "robust_syn"
NOISE_TYPE = "student-t"
CONTEXT_DIM = 20
NUM_ACTIONS = 30
LAYER_SIZES = "32,32"
RISK_MEASURE = "cvar"
ALPHA = 0.05
TAU_N = 1.0
NUM_SIM = 10      
NUM_TEST = 500    
AGENT_EVAL_METHOD = "local"   
ORACLE_EVAL_METHOD = "global" 
LAMBDA0 = 10.0                
POLICY_TYPE = "risk-aware"    

# 3. Vòng lặp thực nghiệm
for algo in ALGO_GROUPS:
    for func in FUNCTION_TYPES:
        # Tinh chỉnh tham số cho Risk-Exact
        BETA = 0.1
        NUM_STEPS = 1000
        LR = 1e-3
        LAMBDA = 1e-4

        for n in N_VALUES:
            print(f"\n" + "="*60)
            print(f"RUNNING: Algo={algo} | Func={func} | N={n}")
            print("="*60)
            
            cmd = [
                sys.executable, "realworld_main.py",
                "--data_type", DATA_TYPE,
                "--function_type", func,
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
                "--algo_group", algo,
                "--agent_eval_method", AGENT_EVAL_METHOD,
                "--oracle_eval_method", ORACLE_EVAL_METHOD,
                "--lambd0", str(LAMBDA0),
                "--lambd", str(LAMBDA),
                "--lr", str(LR),
                "--policy_type", POLICY_TYPE,
                "--nouse_wandb"
            ]
            
            subprocess.run(cmd)

print(f"\nThực nghiệm Risk-Exact trên hàm Cosine đã hoàn tất!")
