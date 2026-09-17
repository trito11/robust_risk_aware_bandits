import subprocess
import os

# ──────────────────────────────────────────────────────────────────
# 1. Cấu hình thực nghiệm
# ──────────────────────────────────────────────────────────────────
ALGO_GROUPS = [
    "robust-offline",       # RobustOfflineBatchNeuraLCB
    "risk-exact",           # RiskExactNeuraLCBV2
    "quantile-risk",        # QuantileRiskNeuralBandit
    "risk-lin-lcb",         # RiskLinLCB
    # "neural-regression",  # NeuralRegressionOffline
    "pessimistic-cdf",      # paper arXiv:2605.15620, context-independent
    "pessimistic-cdf-ctx",  # context-aware (chậm hơn, có thể comment nếu chỉ test global)
]

N_VALUES     = [0, 100, 200, 400]
ALPHA_VALUES = [0.01, 0.05, 0.1, 0.2]

# ──────────────────────────────────────────────────────────────────
# 2. Cấu hình cố định cho Simglucose
# ──────────────────────────────────────────────────────────────────
DATA_TYPE          = "simglucose"
CONTEXT_DIM        = 4
NUM_ACTIONS        = 11
NUM_SIM            = 10
LAYER_SIZES        = "32,32"
NUM_TEST           = 500
RISK_MEASURE       = "cvar"
TAU_N              = 50.0       # Truncation threshold for heavy-tail robustness
LAMBDA0            = 10.0
POLICY_TYPE        = "risk-aware"
AGENT_EVAL_METHOD  = "local"
ORACLE_EVAL_METHOD = "global"

# ──────────────────────────────────────────────────────────────────
# 3. Cấu hình riêng cho Pessimistic CDF (arXiv:2605.15620)
# ──────────────────────────────────────────────────────────────────
# Chọn một hoặc nhiều estimator để so sánh:
#   "IS"  – clipped Importance Sampling (Eq. 5, Theorem 2)
#   "WIS" – Weighted IS (Eq. 6, Theorem 3)
#   "DR"  – Doubly Robust (Eq. 8, Theorem 4)
PESS_CDF_ESTIMATORS = ["IS", "WIS", "DR"]   # sweep cả 3 (hoặc chỉnh ["DR"] nếu muốn chạy nhanh)
PESS_DELTA          = 0.05   # confidence level cho R(pi)
PESS_D_PI           = 1      # policy class complexity d_Pi

# ──────────────────────────────────────────────────────────────────
# 4. Vòng lặp thực nghiệm
# ──────────────────────────────────────────────────────────────────
for algo in ALGO_GROUPS:
    # Fine-tune hyperparams theo từng thuật toán
    if algo == "quantile-risk":
        BETA = 0.01      # Increased slightly for stability
        NUM_STEPS = 3000
        LR = 5e-4
        LAMBDA = 1e-3
        TRUNC_MODE = "none"
    elif algo == "robust-offline":
        BETA = 0.05      # Avoid over-pessimism and focus on Action 3 accuracy
        NUM_STEPS = 5000 # More steps for Tofu loss convergence
        LR = 1e-3
        LAMBDA = 1e-2    # Regularization to stabilize learning
        TRUNC_MODE = "clip" # Re-enabled for heavy-tail robustness
    elif algo in ("pessimistic-cdf", "pessimistic-cdf-ctx"):
        BETA = 0.1
        NUM_STEPS = 1000
        LR = 1e-3
        LAMBDA = 1e-4
        TRUNC_MODE = "none"
    else:
        BETA = 0.5       # Standard Neural LCB
        NUM_STEPS = 2000
        LR = 1e-3
        LAMBDA = 1e-4
        TRUNC_MODE = "none"

    # Với pessimistic-cdf: sweep từng estimator (IS, WIS, DR)
    if algo in ("pessimistic-cdf", "pessimistic-cdf-ctx"):
        estimator_list = PESS_CDF_ESTIMATORS
    else:
        estimator_list = [None]   # không dùng flag --cdf_estimator

    for estimator in estimator_list:
        for alpha in ALPHA_VALUES:
            for n in N_VALUES:
                est_label = f" | CDF={estimator}" if estimator else ""
                print(f"\n{'='*62}")
                print(f"RUNNING SIMGUCOSE: Algo={algo}{est_label} | alpha={alpha} | N={n} | Trunc={TRUNC_MODE}")
                print("="*62)

                cmd = [
                    "python", "realworld_main.py",
                    "--data_type",          DATA_TYPE,
                    "--context_dim",        str(CONTEXT_DIM),
                    "--num_actions",        str(NUM_ACTIONS),
                    "--num_contexts",       str(n),
                    "--algo_group",         algo,
                    "--num_sim",            str(NUM_SIM),
                    "--layer_sizes",        LAYER_SIZES,
                    "--beta",               str(BETA),
                    "--risk_measure",       RISK_MEASURE,
                    "--alpha",              str(alpha),
                    "--tau_n",              str(TAU_N),
                    "--num_steps",          str(NUM_STEPS),
                    "--num_test_contexts",  str(NUM_TEST),
                    "--lambd0",             str(LAMBDA0),
                    "--lambd",              str(LAMBDA),
                    "--lr",                 str(LR),
                    "--policy_type",        POLICY_TYPE,
                    "--truncation_mode",    TRUNC_MODE,
                    "--agent_eval_method",  AGENT_EVAL_METHOD,
                    "--oracle_eval_method", ORACLE_EVAL_METHOD,
                    "--nouse_wandb",
                ]

                # Flag riêng cho Pessimistic CDF
                if estimator is not None:
                    cmd += [
                        "--cdf_estimator", estimator,
                        "--pess_delta",    str(PESS_DELTA),
                        "--pess_d_pi",     str(PESS_D_PI),
                    ]

                try:
                    subprocess.run(cmd, check=True)
                except Exception as e:
                    print(f"Error running simulation for {algo} at alpha={alpha}, N={n}: {e}")
                    continue

print("\nToàn bộ thực nghiệm Simglucose đã hoàn tất! Kết quả NPZ trong thư mục results/")
