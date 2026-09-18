import subprocess
import os

# ──────────────────────────────────────────────────────────────────
# 1. Cấu hình thực nghiệm
# ──────────────────────────────────────────────────────────────────
ALGO_GROUPS = [
    "risk-exact",
    "risk-lin-lcb",
    "pessimistic-cdf",        # paper arXiv:2605.15620, context-independent
    # "pessimistic-cdf-ctx",  # context-aware (chậm hơn, bỏ comment để chạy)
]

FUNCTION_TYPES = ["linear", "quadratic"]
N_VALUES       = [100, 500, 1000, 5000, 10000, 20000, 50000]
ALPHA_VALUES   = [0.01, 0.05, 0.1, 0.2]

# ──────────────────────────────────────────────────────────────────
# 2. Cấu hình cố định
# ──────────────────────────────────────────────────────────────────
DATA_TYPE           = "robust_syn"
NOISE_TYPE          = "student-t"
CONTEXT_DIM         = 20
NUM_ACTIONS         = 30
LAYER_SIZES         = "32,32"
RISK_MEASURE        = "cvar"
TAU_N               = 1.0
NUM_SIM             = 10
NUM_TEST            = 500
AGENT_EVAL_METHOD   = "local"
ORACLE_EVAL_METHOD  = "global"
LAMBDA0             = 10.0
POLICY_TYPE         = "risk-aware"

# ──────────────────────────────────────────────────────────────────
# 3. Cấu hình riêng cho Pessimistic CDF (arXiv:2605.15620)
# ──────────────────────────────────────────────────────────────────
# Chọn một hoặc nhiều estimator để so sánh:
#   "IS"  – clipped Importance Sampling (Eq. 5, Theorem 2)
#   "WIS" – Weighted IS (Eq. 6, Theorem 3)
#   "DR"  – Doubly Robust (Eq. 8, Theorem 4)
PESS_CDF_ESTIMATORS = ["IS", "WIS", "DR"]   # sweep cả 3
PESS_DELTA          = 0.05   # confidence level cho R(pi)
PESS_D_PI           = 1      # policy class complexity d_Pi

# ──────────────────────────────────────────────────────────────────
# 4. Vòng lặp thực nghiệm
# ──────────────────────────────────────────────────────────────────
for algo in ALGO_GROUPS:
    for func in FUNCTION_TYPES:

        # Tinh chỉnh hyperparams theo thuật toán
        if algo == "quantile-risk":
            BETA = 0.005; NUM_STEPS = 3000; LR = 5e-4; LAMBDA = 1e-3
        elif algo in ("pessimistic-cdf", "pessimistic-cdf-ctx"):
            BETA = 0.05;  NUM_STEPS = 1000; LR = 1e-3; LAMBDA = 1e-4
        else:
            BETA = 0.1;   NUM_STEPS = 1000; LR = 1e-3; LAMBDA = 1e-4

        # Với pessimistic-cdf: sweep từng estimator
        if algo in ("pessimistic-cdf", "pessimistic-cdf-ctx"):
            estimator_list = PESS_CDF_ESTIMATORS
        else:
            estimator_list = [None]   # không dùng flag --cdf_estimator

        for estimator in estimator_list:
            for alpha in ALPHA_VALUES:
                for n in N_VALUES:
                    est_label = f" | CDF={estimator}" if estimator else ""
                    print(f"\n{'='*62}")
                    print(f"Algo={algo}{est_label} | Func={func} | alpha={alpha} | N={n}")
                    print("="*62)

                    cmd = [
                        "python", "realworld_main.py",
                        "--data_type",          DATA_TYPE,
                        "--function_type",      func,
                        "--noise_type",         NOISE_TYPE,
                        "--context_dim",        str(CONTEXT_DIM),
                        "--num_actions",        str(NUM_ACTIONS),
                        "--num_contexts",       str(n),
                        "--beta",               str(BETA),
                        "--risk_measure",       RISK_MEASURE,
                        "--alpha",              str(alpha),
                        "--tau_n",              str(TAU_N),
                        "--num_sim",            str(NUM_SIM),
                        "--num_steps",          str(NUM_STEPS),
                        "--num_test_contexts",  str(NUM_TEST),
                        "--algo_group",         algo,
                        "--agent_eval_method",  AGENT_EVAL_METHOD,
                        "--oracle_eval_method", ORACLE_EVAL_METHOD,
                        "--lambd0",             str(LAMBDA0),
                        "--lambd",              str(LAMBDA),
                        "--lr",                 str(LR),
                        "--policy_type",        POLICY_TYPE,
                        "--layer_sizes",        LAYER_SIZES,
                        "--nouse_wandb",
                    ]

                    # Flag riêng cho Pessimistic CDF
                    if estimator is not None:
                        cmd += [
                            "--cdf_estimator", estimator,
                            "--pess_delta",    str(PESS_DELTA),
                            "--pess_d_pi",     str(PESS_D_PI),
                        ]

                    subprocess.run(cmd)

print(f"\nToàn bộ thực nghiệm đã hoàn tất! Kết quả NPZ trong thư mục results/")
