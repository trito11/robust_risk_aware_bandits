import subprocess
import itertools
import sys

# 1. Định nghĩa không gian tìm kiếm (Sweep space)
data_types = ["robust_syn", "mnist", "adult"]
num_contexts_list = [1000, 5000, 10000]  # Các mức n khác nhau
layer_configs = ["16,16", "32,32", ] # Các cấu trúc mạng khác nhau
betas = [0.01, 0.1, 1.0]                # Các mức beta khác nhau

# 2. Các tham số cố định cho nhóm robust-offline
risk_measure = "cvar"
alpha = 0.05
tau_n = 1.0
num_sim = 5
num_steps = 1000

# 3. Vòng lặp chạy tất cả các tổ hợp
for data, n, layers, beta in itertools.product(data_types, num_contexts_list, layer_configs, betas):
    print(f"\n" + "="*60)
    print(f"RUNNING: Data={data}, N={n}, Layers=[{layers}], Beta={beta}")
    print("="*60)
    
    cmd = [
        "python", "realworld_main.py",
        "--data_type", data,
        "--num_contexts", str(n),
        "--layer_sizes", layers,
        "--beta", str(beta),
        "--algo_group", "robust-offline",
        "--risk_measure", risk_measure,
        "--alpha", str(alpha),
        "--tau_n", str(tau_n),
        "--num_sim", str(num_sim),
        "--num_steps", str(num_steps),
        "--nouse_wandb"
    ]
    
    # Thực thi lệnh
    process = subprocess.run(cmd)
    
    if process.returncode != 0:
        print(f"CRITICAL: Experiment failed for config above.")
        # Tùy chọn: sys.exit(1) nếu muốn dừng toàn bộ khi có 1 lỗi
