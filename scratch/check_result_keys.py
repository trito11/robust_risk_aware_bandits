import numpy as np
import os

def check_keys(filepath):
    print(f"Checking keys for {filepath}")
    data = np.load(filepath, allow_pickle=True)
    print(f"Keys: {list(data.keys())}")
    if 'algo_names' in data:
        print(f"Algo names: {data['algo_names']}")

if __name__ == "__main__":
    path = "results/simglucose_d=4_a=11_pi=eps-greedy0.1_std=N/A/risk_exact_simglucose_agent=local_oracle=global_risk=cvar_alpha=0.05_beta=0.5_n=50_layers=32-32.npz"
    if os.path.exists(path):
        check_keys(path)
    else:
        print("Path not found!")
