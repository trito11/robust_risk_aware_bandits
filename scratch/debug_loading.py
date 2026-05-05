import numpy as np
import os
import re
import glob

results_root = "/home/tri/offline_rl/robust_risk_aware_bandits/results"
n_list = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]

def get_algo_prefixes(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    prefixes = set()
    for f in files:
        basename = os.path.basename(f)
        match = re.match(r"^(.*?)_(robust_syn|mushroom|insulin)_", basename)
        if match:
            prefixes.add(match.group(1))
    return sorted(list(prefixes))

data_dir = "/home/tri/offline_rl/robust_risk_aware_bandits/results/robust_syn_cosine_d=20_a=30_pi=eps-greedy0.1_std=0.1"
all_files = glob.glob(os.path.join(data_dir, "*.npz"))
prefixes = get_algo_prefixes(data_dir)

print(f"Prefixes: {prefixes}")

for prefix in prefixes:
    if prefix != 'risk_lin_lcb': continue
    print(f"\nChecking prefix: {prefix}")
    for n in n_list:
        target_files = []
        for f in all_files:
            fname = os.path.basename(f)
            # FIXED regex
            pattern = fr"_n={n}(_|\.npz)"
            if fname.startswith(prefix) and re.search(pattern, fname):
                target_files.append(f)
        
        if not target_files:
            print(f"  n={n}: No files found matching prefix {prefix}")
            continue
            
        print(f"  n={n}: Found {len(target_files)} files. First: {os.path.basename(target_files[0])}")
        d = np.load(target_files[0], allow_pickle=True)
        
        algo_name = d['algo_names'][0].replace(' ', '_') if 'algo_names' in d else prefix
        
        def get_d(s): 
            if algo_name and f"{algo_name}_{s}" in d: return d[f"{algo_name}_{s}"]
            return d[s] if s in d else None

        r = get_d('regrets')
        print(f"    regrets: {'Found' if r is not None else 'NOT FOUND'}")
