import os
import glob
import re
import numpy as np

results_root = "results"
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

def load_algorithms_from_dir(data_dir, n_list):
    prefixes = get_algo_prefixes(data_dir)
    all_results = {}
    all_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    for prefix in prefixes:
        summary = {'n_regret': [], 'regret_mean': []}
        for n in n_list:
            target_files = []
            for f in all_files:
                fname = os.path.basename(f)
                if fname.startswith(prefix) and re.search(fr"_n={n}(_|\.npz)", fname):
                    target_files.append(f)
            
            if not target_files: continue
            d = np.load(target_files[0], allow_pickle=True)
            algo_name = d['algo_names'][0].replace(' ', '_') if 'algo_names' in d else prefix

            def get_d(s): 
                if algo_name and f"{algo_name}_{s}" in d: return d[f"{algo_name}_{s}"]
                return d[s] if s in d else None

            r = get_d('regrets')
            if r is not None:
                summary['n_regret'].append(n)
                summary['regret_mean'].append(np.mean(r))
        all_results[prefix] = summary
    return all_results

all_function_data = {}
for dname in os.listdir(results_root):
    if dname.startswith("robust_syn_") and os.path.isdir(os.path.join(results_root, dname)):
        parts = dname.split('_')
        if len(parts) >= 3:
            func_name = parts[2]
            full_path = os.path.join(results_root, dname)
            all_function_data[func_name] = load_algorithms_from_dir(full_path, n_list)

for func, data in all_function_data.items():
    print(f"Function: {func}")
    for algo, summary in data.items():
        print(f"  Algo: {algo}, n_regret: {summary['n_regret']}")
