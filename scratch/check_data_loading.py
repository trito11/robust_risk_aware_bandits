import numpy as np
import os
import glob
import re

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
        summary = {'n_regret': [], 'n_extra': [], 'regret_mean': [], 'regret_std': [], 
                   'subopt_mean': [], 'subopt_std': [], 'gt_cvar_mean': [], 'gt_cvar_std': [],
                   'oracle_cvar': []}
        
        for n in n_list:
            target_files = []
            for f in all_files:
                fname = os.path.basename(f)
                # Correct regex: we want to match _n=X followed by either _ (for more params) or .npz (for end of name)
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
                summary['regret_std'].append(np.std(r) / np.sqrt(len(r.flatten())))
            
            c = get_d('gt_cvars')
            if c is not None:
                summary['n_extra'].append(n)
                o_cvar = np.mean(d['oracle_cvars'])
                summary['gt_cvar_mean'].append(np.mean(c))
                summary['gt_cvar_std'].append(np.std(c) / np.sqrt(len(c.flatten())))
                sub = o_cvar - c
                summary['subopt_mean'].append(np.mean(sub))
                summary['subopt_std'].append(np.std(sub) / np.sqrt(len(c.flatten())))
                summary['oracle_cvar'].append(o_cvar)
        all_results[prefix] = summary
    return all_results

data_dir = 'results/robust_syn_linear_d=20_a=30_pi=eps-greedy0.1_std=0.1'
res = load_algorithms_from_dir(data_dir, n_list)
for algo, summary in res.items():
    print(f"Algo: {algo}, n_regret: {summary['n_regret']}")
