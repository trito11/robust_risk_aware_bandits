import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

# 1. Configuration
DATA_DIR = "results/simglucose_d=4_a=11_pi=eps-greedy0.1_std=N/A"
N_LIST = [100, 200, 500, 1000]

color_map = {
    'robust': '#ff7f0e',        # Orange (Risk Neural LCB)
    'quantile_risk': '#1f77b4', # Blue (Direct Distributional)
    'risk_lin_lcb': '#2ca02c',  # Green (LinLCB)
    'risk_exact': '#d62728',    # Red (Standard Neural LCB)
    'neural_regression': '#9467bd' # Purple (Baseline)
}

label_map = {
    'risk_exact': 'Standard Neural LCB',
    'risk_lin_lcb': 'LinLCB',
    'robust': 'Risk Neural LCB',
    'quantile_risk': 'Direct Distributional Estimation',
    'neural_regression': 'Neural Regression'
}

def get_algo_prefixes(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    prefixes = set()
    for f in files:
        basename = os.path.basename(f)
        # Matches prefixes like robust_simglucose or risk_exact_simglucose
        match = re.match(r"^(.*?)_simglucose_", basename)
        if match:
            prefixes.add(match.group(1))
    return sorted(list(prefixes))

def load_data(data_dir, n_list):
    prefixes = get_algo_prefixes(data_dir)
    print(f"Found prefixes: {prefixes}")
    all_results = {}
    all_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    for prefix in prefixes:
        summary = {
            'n': [], 
            'regret_mean': [], 'regret_std': [], 
            'subopt_mean': [], 'subopt_std': [],
            'acc_mean': [], 'acc_std': []
        }
        
        for n in n_list:
            target_files = []
            for f in all_files:
                fname = os.path.basename(f)
                # Ensure prefix match and N match
                if fname.startswith(prefix) and re.search(fr"_n={n}(_|\\.npz)", fname):
                    target_files.append(f)
            
            if not target_files:
                continue
            
            # Load the first matching file for this (prefix, n)
            d = np.load(target_files[0], allow_pickle=True)
            
            # Identify the key used in the npz
            # The keys are usually like 'RobustOfflineBatchNeuraLCB_regrets'
            algo_key = None
            for key in d.keys():
                if key.endswith("_regrets"):
                    algo_key = key.replace("_regrets", "")
                    break
            
            if algo_key is None:
                print(f"Warning: No regrets key found for {prefix} N={n}")
                continue

            # Extract metrics
            regrets = d[f"{algo_key}_regrets"]
            accs = d[f"{algo_key}_accs"]
            subopts = d[f"{algo_key}_subopt"]
            
            summary['n'].append(n)
            summary['regret_mean'].append(np.mean(regrets))
            summary['regret_std'].append(np.std(regrets) / np.sqrt(len(regrets)))
            
            summary['acc_mean'].append(np.mean(accs))
            summary['acc_std'].append(np.std(accs) / np.sqrt(len(accs)))
            
            summary['subopt_mean'].append(np.mean(subopts))
            summary['subopt_std'].append(np.std(subopts) / np.sqrt(len(subopts)))
            
        if summary['n']:
            all_results[prefix] = summary
            
    return all_results

def plot_results(all_results):
    metrics = [
        ('regret_mean', 'regret_std', 'Average Regret (Lower is better)'),
        ('subopt_mean', 'subopt_std', 'Suboptimality (CVaR Gap)'),
        ('acc_mean', 'acc_std', 'Action Accuracy (Higher is better)')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, (mean_key, std_key, title) in enumerate(metrics):
        ax = axes[i]
        for prefix, data in all_results.items():
            n = data['n']
            m = np.array(data[mean_key])
            s = np.array(data[std_key])
            
            label = label_map.get(prefix, prefix)
            color = color_map.get(prefix, None)
            
            ax.plot(n, m, marker='o', label=label, color=color, linewidth=2)
            ax.fill_between(n, m - s, m + s, color=color, alpha=0.2)
            
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Training Samples (N)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        if i == 0:
            ax.legend(fontsize=10)
            
    plt.tight_layout()
    output_plot = "simglucose_convergence.png"
    plt.savefig(output_plot, dpi=300)
    print(f"Plot saved to {output_plot}")
    plt.show()

if __name__ == "__main__":
    results = load_data(DATA_DIR, N_LIST)
    if results:
        plot_results(results)
    else:
        print("No results found to plot.")
