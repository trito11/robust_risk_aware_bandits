import json
import re

notebook_path = '/home/tri/offline_rl/robust_risk_aware_bandits/plot_convergence.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Regret plot
        regret_pattern = r"plt\.errorbar\(summary\['n_regret'\], summary\['regret_mean'\], yerr=summary\['regret_std'\],.*?\)"
        regret_replacement = """plt.plot(summary['n_regret'], summary['regret_mean'], '-o', label='Agent Regret', markersize=8)
plt.fill_between(summary['n_regret'], 
                 np.array(summary['regret_mean']) - np.array(summary['regret_std']), 
                 np.array(summary['regret_mean']) + np.array(summary['regret_std']), 
                 alpha=0.2)"""
        
        # Subopt plot
        subopt_pattern = r"plt\.errorbar\(summary\['n_extra'\], summary\['subopt_mean'\], yerr=summary\['subopt_std'\],.*?\)"
        subopt_replacement = """plt.plot(summary['n_extra'], summary['subopt_mean'], '-s', color='orange', label='Suboptimality Gap', markersize=8)
plt.fill_between(summary['n_extra'], 
                 np.array(summary['subopt_mean']) - np.array(summary['subopt_std']), 
                 np.array(summary['subopt_mean']) + np.array(summary['subopt_std']), 
                 color='orange', alpha=0.2)"""
        
        # CVaR plot
        cvar_pattern = r"plt\.errorbar\(summary\['n_extra'\], summary\['gt_cvar_mean'\], yerr=summary\['gt_cvar_std'\],.*?\)"
        cvar_replacement = """plt.plot(summary['n_extra'], summary['gt_cvar_mean'], '-o', label='Agent CVaR', markersize=8)
plt.fill_between(summary['n_extra'], 
                 np.array(summary['gt_cvar_mean']) - np.array(summary['gt_cvar_std']), 
                 np.array(summary['gt_cvar_mean']) + np.array(summary['gt_cvar_std']), 
                 alpha=0.2)"""

        # Also handle if already partially converted with wrong indentation
        source = re.sub(regret_pattern, regret_replacement, source, flags=re.DOTALL)
        source = re.sub(subopt_pattern, subopt_replacement, source, flags=re.DOTALL)
        source = re.sub(cvar_pattern, cvar_replacement, source, flags=re.DOTALL)
        
        # Fix the 4-space indentation issue if it happened
        # Look for lines that start with 4 spaces that shouldn't (relative to the rest of the block)
        # In this specific notebook, all code starts at column 0 in the string.
        lines = source.splitlines(keepends=True)
        new_lines = []
        for l in lines:
            if l.startswith("    plt."):
                new_lines.append(l[4:])
            else:
                new_lines.append(l)
        
        cell['source'] = new_lines

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)
