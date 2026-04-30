import json

notebook_path = '/home/tri/offline_rl/robust_risk_aware_bandits/plot_convergence.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for line in source:
            if "plt.errorbar" in line:
                if "regret_mean" in line:
                    new_source.append("plt.plot(summary['n_regret'], summary['regret_mean'], '-o', label='Agent Regret', markersize=8)\n")
                    new_source.append("plt.fill_between(summary['n_regret'], \n")
                    new_source.append("                 np.array(summary['regret_mean']) - np.array(summary['regret_std']), \n")
                    new_source.append("                 np.array(summary['regret_mean']) + np.array(summary['regret_std']), \n")
                    new_source.append("                 alpha=0.2)\n")
                elif "subopt_mean" in line:
                    new_source.append("plt.plot(summary['n_extra'], summary['subopt_mean'], '-s', color='orange', label='Suboptimality Gap', markersize=8)\n")
                    new_source.append("plt.fill_between(summary['n_extra'], \n")
                    new_source.append("                 np.array(summary['subopt_mean']) - np.array(summary['subopt_std']), \n")
                    new_source.append("                 np.array(summary['subopt_mean']) + np.array(summary['subopt_std']), \n")
                    new_source.append("                 color='orange', alpha=0.2)\n")
                elif "gt_cvar_mean" in line:
                    new_source.append("plt.plot(summary['n_extra'], summary['gt_cvar_mean'], '-o', label='Agent CVaR', markersize=8)\n")
                    new_source.append("plt.fill_between(summary['n_extra'], \n")
                    new_source.append("                 np.array(summary['gt_cvar_mean']) - np.array(summary['gt_cvar_std']), \n")
                    new_source.append("                 np.array(summary['gt_cvar_mean']) + np.array(summary['gt_cvar_std']), \n")
                    new_source.append("                 alpha=0.2)\n")
                else:
                    new_source.append(line)
            elif "fmt='-o'" in line or "fmt='-s'" in line:
                continue
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)
