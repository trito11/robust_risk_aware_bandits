import json
import os

notebook_path = '/home/tri/offline_rl/robust_risk_aware_bandits/plot_convergence.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Update Data Loading Cell (Cell 1 in code, index 1 in nb['cells'])
# Actually, let's just find the cell that defines load_summary_data
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'load_summary_data' in "".join(cell['source']):
        cell['source'] = [
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import os\n",
            "import glob\n",
            "import re\n",
            "\n",
            "# 1. Cấu hình đường dẫn dữ liệu\n",
            "data_dir = \"results/robust_syn_d=20_a=30_pi=eps-greedy0.1_std=0.1\"\n",
            "n_list = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]\n",
            "\n",
            "if not os.path.exists(data_dir):\n",
            "    print(f\"Không tìm thấy thư mục: {data_dir}\")\n",
            "\n",
            "def get_algo_prefixes(data_dir):\n",
            "    files = glob.glob(os.path.join(data_dir, \"*.npz\"))\n",
            "    prefixes = set()\n",
            "    for f in files:\n",
            "        basename = os.path.basename(f)\n",
            "        # Match everything before '_robust_syn_'\n",
            "        match = re.match(r\"^(.*?)_robust_syn_\", basename)\n",
            "        if match:\n",
            "            prefixes.add(match.group(1))\n",
            "    return sorted(list(prefixes))\n",
            "\n",
            "def load_all_algorithms(data_dir, n_list):\n",
            "    prefixes = get_algo_prefixes(data_dir)\n",
            "    print(f\"Tìm thấy các thuật toán: {prefixes}\")\n",
            "    \n",
            "    all_results = {}\n",
            "    \n",
            "    for prefix in prefixes:\n",
            "        summary = {'n_regret': [], 'n_extra': [], 'regret_mean': [], 'regret_std': [], \n",
            "                   'subopt_mean': [], 'subopt_std': [], 'gt_cvar_mean': [], 'gt_cvar_std': [],\n",
            "                   'oracle_cvar': []}\n",
            "        \n",
            "        for n in n_list:\n",
            "            pattern = os.path.join(data_dir, f\"{prefix}_*_n={n}_layers=*.npz\")\n",
            "            files = glob.glob(pattern)\n",
            "            if not files: continue\n",
            "                \n",
            "            d = np.load(files[0], allow_pickle=True)\n",
            "            algo_name = d['algo_names'][0].replace(' ', '_') if 'algo_names' in d else prefix\n",
            "\n",
            "            def get_d(s): \n",
            "                if algo_name and f\"{algo_name}_{s}\" in d: return d[f\"{algo_name}_{s}\"]\n",
            "                return d[s] if s in d else None\n",
            "\n",
            "            # Regret\n",
            "            r = get_d('regrets')\n",
            "            if r is not None:\n",
            "                summary['n_regret'].append(n)\n",
            "                summary['regret_mean'].append(np.mean(r))\n",
            "                summary['regret_std'].append(np.std(r) / np.sqrt(len(r.flatten())))\n",
            "            \n",
            "            # CVaR stats\n",
            "            c = get_d('gt_cvars')\n",
            "            if c is not None:\n",
            "                summary['n_extra'].append(n)\n",
            "                o_cvar = np.mean(d['oracle_cvars'])\n",
            "                summary['gt_cvar_mean'].append(np.mean(c))\n",
            "                summary['gt_cvar_std'].append(np.std(c) / np.sqrt(len(c.flatten())))\n",
            "                \n",
            "                # Subopt\n",
            "                sub = o_cvar - c\n",
            "                summary['subopt_mean'].append(np.mean(sub))\n",
            "                summary['subopt_std'].append(np.std(sub) / np.sqrt(len(c.flatten())))\n",
            "                summary['oracle_cvar'].append(o_cvar)\n",
            "        \n",
            "        all_results[prefix] = summary\n",
            "        \n",
            "    return all_results\n",
            "\n",
            "all_summaries = load_all_algorithms(data_dir, n_list)\n",
            "print(\"Dữ liệu của tất cả thuật toán đã nạp xong!\")"
        ]

# Update Regret Plot (Cell 4 in nb['cells'] roughly)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'regret_mean' in "".join(cell['source']) and 'plt.figure' in "".join(cell['source']):
        cell['source'] = [
            "plt.figure(figsize=(10, 6))\n",
            "for algo, summary in all_summaries.items():\n",
            "    if not summary['n_regret']: continue\n",
            "    plt.plot(summary['n_regret'], summary['regret_mean'], '-o', label=f\"{algo} Regret\", markersize=8)\n",
            "    plt.fill_between(summary['n_regret'], \n",
            "                     np.array(summary['regret_mean']) - np.array(summary['regret_std']), \n",
            "                     np.array(summary['regret_mean']) + np.array(summary['regret_std']), \n",
            "                     alpha=0.2)\n",
            "\n",
            "plt.xscale('log'); plt.yscale('log')\n",
            "plt.title(\"Expected Regret vs Sample Size (Log-Log)\", fontsize=14)\n",
            "plt.xlabel(\"n (Training Samples)\", fontsize=12); plt.ylabel(\"Regret\", fontsize=12)\n",
            "plt.grid(True, which=\"both\", alpha=0.3)\n",
            "plt.legend()\n",
            "plt.show()"
        ]

# Update Suboptimality Plot
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'subopt_mean' in "".join(cell['source']) and 'plt.figure' in "".join(cell['source']):
        cell['source'] = [
            "plt.figure(figsize=(10, 6))\n",
            "for algo, summary in all_summaries.items():\n",
            "    if not summary['n_extra']: continue\n",
            "    plt.plot(summary['n_extra'], summary['subopt_mean'], '-s', label=f\"{algo} Suboptimality\", markersize=8)\n",
            "    plt.fill_between(summary['n_extra'], \n",
            "                     np.array(summary['subopt_mean']) - np.array(summary['subopt_std']), \n",
            "                     np.array(summary['subopt_mean']) + np.array(summary['subopt_std']), \n",
            "                     alpha=0.2)\n",
            "\n",
            "plt.axhline(0, color='black', linestyle='--', alpha=0.6)\n",
            "plt.title(\"Suboptimality Gap vs Sample Size\", fontsize=14)\n",
            "plt.xlabel(\"n (Training Samples)\", fontsize=12); plt.ylabel(\"Gap\", fontsize=12)\n",
            "plt.grid(True, alpha=0.3)\n",
            "plt.legend()\n",
            "plt.show()"
        ]

# Update CVaR Plot
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'gt_cvar_mean' in "".join(cell['source']) and 'plt.figure' in "".join(cell['source']):
        cell['source'] = [
            "plt.figure(figsize=(10, 6))\n",
            "first = True\n",
            "for algo, summary in all_summaries.items():\n",
            "    if not summary['n_extra']: continue\n",
            "    plt.plot(summary['n_extra'], summary['gt_cvar_mean'], '-o', label=f\"{algo} CVaR\", markersize=8)\n",
            "    plt.fill_between(summary['n_extra'], \n",
            "                     np.array(summary['gt_cvar_mean']) - np.array(summary['gt_cvar_std']), \n",
            "                     np.array(summary['gt_cvar_mean']) + np.array(summary['gt_cvar_std']), \n",
            "                     alpha=0.2)\n",
            "    if first:\n",
            "        plt.plot(summary['n_extra'], summary['oracle_cvar'], 'k--', label=\"Oracle CVaR\", alpha=0.8)\n",
            "        first = False\n",
            "\n",
            "plt.title(\"Agent CVaR vs Oracle CVaR\", fontsize=14)\n",
            "plt.xlabel(\"n (Training Samples)\", fontsize=12); plt.ylabel(\"CVaR\", fontsize=12)\n",
            "plt.grid(True, alpha=0.3)\n",
            "plt.legend()\n",
            "plt.show()"
        ]

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)
