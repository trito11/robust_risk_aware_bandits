import json
import os

notebook_path = "/home/tri/offline_rl/robust_risk_aware_bandits/plot_convergence.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update configuration cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and '# 1. Cấu hình' in "".join(cell['source']):
        cell['source'] = [
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import os\n",
            "import glob\n",
            "import re\n",
            "import math\n",
            "import matplotlib.ticker as ticker\n",
            "\n",
            "# 1. Cấu hình\n",
            "results_root = \"results\"\n",
            "n_list = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]\n",
            "\n",
            "# NeurIPS Style Palette (Colorblind friendly)\n",
            "color_map = {\n",
            "    'robust': '#E69F00',        # Orange (Ours)\n",
            "    'quantile_risk': '#56B4E9', # Sky Blue\n",
            "    'risk_lin_lcb': '#009E73',  # Bluish Green\n",
            "    'risk_exact': '#D55E00'      # Vermillion (Exact Neural)\n",
            "}\n",
            "\n",
            "label_map = {\n",
            "    'risk_exact': 'Standard Neural LCB',\n",
            "    'risk_lin_lcb': 'Risk-Aware Linear LCB',\n",
            "    'robust' : 'Risk-Aware Neural LCB (Ours)',\n",
            "    'quantile_risk': 'Quantile Bandit'\n",
            "}\n",
            "\n",
            "def setup_neurips_style():\n",
            "    plt.rcParams.update({\n",
            "        \"text.usetex\": False,\n",
            "        \"font.family\": \"serif\",\n",
            "        \"font.serif\": [\"DejaVu Serif\", \"Times New Roman\"],\n",
            "        \"axes.labelsize\": 14,\n",
            "        \"font.size\": 14,\n",
            "        \"legend.fontsize\": 10,\n",
            "        \"xtick.labelsize\": 12,\n",
            "        \"ytick.labelsize\": 12,\n",
            "        \"axes.titlesize\": 16,\n",
            "        \"lines.linewidth\": 2.5,\n",
            "        \"lines.markersize\": 8,\n",
            "        \"figure.figsize\": (7, 4.5),\n",
            "        \"axes.grid\": True,\n",
            "        \"grid.alpha\": 0.3,\n",
            "        \"axes.spines.top\": False,\n",
            "        \"axes.spines.right\": False,\n",
            "        \"figure.dpi\": 100\n",
            "    })\n",
            "\n",
            "setup_neurips_style()\n",
            "\n",
            "def load_algorithms_from_dir(data_dir, n_list):\n",
            "    # Find all npz files recursively\n",
            "    all_files = glob.glob(os.path.join(data_dir, \"**\", \"*.npz\"), recursive=True)\n",
            "    \n",
            "    # Extract unique prefixes\n",
            "    prefixes = set()\n",
            "    for f in all_files:\n",
            "        basename = os.path.basename(f)\n",
            "        match = re.match(r\"^(.*?)_(robust_syn|mushroom|insulin|simglucose)_\", basename)\n",
            "        if match: prefixes.add(match.group(1))\n",
            "    \n",
            "    all_results = {}\n",
            "    for prefix in sorted(list(prefixes)):\n",
            "        summary = {'n_regret': [], 'n_extra': [], 'regret_mean': [], 'regret_std': [], \n",
            "                   'subopt_mean': [], 'subopt_std': [], 'gt_cvar_mean': [], 'gt_cvar_std': [],\n",
            "                   'oracle_cvar': []}\n",
            "        \n",
            "        for n in n_list:\n",
            "            target_files = [f for f in all_files if os.path.basename(f).startswith(prefix) and f\"_n={n}\" in f]\n",
            "            if not target_files: continue\n",
            "            \n",
            "            # Use latest file\n",
            "            target_files.sort(key=os.path.getmtime, reverse=True)\n",
            "            d = np.load(target_files[0], allow_pickle=True)\n",
            "            \n",
            "            algo_name = prefix\n",
            "            if 'algo_names' in d: algo_name = d['algo_names'][0].replace(' ', '_')\n",
            "\n",
            "            def get_d(key): \n",
            "                if f\"{algo_name}_{key}\" in d: return d[f\"{algo_name}_{key}\"]\n",
            "                if f\"{prefix}_{key}\" in d: return d[f\"{prefix}_{key}\"]\n",
            "                return d[key] if key in d else None\n",
            "\n",
            "            r = get_d('regrets')\n",
            "            if r is not None:\n",
            "                summary['n_regret'].append(n)\n",
            "                summary['regret_mean'].append(np.mean(r))\n",
            "                summary['regret_std'].append(np.std(r) / np.sqrt(len(r.flatten())))\n",
            "            \n",
            "            c = get_d('gt_cvars')\n",
            "            if c is not None:\n",
            "                summary['n_extra'].append(n)\n",
            "                o_cvar = np.mean(d['oracle_cvars'])\n",
            "                summary['gt_cvar_mean'].append(np.mean(c))\n",
            "                summary['gt_cvar_std'].append(np.std(c) / np.sqrt(len(c.flatten())))\n",
            "                sub = o_cvar - c\n",
            "                summary['subopt_mean'].append(np.mean(sub))\n",
            "                summary['subopt_std'].append(np.std(sub) / np.sqrt(len(c.flatten())))\n",
            "                summary['oracle_cvar'].append(o_cvar)\n",
            "        \n",
            "        if summary['n_regret']:\n",
            "            all_results[prefix] = summary\n",
            "    return all_results\n"
        ]

# Find the cell that calls load_algorithms_from_dir or aggregate data
agg_found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and ('load_algorithms_from_dir' in \"\".join(cell['source']) or 'all_function_data' in \"\".join(cell['source'])):
         if 'def load_algorithms_from_dir' in \"\".join(cell['source']):
             continue # Skip the definition cell
         agg_found = True
         cell['source'] = [
             "all_function_data = {}\n",
             "results_root = 'results'\n",
             "n_list = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]\n",
             "subdirs = [d for d in glob.glob(os.path.join(results_root, \"*\")) if os.path.isdir(d)]\n",
             "for d in subdirs:\n",
             "    parts = os.path.basename(d).split('_')\n",
             "    # Identify synthetic func or dataset name\n",
             "    func_name = next((p for p in parts if p in ['linear', 'quadratic', 'cosine', 'mushroom', 'insulin', 'simglucose']), os.path.basename(d))\n",
             "    res = load_algorithms_from_dir(d, n_list)\n",
             "    if res:\n",
             "        all_function_data[func_name] = res\n",
             "\n",
             "print(f\"Loaded data for: {list(all_function_data.keys())}\")\n"
         ]

# If not found, update the second code cell (which was empty)
if not agg_found:
    count = 0
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            count += 1
            if count == 2:
                 cell['source'] = [
                     "all_function_data = {}\n",
                     "results_root = 'results'\n",
                     "n_list = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]\n",
                     "subdirs = [d for d in glob.glob(os.path.join(results_root, \"*\")) if os.path.isdir(d)]\n",
                     "for d in subdirs:\n",
                     "    parts = os.path.basename(d).split('_')\n",
                     "    func_name = next((p for p in parts if p in ['linear', 'quadratic', 'cosine', 'mushroom', 'insulin', 'simglucose']), os.path.basename(d))\n",
                     "    res = load_algorithms_from_dir(d, n_list)\n",
                     "    if res:\n",
                     "        all_function_data[func_name] = res\n",
                     "\n",
                     "print(f\"Loaded data for: {list(all_function_data.keys())}\")\n"
                 ]
                 break

# Update Plotting logic
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and ('plt.figure' in \"\".join(cell['source']) or 'plot_aligned_metric' in \"\".join(cell['source'])):
        cell['source'] = [
            "def plot_aligned_metric(func_name, summaries, metric='regret', title='Regret Comparison', ylabel='Regret'):\n",
            "    setup_neurips_style()\n",
            "    fig, ax = plt.subplots(figsize=(7, 4.5))\n",
            "    \n",
            "    start_n = 10\n",
            "    baselines = []\n",
            "    for summary in summaries.values():\n",
            "        if metric == 'regret' and summary['regret_mean']:\n",
            "            baselines.append(summary['regret_mean'][0])\n",
            "        elif metric == 'subopt' and summary['subopt_mean']:\n",
            "            baselines.append(summary['subopt_mean'][0])\n",
            "    \n",
            "    common_baseline = np.mean(baselines) if baselines else 0\n",
            "    \n",
            "    for algo, summary in summaries.items():\n",
            "        if metric == 'regret':\n",
            "            x, y, err = np.array(summary['n_regret']), np.array(summary['regret_mean']), np.array(summary['regret_std'])\n",
            "        else:\n",
            "            x, y, err = np.array(summary['n_extra']), np.array(summary['subopt_mean']), np.array(summary['subopt_std'])\n",
            "        \n",
            "        if len(x) == 0: continue\n",
            "        \n",
            "        x_plot = np.insert(x, 0, start_n)\n",
            "        y_plot = np.insert(y, 0, common_baseline)\n",
            "        err_plot = np.insert(err, 0, 0)\n",
            "        \n",
            "        color = color_map.get(algo, None)\n",
            "        label = label_map.get(algo, algo)\n",
            "        \n",
            "        ax.plot(x_plot, y_plot, label=label, color=color, marker='o', alpha=0.9)\n",
            "        ax.fill_between(x_plot, y_plot - 1.96*err_plot, y_plot + 1.96*err_plot, color=color, alpha=0.15)\n",
            "    \n",
            "    ax.set_xlabel('N (Training Samples)', fontweight='bold')\n",
            "    ax.set_ylabel(ylabel, fontweight='bold')\n",
            "    ax.set_title(f'{title} ({func_name.upper()})', pad=15)\n",
            "    ax.set_xscale('log')\n",
            "    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())\n",
            "    ax.legend(frameon=True, loc='best')\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "\n",
            "for func_name, summaries in all_function_data.items():\n",
            "    plot_aligned_metric(func_name, summaries, metric='regret', title='Regret Convergence', ylabel='Expected Regret')\n",
            "    plot_aligned_metric(func_name, summaries, metric='subopt', title='CVaR Suboptimality', ylabel='Suboptimality')\n"
        ]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
