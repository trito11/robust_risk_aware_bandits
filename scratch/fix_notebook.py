import json

notebook_path = '/home/tri/offline_rl/robust_risk_aware_bandits/plot_convergence.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if 'if fname.startswith(prefix) and re.search(fr"_n={n}(_|\\\\.npz)", fname):' in line:
                line = line.replace('(_|\\\\.npz)', '(_|\\.npz)')
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)
