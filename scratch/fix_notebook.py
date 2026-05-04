import json

notebook_path = 'plot_convergence.ipynb'
with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for line in source:
            if 're.search(fr"_n={n}(_|\\\\\\\\.npz)", fname)' in line:
                # Note: JSON stores backslashes escaped. 
                # In the raw file, it might look like \\\\\\\\.npz or similar.
                # Let's use a more robust replacement.
                new_line = line.replace('re.search(fr"_n={n}(_|\\\\\\\\.npz)", fname)', 're.search(fr"_n={n}(_|\.npz)", fname)')
                new_source.append(new_line)
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)
