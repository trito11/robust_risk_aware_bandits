import glob
import os
import re

data_dir = 'results/robust_syn_linear_d=20_a=30_pi=eps-greedy0.1_std=0.1'
files = glob.glob(os.path.join(data_dir, '*.npz'))
prefixes = set()
for f in files:
    basename = os.path.basename(f)
    match = re.match(r"^(.*?)_(robust_syn|mushroom|insulin)_", basename)
    if match:
        prefixes.add(match.group(1))
    else:
        print(f"No match for {basename}")
print(f"Final prefixes: {sorted(list(prefixes))}")
