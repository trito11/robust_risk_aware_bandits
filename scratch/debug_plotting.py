import numpy as np
import glob
import os
import re

DATA_DIR = "results/simglucose_d=4_a=11_pi=eps-greedy0.1_std=N/A"
N_LIST = [50, 100, 200, 300, 400, 450, 500]

def debug_load():
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Directory {DATA_DIR} does not exist!")
        return

    all_files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    print(f"Found {len(all_files)} total .npz files.")
    
    prefixes = set()
    for f in all_files:
        basename = os.path.basename(f)
        match = re.match(r"^(.*?)_(simglucose|robust_syn|mushroom|insulin|realworld)_", basename)
        if match:
            prefixes.add(match.group(1))
        else:
            print(f"  File skipped (no regex match): {basename}")
            
    print(f"Detected prefixes: {prefixes}")
    
    for prefix in prefixes:
        print(f"\nChecking prefix: {prefix}")
        for n in N_LIST:
            target_files = []
            for f in all_files:
                fname = os.path.basename(f)
                if fname.startswith(prefix) and re.search(fr"_n={n}(_|\\.npz)", fname):
                    target_files.append(f)
            if target_files:
                print(f"  N={n}: Found {len(target_files)} files.")
            else:
                pass # print(f"  N={n}: No files found.")

if __name__ == "__main__":
    debug_load()
