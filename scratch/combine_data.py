import numpy as np
import os

file1 = 'data/simglucose_offline.npz'
file2 = 'data/simglucose_offline1.npz'

def inspect_npz(path):
    print(f"\nInspecting {path}:")
    data = np.load(path)
    for key in data.keys():
        print(f"  {key}: {data[key].shape}")
    return data

data1 = inspect_npz(file1)
data2 = inspect_npz(file2)

combined = {}
# Keys to concatenate: 
# 'train_contexts', 'train_actions', 'train_rewards' are likely the ones that need merging.
# 'test_contexts', 'test_mean', 'test_cvar' might also be unique or shared.
# Looking at the verify script output:
# Train samples: 408, Test samples: 192 (Total 600)
# Usually we want to combine both train and test samples if they are from different runs.

for key in data1.keys():
    if key in data2:
        combined[key] = np.concatenate([data1[key], data2[key]], axis=0)
        print(f"Combined {key}: {combined[key].shape}")
    else:
        combined[key] = data1[key]
        print(f"Key {key} only in file1, copied.")

output_file = 'data/simglucose_offline_combined.npz'
np.savez(output_file, **combined)
print(f"\nSaved combined data to {output_file}")
