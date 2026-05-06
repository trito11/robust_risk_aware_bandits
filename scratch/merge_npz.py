import numpy as np
import os

def merge_npz(file1, file2, out_file):
    print(f"Merging {file1} and {file2} into {out_file}...")
    
    with np.load(file1, allow_pickle=True) as data1:
        with np.load(file2, allow_pickle=True) as data2:
            
            keys = set(data1.keys()) | set(data2.keys())
            merged_data = {}
            
            for key in keys:
                if key in data1 and key in data2:
                    d1 = data1[key]
                    d2 = data2[key]
                    
                    # Handle empty arrays
                    if isinstance(d1, np.ndarray) and d1.size == 0:
                        merged_data[key] = d2
                        print(f" - Key '{key}': File 1 is empty, taking File 2 ({len(d2)} samples)")
                        continue
                    if isinstance(d2, np.ndarray) and d2.size == 0:
                        merged_data[key] = d1
                        print(f" - Key '{key}': File 2 is empty, taking File 1 ({len(d1)} samples)")
                        continue
                    
                    # Concatenate if both are non-empty
                    if isinstance(d1, np.ndarray) and isinstance(d2, np.ndarray):
                        if d1.ndim == d2.ndim:
                            merged_data[key] = np.concatenate([d1, d2], axis=0)
                        else:
                            # Try to fix dim mismatch if one is effectively empty or single sample
                            print(f" ! Dim mismatch for '{key}': {d1.shape} vs {d2.shape}. Skipping concatenation and taking File 2.")
                            merged_data[key] = d2
                    else:
                        merged_data[key] = list(d1) + list(d2)
                    
                    print(f" - Key '{key}': {len(d1)} + {len(d2)} -> {len(merged_data[key])} samples")
                elif key in data1:
                    merged_data[key] = data1[key]
                    print(f" - Key '{key}': {len(data1[key])} samples (only in file 1)")
                else:
                    merged_data[key] = data2[key]
                    print(f" - Key '{key}': {len(data2[key])} samples (only in file 2)")
            
            np.savez(out_file, **merged_data)
            print(f"Successfully saved to {out_file}")

if __name__ == "__main__":
    f1 = "data/simglucose_offline copy.npz"
    f2 = "data/simglucose_offline_full.npz"
    out = "data/simglucose_offline.npz"
    
    merge_npz(f1, f2, out)
