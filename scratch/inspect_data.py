import numpy as np

def inspect_npz(filename):
    print(f"Inspecting {filename}:")
    with np.load(filename, allow_pickle=True) as data:
        for key in data.keys():
            val = data[key]
            if isinstance(val, np.ndarray):
                print(f" - {key}: shape={val.shape}, dtype={val.dtype}")
            else:
                print(f" - {key}: type={type(val)}")

inspect_npz("data/simglucose_offline copy.npz")
inspect_npz("data/simglucose_offline_full.npz")
