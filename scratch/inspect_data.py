import numpy as np

def inspect_npz(filename):
    print(f"Inspecting {filename}:")
    with np.load(filename, allow_pickle=True) as data:
        for key in data.keys():
            val = data[key]
            if isinstance(val, np.ndarray):
                print(f" - {key}: shape={val.shape}, dtype={val.dtype}")
                if key == 'train_actions':
                    unique, counts = np.unique(val, return_counts=True)
                    print(f"   Action distribution: {dict(zip(unique, counts))}")
                if key == 'train_rewards':
                    print(f"   Reward range: [{np.min(val)}, {np.max(val)}], mean: {np.mean(val)}")
            else:
                print(f" - {key}: type={type(val)}")

inspect_npz("data/simglucose_2patients.npz")
