import numpy as np
import os

def create_subset_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    data = np.load(input_path, allow_pickle=True)
    
    train_ctx = data['train_contexts']
    train_act = data['train_actions']
    train_rew = data['train_rewards']
    test_ctx = data['test_contexts']
    test_mean = data['test_mean']
    test_cvar = data['test_cvar']
    
    # Indices: 0=child#001, 1=child#002, 2=adolescent#001, 3=adult#001
    # We want 0 and 2.
    
    # Filtering Train
    mask_train = (train_ctx[:, 2] == 0) | (train_ctx[:, 2] == 2)
    filtered_train_ctx = train_ctx[mask_train]
    filtered_train_act = train_act[mask_train]
    filtered_train_rew = train_rew[mask_train]
    
    # To get exactly 1000, we take 500 from child#001 and 500 from adolescent#001
    idx_0_train = np.where(filtered_train_ctx[:, 2] == 0)[0][:500]
    idx_2_train = np.where(filtered_train_ctx[:, 2] == 2)[0][:500]
    final_train_idx = np.concatenate([idx_0_train, idx_2_train])
    
    # Filtering Test
    mask_test = (test_ctx[:, 2] == 0) | (test_ctx[:, 2] == 2)
    filtered_test_ctx = test_ctx[mask_test]
    filtered_test_mean = test_mean[mask_test]
    filtered_test_cvar = test_cvar[mask_test]
    
    # To get exactly 200, we take 100 from child#001 and 100 from adolescent#001
    idx_0_test = np.where(filtered_test_ctx[:, 2] == 0)[0][:100]
    idx_2_test = np.where(filtered_test_ctx[:, 2] == 2)[0][:100]
    final_test_idx = np.concatenate([idx_0_test, idx_2_test])
    
    new_data = {
        'train_contexts': filtered_train_ctx[final_train_idx],
        'train_actions': filtered_train_act[final_train_idx],
        'train_rewards': filtered_train_rew[final_train_idx],
        'test_contexts': filtered_test_ctx[final_test_idx],
        'test_mean': filtered_test_mean[final_test_idx],
        'test_cvar': filtered_test_cvar[final_test_idx]
    }
    
    print(f"New Dataset Stats:")
    print(f" - Train samples: {len(new_data['train_contexts'])}")
    print(f" - Test samples:  {len(new_data['test_contexts'])}")
    
    # Verification of distribution
    for i in [0, 2]:
        n_tr = np.sum(new_data['train_contexts'][:, 2] == i)
        n_te = np.sum(new_data['test_contexts'][:, 2] == i)
        name = "child#001" if i == 0 else "adolescent#001"
        print(f" - {name}: {n_tr} train, {n_te} test")
        
    np.savez(output_path, **new_data)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    create_subset_data("data/simglucose_offline.npz", "data/simglucose_2patients.npz")
