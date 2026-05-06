import numpy as np
import os

def split_and_analyze(input_path):
    print(f"Loading data from {input_path}...")
    data = np.load(input_path, allow_pickle=True)
    
    train_ctx = data['train_contexts']
    train_act = data['train_actions']
    train_rew = data['train_rewards']
    test_ctx = data['test_contexts']
    test_mean = data['test_mean']
    test_cvar = data['test_cvar']
    
    # 0: child#001, 2: adolescent#001
    patients = {0: "child#001", 2: "adolescent#001"}
    
    for p_idx, p_name in patients.items():
        print(f"\n" + "="*40)
        print(f"ANALYZING: {p_name} (idx={p_idx})")
        print("="*40)
        
        # Filter Train
        mask_tr = (train_ctx[:, 2] == p_idx)
        p_train_ctx = train_ctx[mask_tr]
        p_train_act = train_act[mask_tr]
        p_train_rew = train_rew[mask_tr]
        
        # Filter Test
        mask_te = (test_ctx[:, 2] == p_idx)
        p_test_ctx = test_ctx[mask_te]
        p_test_mean = test_mean[mask_te]
        p_test_cvar = test_cvar[mask_te]
        
        # Stats
        print(f"Samples: {len(p_train_ctx)} train, {len(p_test_ctx)} test")
        
        # Action distribution
        unique, counts = np.unique(p_train_act, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"Train Action Dist: {dist}")
        
        # Reward stats
        print(f"Train Reward Mean: {np.mean(p_train_rew):.2f}")
        print(f"Train Reward Range: [{np.min(p_train_rew):.2f}, {np.max(p_train_rew):.2f}]")
        
        # Oracle info
        oracle_acts = np.argmax(p_test_mean, axis=1)
        o_unique, o_counts = np.unique(oracle_acts, return_counts=True)
        print(f"Oracle Optimal Action Dist: {dict(zip(o_unique, o_counts))}")
        
        # Save to new file
        output_file = f"data/simglucose_{p_name.replace('#', '')}.npz"
        np.savez(output_file,
                 train_contexts=p_train_ctx,
                 train_actions=p_train_act,
                 train_rewards=p_train_rew,
                 test_contexts=p_test_ctx,
                 test_mean=p_test_mean,
                 test_cvar=p_test_cvar)
        print(f"Saved to {output_file}")

if __name__ == "__main__":
    split_and_analyze("data/simglucose_2patients.npz")
