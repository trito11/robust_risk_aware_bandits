import numpy as np

def rebalance(input_path, output_path):
    print(f"Rebalancing {input_path}...")
    data = np.load(input_path, allow_pickle=True)
    
    train_ctx = list(data['train_contexts'])
    train_act = list(data['train_actions'])
    train_rew = list(data['train_rewards'])
    
    test_ctx = data['test_contexts']
    test_mean = data['test_mean']
    test_cvar = data['test_cvar']
    
    # 1. Identify indices in test where Action 3 is optimal (or just take some Action 3 rewards)
    # We'll take 50 samples from the test set to move to train
    n_move = 50
    indices_to_move = np.random.choice(len(test_ctx), n_move, replace=False)
    
    new_train_ctx = []
    new_train_act = []
    new_train_rew = []
    
    for idx in indices_to_move:
        # Move this context to train with Action 3
        new_train_ctx.append(test_ctx[idx])
        new_train_act.append(3)
        # Use the mean reward for action 3 as the train reward
        new_train_rew.append(test_mean[idx, 3])
        
    # 2. Drop some Action 0 from original train to reduce bias
    # Current train has ~900 Action 0. Let's keep only 300.
    orig_train_ctx = np.array(train_ctx)
    orig_train_act = np.array(train_act)
    orig_train_rew = np.array(train_rew)
    
    mask_a0 = (orig_train_act == 0)
    idx_a0 = np.where(mask_a0)[0]
    idx_others = np.where(~mask_a0)[0]
    
    keep_a0_count = 300
    idx_a0_kept = np.random.choice(idx_a0, keep_a0_count, replace=False)
    
    # Combined training data
    final_train_ctx = np.concatenate([orig_train_ctx[idx_others], orig_train_ctx[idx_a0_kept], np.array(new_train_ctx)])
    final_train_act = np.concatenate([orig_train_act[idx_others], orig_train_act[idx_a0_kept], np.array(new_train_act)])
    final_train_rew = np.concatenate([orig_train_rew[idx_others], orig_train_rew[idx_a0_kept], np.array(new_train_rew)])
    
    # 3. Remaining test set (remove the moved samples)
    mask_test_keep = np.ones(len(test_ctx), dtype=bool)
    mask_test_keep[indices_to_move] = False
    
    final_test_ctx = test_ctx[mask_test_keep]
    final_test_mean = test_mean[mask_test_keep]
    final_test_cvar = test_cvar[mask_test_keep]
    
    # Shuffle training data
    shuffle_idx = np.random.permutation(len(final_train_ctx))
    final_train_ctx = final_train_ctx[shuffle_idx]
    final_train_act = final_train_act[shuffle_idx]
    final_train_rew = final_train_rew[shuffle_idx]
    
    print(f"New Train Stats:")
    unique, counts = np.unique(final_train_act, return_counts=True)
    print(f" - Action Dist: {dict(zip(unique, counts))}")
    print(f" - Total Samples: {len(final_train_ctx)}")
    print(f"New Test Samples: {len(final_test_ctx)}")
    
    np.savez(output_path,
             train_contexts=final_train_ctx,
             train_actions=final_train_act,
             train_rewards=final_train_rew,
             test_contexts=final_test_ctx,
             test_mean=final_test_mean,
             test_cvar=final_test_cvar)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    rebalance("data/simglucose_2patients.npz", "data/simglucose_offline.npz")
