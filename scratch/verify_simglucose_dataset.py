import numpy as np
import os

def verify_dataset(path='data/simglucose_offline.npz'):
    if not os.path.exists(path):
        print(f"Error: {path} does not exist.")
        return

    data = np.load(path)
    print(f"Dataset keys: {list(data.keys())}")

    train_ctx = data['train_contexts']
    train_act = data['train_actions']
    train_rew = data['train_rewards']
    test_ctx = data['test_contexts']
    test_mean = data['test_mean']

    print(f"Train contexts shape: {train_ctx.shape}")
    print(f"Train actions shape: {train_act.shape}")
    print(f"Train rewards shape: {train_rew.shape}")
    print(f"Test contexts shape: {test_ctx.shape}")
    print(f"Test mean matrix shape: {test_mean.shape}")

    print("\nSample Train Context (first 5):")
    print(train_ctx[:5])
    print("\nSample Train Actions (first 5):")
    print(train_act[:5])
    print("\nSample Train Rewards (first 5):")
    print(train_rew[:5])

    print("\nReward range:")
    print(f"Min: {np.min(train_rew)}, Max: {np.max(train_rew)}, Mean: {np.mean(train_rew)}")

    # Check if there are any NaNs
    if np.isnan(train_rew).any():
        print("Warning: Found NaNs in rewards!")
    
    # Check if test_mean has the right dimensions (n_test, 11)
    if test_mean.shape[1] != 11:
        print(f"Error: test_mean should have 11 columns, but has {test_mean.shape[1]}")
    else:
        print("Test mean matrix has correct number of actions (11).")

if __name__ == "__main__":
    verify_dataset()
