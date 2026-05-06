import numpy as np
import os
import pandas as pd

def verify_dataset(path='data/simglucose_offline.npz'):
    if not os.path.exists(path):
        print(f"Error: {path} does not exist.")
        return

    data = np.load(path)
    print("="*50)
    print(f"DATASET VERIFICATION: {path}")
    print("="*50)
    print(f"Keys: {list(data.keys())}")

    train_ctx = data['train_contexts']
    train_act = data['train_actions']
    train_rew = data['train_rewards']
    test_ctx = data['test_contexts']
    test_mean = data['test_mean']
    test_cvar = data['test_cvar'] if 'test_cvar' in data else None

    print(f"\n[BASIC STATS]")
    print(f"Train samples: {len(train_ctx)}")
    print(f"Test samples:  {len(test_ctx)}")
    print(f"Total:         {len(train_ctx) + len(test_ctx)}")

    # Patient Distribution
    print(f"\n[PATIENT DISTRIBUTION]")
    patient_names = ['child#001', 'child#002', 'adolescent#001', 'adult#001']
    for i, name in enumerate(patient_names):
        n_train = np.sum(train_ctx[:, 2] == i)
        n_test = np.sum(test_ctx[:, 2] == i)
        print(f" - {name:15}: {n_train:4} train, {n_test:4} test")

    # Reward Stats
    print(f"\n[REWARD STATS]")
    print(f"Train Reward - Min: {np.min(train_rew):.2f}, Max: {np.max(train_rew):.2f}, Mean: {np.mean(train_rew):.2f}")
    
    # Check for simulation failures (-500)
    n_fail_train = np.sum(train_rew <= -499.0)
    print(f"Train Failures (-500): {n_fail_train} ({100*n_fail_train/len(train_rew):.1f}%)")

    # Test Stats
    print(f"\n[TEST ORACLE STATS]")
    print(f"Test Mean  - Min: {np.min(test_mean):.2f}, Max: {np.max(test_mean):.2f}, Avg: {np.mean(test_mean):.2f}")
    if test_cvar is not None:
        print(f"Test CVaR  - Min: {np.min(test_cvar):.2f}, Max: {np.max(test_cvar):.2f}, Avg: {np.mean(test_cvar):.2f}")
        n_fail_test = np.sum(test_cvar <= -499.0)
        print(f"Test Failures (CVaR <= -499): {n_fail_test} ({100*n_fail_test/(test_cvar.size):.1f}% of all action-trials)")

    # Sample Check
    print(f"\n[SAMPLE DATA]")
    print(f"First 3 Train Rewards: {train_rew[:3]}")
    print(f"First Test Mean Action 0: {test_mean[0][0]:.2f}")

    print("\n" + "="*50)

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/simglucose_offline.npz'
    verify_dataset(path)
