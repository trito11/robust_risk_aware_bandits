import numpy as np
import os

class SimglucoseData:
    def __init__(self, path='data/simglucose_offline.npz', test_ratio=0.2):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file {path} not found. Run the generator first.")
        
        data = np.load(path)
        self.all_contexts = data['contexts'].astype(np.float32)
        self.all_actions = data['actions'].astype(np.int32)
        self.all_rewards = data['rewards'].astype(np.float32)
        
        # Split Train/Test
        n = len(self.all_contexts)
        n_test = int(n * test_ratio)
        self.num_contexts = n - n_test
        self.num_test_contexts = n_test
        # Action space for bolus is typically discretized or we take max from data
        # Let's assume max action + 1 is the range
        self.num_actions = int(np.max(self.all_actions)) + 1
        self.context_dim = self.all_contexts.shape[1]
        self.noise_std = np.std(self.all_rewards)

    def reset_data(self, sim_id=0):
        # Statistically consistent split
        np.random.seed(sim_id)
        indices = np.arange(len(self.all_contexts))
        np.random.shuffle(indices)
        
        n = self.num_contexts
        train_idx = indices[:n]
        test_idx = indices[n:]
        
        train_ctx = self.all_contexts[train_idx]
        train_act = self.all_actions[train_idx]
        train_rew = self.all_rewards[train_idx]
        
        test_ctx = self.all_contexts[test_idx]
        # In simglucose we don't have true mean for all K actions readily available in the npz
        # We return a dummy matrix for test_mean_r to satisfy the runner interface
        test_mean = np.zeros((self.num_test_contexts, self.num_actions)) 
        
        return train_ctx, train_act, train_rew, test_ctx, test_mean
