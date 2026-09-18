import numpy as np
import os

class SimglucoseData:
    """
    Data loader for the advanced Simglucose Medical Bandit setup.
    Loads pre-collected Train and Test sets from .npz.
    """
    def __init__(self, path='data/simglucose_offline.npz', num_contexts=None, num_actions=3, alpha=0.05):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file {path} not found. Run the generator first.")
        
        data = np.load(path)
        self.alpha = float(alpha)
        
        # Store full training dataset pool
        self.all_train_contexts = data['train_contexts'].astype(np.float32)
        self.all_train_actions = data['train_actions'].astype(np.int32)
        self.all_train_rewards = data['train_rewards'].astype(np.float32)

        self.num_contexts = int(num_contexts) if num_contexts is not None else len(self.all_train_contexts)
        self.train_contexts = self.all_train_contexts[:self.num_contexts]
        self.train_actions = self.all_train_actions[:self.num_contexts]
        self.train_rewards = self.all_train_rewards[:self.num_contexts]
        
        self.test_contexts = data['test_contexts'].astype(np.float32)
        
        # Determine number of actions (default 3: actions 0, 1, 2)
        raw_test_mean = data['test_mean'].astype(np.float32) # Matrix (N_test, 11)
        if num_actions is not None:
            self.num_actions = int(num_actions)
        else:
            self.num_actions = raw_test_mean.shape[1]

        # Dynamic CVaR computation via raw simulation trials (Option 1)
        if 'test_trials' in data:
            # test_trials shape: (N_test, 11, n_oracle_trials)
            self.test_trials = data['test_trials'].astype(np.float32)[:, :self.num_actions, :]
            n_trials = self.test_trials.shape[-1]
            n_tail = max(1, int(np.ceil(self.alpha * n_trials)))
            sorted_trials = np.sort(self.test_trials, axis=-1)
            self.test_cvar = np.mean(sorted_trials[..., :n_tail], axis=-1)
            self.test_mean = np.mean(self.test_trials, axis=-1)
        else:
            self.test_trials = None
            self.test_mean = raw_test_mean[:, :self.num_actions]
            raw_test_cvar = data['test_cvar'].astype(np.float32) if 'test_cvar' in data else None
            self.test_cvar = raw_test_cvar[:, :self.num_actions] if raw_test_cvar is not None else None
            if self.alpha != 0.05 and self.test_cvar is not None:
                print(f"[WARNING] Simglucose file '{path}' has precomputed test_cvar at alpha=0.05 (no 'test_trials' found). "
                      f"Requested alpha={self.alpha}. Re-run generator to save 'test_trials' for exact CVaR at alpha={self.alpha}.")
        
        # Basic information
        self.num_contexts = len(self.train_contexts)
        self.num_test_contexts = len(self.test_contexts)
        self.context_dim = self.train_contexts.shape[1]

    def recompute_cvar(self, alpha):
        """Dynamically recomputes test_cvar for any given alpha level if test_trials are available."""
        self.alpha = float(alpha)
        if self.test_trials is not None:
            n_trials = self.test_trials.shape[-1]
            n_tail = max(1, int(np.ceil(self.alpha * n_trials)))
            sorted_trials = np.sort(self.test_trials, axis=-1)
            self.test_cvar = np.mean(sorted_trials[..., :n_tail], axis=-1)
        return self.test_cvar
        
    def reset_data(self, sim_id=0):
        """
        Returns a random i.i.d. subset of num_contexts from the full dataset pool for simulation sim_id.
        """
        np.random.seed(sim_id)
        total_samples = len(self.all_train_contexts)
        perm = np.random.permutation(total_samples)
        
        if self.num_contexts < total_samples:
            selected_indices = perm[:self.num_contexts]
        else:
            selected_indices = perm
        
        shuffled_train_ctx = self.all_train_contexts[selected_indices]
        shuffled_train_act = self.all_train_actions[selected_indices]
        shuffled_train_rew = self.all_train_rewards[selected_indices]
        
        return (shuffled_train_ctx, 
                shuffled_train_act, 
                shuffled_train_rew, 
                self.test_contexts, 
                self.test_mean)
