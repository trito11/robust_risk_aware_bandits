import numpy as np
import os

class SimglucoseData:
    """
    Data loader for the advanced Simglucose Medical Bandit setup.
    Loads pre-collected Train and Test sets from .npz.
    """
    def __init__(self, path='data/simglucose_offline.npz', num_contexts=None):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file {path} not found. Run the generator first.")
        
        data = np.load(path)
        
        # Load and potentially slice training data
        all_train_ctx = data['train_contexts'].astype(np.float32)
        all_train_act = data['train_actions'].astype(np.int32)
        all_train_rew = data['train_rewards'].astype(np.float32)

        if num_contexts is not None and num_contexts < len(all_train_ctx):
            self.train_contexts = all_train_ctx[:num_contexts]
            self.train_actions = all_train_act[:num_contexts]
            self.train_rewards = all_train_rew[:num_contexts]
        else:
            self.train_contexts = all_train_ctx
            self.train_actions = all_train_act
            self.train_rewards = all_train_rew
        
        self.test_contexts = data['test_contexts'].astype(np.float32)
        self.test_mean = data['test_mean'].astype(np.float32) # Matrix (N_test, 11)
        
        # Basic information
        self.num_contexts = len(self.train_contexts)
        self.num_test_contexts = len(self.test_contexts)
        self.num_actions = 11  # Discretized bolus units 0-10
        self.context_dim = self.train_contexts.shape[1]
        
    def reset_data(self, sim_id=0):
        """
        Returns the data for the current simulation with optional shuffling.
        """
        np.random.seed(sim_id)
        
        # Shuffle training data to get different samples for each simulation
        indices = np.arange(len(self.train_contexts))
        np.random.shuffle(indices)
        
        shuffled_train_ctx = self.train_contexts[indices]
        shuffled_train_act = self.train_actions[indices]
        shuffled_train_rew = self.train_rewards[indices]
        
        return (shuffled_train_ctx, 
                shuffled_train_act, 
                shuffled_train_rew, 
                self.test_contexts, 
                self.test_mean)
