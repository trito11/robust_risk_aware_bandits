import numpy as np
from core.utils import sample_offline_policy

class RobustSyntheticData:
    """Flexible Synthetic Data Generator.
    Supports Linear/Non-Linear rewards and Light/Heavy-tailed noise.
    
    Functions from Zhou et al. (2020):
        - quadratic: h(u) = 10 * (u^T a)^2
        - quadratic2: h(u) = u^T A^T A u
        - cosine: h(u) = cos(3 * u^T a)
    """
    
    def __init__(self, num_contexts=10000, num_test_contexts=5000, 
                 context_dim=20, num_actions=30, 
                 function_type='quadratic', # linear, quadratic, quadratic2, cosine
                 noise_type='student-t',    # gaussian, student-t, binary-heavy
                 noise_std=0.1, 
                 pi='eps-greedy', eps=0.1):
        self.num_contexts = num_contexts
        self.num_test_contexts = num_test_contexts
        self.context_dim = context_dim
        self.num_actions = num_actions
        self.function_type = function_type
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.pi = pi
        self.eps = eps

    def reset_data(self, sim_id=0):
        np.random.seed(sim_id)
        
        # 1. Generate latent parameters (thetas or A)
        if self.function_type == 'quadratic2':
            # Matrix A for each action: (d, d, K)
            A = np.random.normal(0, 1, (self.context_dim, self.context_dim, self.num_actions))
        else:
            # Vectors a for each action: (d, K)
            thetas = np.random.uniform(-1, 1, (self.context_dim, self.num_actions))
            thetas /= np.linalg.norm(thetas, axis=0)[None, :]

        # 2. Reward Function Definition
        def get_mean_rewards(X):
            # X: (N, d)
            if self.function_type == 'linear':
                return X @ thetas
            elif self.function_type == 'quadratic':
                return 10.0 * np.square(X @ thetas)
            elif self.function_type == 'quadratic2':
                # sum_d (X @ A_a)^2
                res = []
                for a in range(self.num_actions):
                    res.append(np.sum(np.square(X @ A[:,:,a]), axis=1, keepdims=True))
                return np.hstack(res)
            elif self.function_type == 'cosine':
                return np.cos(3.0 * X @ thetas)
            return X @ thetas

        # 3. Generate Contexts (Uniform on unit sphere)
        def sample_contexts(n):
            X = np.random.normal(0, 1, (n, self.context_dim))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            return X

        train_contexts = sample_contexts(self.num_contexts)
        test_contexts = sample_contexts(self.num_test_contexts)
        
        train_mean_r = get_mean_rewards(train_contexts)
        test_mean_r = get_mean_rewards(test_contexts)

        # 4. Generate Noise
        def generate_noise(shape):
            if self.noise_type == 'student-t':
                 return np.random.standard_t(df=2.1, size=shape)
            elif self.noise_type == 'binary-heavy':
                 delta = np.sqrt(1.0 / (40 * self.num_contexts))
                 return np.random.choice([-1/delta, 1/delta, -delta, delta], 
                                         size=shape, p=[0.01, 0.01, 0.49, 0.49])
            else: # gaussian
                 return np.random.normal(0, self.noise_std, size=shape)

        noise = generate_noise(train_mean_r.shape)
        rewards = train_mean_r + noise
        
        # 5. Offline Data Collection
        actions = sample_offline_policy(train_mean_r, self.num_contexts, self.num_actions, self.pi, self.eps)
        
        # In this shared-context model, train_contexts is already (N, d)
        return train_contexts, actions, rewards, test_contexts, test_mean_r
