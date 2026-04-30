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
            self.A = np.random.normal(0, 1, (self.context_dim, self.context_dim, self.num_actions))
        else:
            # Vectors a for each action: (d, K)
            self.thetas = np.random.uniform(-1, 1, (self.context_dim, self.num_actions))
            self.thetas /= np.linalg.norm(self.thetas, axis=0)[None, :]

        # 2. Reward Function Definition
        def get_mean_rewards(X):
            # X: (N, d)
            if self.function_type == 'linear':
                return X @ self.thetas
            elif self.function_type == 'quadratic':
                return 10.0 * np.square(X @ self.thetas)
            elif self.function_type == 'quadratic2':
                res = []
                for a in range(self.num_actions):
                    res.append(np.sum(np.square(X @ self.A[:,:,a]), axis=1, keepdims=True))
                return np.hstack(res)
            elif self.function_type == 'cosine':
                return np.cos(3.0 * X @ self.thetas)
            return X @ self.thetas
        self.get_mean_rewards = get_mean_rewards

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
        noise = self.generate_noise(train_mean_r.shape)
        rewards = train_mean_r + noise
        
        # 5. Offline Data Collection
        actions = sample_offline_policy(train_mean_r, self.num_contexts, self.num_actions, self.pi, self.eps)
        
        # In this shared-context model, train_contexts is already (N, d)
        return train_contexts, actions, rewards, test_contexts, test_mean_r

    def generate_noise(self, shape):
        if self.noise_type == 'student-t':
                return np.random.standard_t(df=2.1, size=shape)
        elif self.noise_type == 'binary-heavy':
                delta = np.sqrt(1.0 / (40 * self.num_contexts))
                return np.random.choice([-1/delta, 1/delta, -delta, delta], 
                                        size=shape, p=[0.01, 0.01, 0.49, 0.49])
        else: # gaussian
                return np.random.normal(0, self.noise_std, size=shape)

    def get_oracle_optimal_actions(self, test_ctx, noise_samples, alpha):
        """Standard point-wise oracle (argmax of CVaR for each context)."""
        test_mean = self.get_mean_rewards(test_ctx)
        # In this i.i.d noise case, argmax CVaR is equivalent to argmax Mean.
        return np.argmax(test_mean, axis=1)

    def get_oracle_optimal_actions_milp(self, test_ctx, alpha, noise_samples=None):
        """Marginal MILP oracle using true means."""
        test_mean = self.get_mean_rewards(test_ctx)
        num_test = test_ctx.shape[0]
        num_actions = self.num_actions
        
        if noise_samples is None:
            # Fallback to mean if no noise samples provided
            return np.argmax(test_mean, axis=1)
            
        num_noise = noise_samples.shape[0]
        
        import gurobipy as gp
        from gurobipy import GRB
        
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        model = gp.Model("Oracle_Marginal_CVaR", env=env)
        
        # Decision variables: z[i, a] = 1 if action a is chosen for context i
        z = model.addVars(num_test, num_actions, vtype=GRB.BINARY, name="z")
        
        # Auxiliary variables for CVaR
        zeta = model.addVar(lb=-GRB.INFINITY, name="zeta")
        u = model.addVars(num_test, num_noise, lb=0.0, name="u")
        
        # Constraints: One action per context
        for i in range(num_test):
            model.addConstr(gp.quicksum(z[i, a] for a in range(num_actions)) == 1)
            
        # CVaR objective constraints: u_{i,k} >= zeta - (sum_a z_{i,a} * mu_{i,a} + noise_k)
        for i in range(num_test):
            for k in range(num_noise):
                reward_ik = gp.quicksum(z[i, a] * test_mean[i, a] for a in range(num_actions)) + noise_samples[k]
                model.addConstr(u[i, k] >= zeta - reward_ik)
                
        # Objective: Maximize CVaR = zeta - (1/(N*alpha)) * sum(u)
        model.setObjective(zeta - (1.0 / (num_test * num_noise * alpha)) * gp.quicksum(u[i, k] for i in range(num_test) for k in range(num_noise)), GRB.MAXIMIZE)
        
        model.optimize()
        
        res_actions = np.zeros(num_test, dtype=int)
        if model.status == GRB.OPTIMAL:
            for i in range(num_test):
                for a in range(num_actions):
                    if z[i, a].X > 0.5:
                        res_actions[i] = a
                        break
        else:
            # Fallback
            res_actions = np.argmax(test_mean, axis=1)
            
        return res_actions
