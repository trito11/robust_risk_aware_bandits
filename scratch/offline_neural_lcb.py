class OfflineBatchNeuraLCB(BanditAlgorithm):
    """Standard Offline Batch NeuraLCB with Reward Clipping.
    
    This algorithm is the standard NeuraLCB baseline but adapted for the pure offline
    batch regime. It uses standard confidence bounds (no risk measures for action selection),
    but includes reward clipping to avoid long-tail gradient collapse, and evaluates
    risk metrics to allow fair comparison against RobustOfflineBatchNeuraLCB.
    """

    def __init__(self, hparams, update_freq=1, name='OfflineBatchNeuraLCB'):
        self.name = name
        self.hparams = hparams
        self.update_freq = update_freq

        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModelV2(opt, hparams, '{}-net'.format(name))
        
        self.data = BanditDataset(
            hparams.context_dim,
            hparams.num_actions,
            hparams.buffer_s,
            '{}-data'.format(name),
        )

        self.historical_residuals = None  # shape (N,)
        self.rho_residuals = None         # scalar
        self.Z_inv = None                 # shape (p, p)

    def _compute_risk_functional(self, Y, axis=-1):
        """Computes the requested risk measure on the sample distribution Y.
        (Used purely for evaluation/comparison)."""
        measure = getattr(self.hparams, 'risk_measure', 'cvar')
        
        if measure == 'mean':
            return jnp.mean(Y, axis=axis)
        elif measure == 'cvar':
            alpha = self.hparams.alpha
            N = Y.shape[axis]
            k = jnp.maximum(1, int(alpha * N))
            Y_sorted = jnp.sort(Y, axis=axis)
            if axis == -1 or axis == 1:
                return jnp.mean(Y_sorted[:, :k], axis=axis)
            else:
                return jnp.mean(Y_sorted[:k], axis=axis)
        elif measure == 'entropic':
            theta = getattr(self.hparams, 'entropic_theta', 1.0)
            N = Y.shape[axis]
            lse = jax.scipy.special.logsumexp(-theta * Y, axis=axis)
            return -(1.0 / theta) * (lse - jnp.log(N))
        elif measure == 'mean_variance':
            lam = getattr(self.hparams, 'variance_lambda', 0.1)
            return jnp.mean(Y, axis=axis) - lam * jnp.var(Y, axis=axis)
        else:
            raise ValueError(f"Unknown risk_measure: {measure}")

    def reset(self, seed):
        self.nn.reset(seed)
        self.data.reset()
        self.historical_residuals = None
        self.Z_inv = None

    def train_offline_batch(self, contexts, actions, rewards):
        """Train the network on the full offline dataset and compute Z_inv."""
        tau_n = self.hparams.tau_n
        truncation_mode = getattr(self.hparams, 'truncation_mode', 'clip')

        if truncation_mode == 'clip':
            # r_tilde = r * I(|r| <= tau) + tau * sgn(r) * I(|r| > tau)
            r_tilde = jnp.where(
                jnp.abs(rewards) <= tau_n,
                rewards,
                tau_n * jnp.sign(rewards)
            )
            train_ctx, train_act, train_rew = contexts, actions, r_tilde
        elif truncation_mode == 'mask':
            mask = (jnp.abs(rewards) <= tau_n).ravel()
            train_ctx, train_act, train_rew = contexts[mask], actions[mask], rewards[mask]
        else:
            train_ctx, train_act, train_rew = contexts, actions, rewards

        self.data.reset()
        self.data.add(train_ctx, train_act.reshape(-1, 1), train_rew.reshape(-1, 1))
        self.nn.train(self.data, self.hparams.num_steps)

        # Residuals are calculated using RAW rewards to preserve heavy-tail info for risk evaluation.
        f_hist = self.nn.out(self.nn.params, contexts, actions).ravel()
        self.historical_residuals = rewards.ravel() - f_hist

        p = self.nn.num_params
        Z = self.hparams.lambd0 * jnp.eye(p)
        
        num_train = contexts.shape[0]
        z_chunk_size = getattr(self.hparams, 'chunk_size', 500)
        
        if self.hparams.verbose:
            print(f'[{self.name}] Computing Z matrix in chunks (p={p})...')
            pbar = tqdm(total=num_train, desc="Computing Z Matrix", unit="samples")

        for i in range(0, num_train, z_chunk_size):
            end_idx = min(i + z_chunk_size, num_train)
            g_chunk = self.nn.grad_out(self.nn.params, contexts[i:end_idx], actions[i:end_idx]) / jnp.sqrt(self.nn.m)
            Z = Z + g_chunk.T @ g_chunk
            del g_chunk
            if self.hparams.verbose:
                pbar.update(end_idx - i)

        if self.hparams.verbose:
            pbar.close()

        self.Z_inv = jnp.linalg.inv(Z)
        del Z

        # Evaluate risk functional of residuals purely for offline evaluation compatibility
        self.rho_residuals = self._compute_risk_functional(self.historical_residuals, axis=0)

        if self.hparams.verbose:
            print(f'[{self.name}] Batch Training: mean |resid|={jnp.mean(jnp.abs(self.historical_residuals)):.4f} | rho(resid)={self.rho_residuals:.4f}')

    def sample_action(self, contexts):
        """Chooses actions using standard Lower Confidence Bounds (No Risk Functional)."""
        assert self.historical_residuals is not None, "Call train_offline_batch() first."
        assert self.Z_inv is not None, "Call train_offline_batch() first."

        num_contexts = contexts.shape[0]
        chunk_size = getattr(self.hparams, 'chunk_size', 500)
        all_actions = []

        if self.hparams.verbose:
            pbar = tqdm(total=num_contexts, desc=f"Evaluating actions (chunk_size={chunk_size})", unit="samples")

        for i in range(0, num_contexts, chunk_size):
            batch_contexts = contexts[i : i + chunk_size]
            B = batch_contexts.shape[0]
            lcbs = []
            
            for a in range(self.hparams.num_actions):
                actions_tmp = jnp.ones(shape=(B,)) * a
                mu_a = self.nn.out(self.nn.params, batch_contexts, actions_tmp).ravel()

                g_test = self.nn.grad_out(self.nn.params, batch_contexts, actions_tmp) / jnp.sqrt(self.nn.m)
                R_a = jnp.sqrt(jnp.sum((g_test @ self.Z_inv) * g_test, axis=-1))

                # Standard pessimism, no Lipschitz scaling or risk shifting
                lcbs.append((mu_a - self.hparams.beta * R_a).reshape(-1, 1))

            batch_LCBs = jnp.hstack(lcbs)
            all_actions.append(jnp.argmax(batch_LCBs, axis=1))
            
            if self.hparams.verbose:
                pbar.update(B)

        if self.hparams.verbose:
            pbar.close()

        return jnp.concatenate(all_actions, axis=0)

    def evaluate_offline_policy(self, contexts, pi_actions):
        """Evaluates the Global Marginal Risk of a specified policy."""
        assert self.historical_residuals is not None, "Call train_offline_batch() first."
        
        # 1. Marginal Return Distribution
        mu = self.nn.out(self.nn.params, contexts, pi_actions).ravel()
        # Risk evaluated exactly like RobustOfflineBatchNeuraLCB to allow fair comparison
        marginal_risk = jnp.mean(mu) + self.rho_residuals
        
        # 2. Uncertainty Penalty
        g_test = self.nn.grad_out(self.nn.params, contexts, pi_actions) / jnp.sqrt(self.nn.m)
        pointwise_R = jnp.sqrt(jnp.sum((g_test @ self.Z_inv) * g_test, axis=-1))
        
        mean_R_pi = jnp.mean(pointwise_R)
        lcb_marginal = marginal_risk - self.hparams.beta * mean_R_pi
        
        return {
            "marginal_risk": marginal_risk,
            "mean_R_pi": mean_R_pi,
            "lcb_marginal": lcb_marginal,
            "pointwise_R": pointwise_R
        }

    def save_model(self, path):
        import pickle
        data = {
            'nn_params': self.nn.params,
            'nn_opt_state': self.nn.opt_state,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        if self.hparams.verbose:
            print(f'[{self.name}] Model saved to {path}')

    def load_model(self, path):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.nn.params = data['nn_params']
        self.nn.opt_state = data['nn_opt_state']
        if self.hparams.verbose:
            print(f'[{self.name}] Model loaded from {path}')

    def update_buffer(self, contexts, actions, rewards):
        self.data.add(contexts, actions.reshape(-1, 1), rewards.reshape(-1, 1))

    def update(self, contexts, actions, rewards):
        pass

    def monitor(self, contexts=None, actions=None, rewards=None):
        if contexts is not None and actions is not None and rewards is not None:
            f = self.nn.out(self.nn.params, contexts, actions.ravel()).ravel()
            mse = float(jnp.mean(jnp.square(f - rewards.ravel())))
            print(f'[{self.name}] MSE={mse:.4f}')
        else:
            print(f'[{self.name}] Model ready.')

