"""Risk-Aware Linear LCB offline bandit (RiskLinLCB)."""

import jax 
import jax.numpy as jnp
import numpy as np 
from jax.scipy.linalg import cho_factor, cho_solve
from core.bandit_algorithm import BanditAlgorithm 
import math

class RiskLinLCB(BanditAlgorithm):
    """
    Risk-Aware Linear LCB with Tofu Loss (reward clipping) and rho-LR risk bound.
    Implements: argmax { (mu_a + rho_residuals) - beta * L_rho * R_a }
    """
    def __init__(self, hparams, update_freq=1, name='RiskLinLCB'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq
        self.reset()

    def _compute_risk_functional(self, Y):
        """Computes the risk functional (e.g. CVaR) on the residuals."""
        measure = getattr(self.hparams, 'risk_measure', 'cvar')
        if measure == 'mean':
            return jnp.mean(Y)
        elif measure == 'cvar':
            alpha = self.hparams.alpha
            N = len(Y)
            k = jnp.maximum(1, int(alpha * N))
            Y_sorted = jnp.sort(Y)
            return jnp.mean(Y_sorted[:k])
        return jnp.mean(Y)

    def _get_risk_lipschitz_factor(self):
        """Lipschitz constant L_rho of the risk measure."""
        measure = getattr(self.hparams, 'risk_measure', 'cvar')
        if measure == 'cvar':
            tail_prob = self.hparams.alpha if self.hparams.alpha < 0.5 else (1.0 - self.hparams.alpha)
            return 1.0 / tail_prob
        return 1.0

    def reset(self, seed=None):
        p = self.hparams.context_dim * self.hparams.num_actions
        self.Sigma_hat = self.hparams.lambd0 * jnp.eye(p)
        self.y_hat = jnp.zeros((p, 1))
        self.theta_hat = None
        self.rho_residuals = None

    def train_offline_batch(self, contexts, actions, rewards):
        """Train Linear LCB on offline batch using Tofu loss and estimate risk."""
        # 1. Truncation / Tofu Loss
        truncation_mode = getattr(self.hparams, 'truncation_mode', 'none')
        if truncation_mode == 'clip':
            tau_n = getattr(self.hparams, 'tau_n', 1.0)
            r_tilde = jnp.where(jnp.abs(rewards) <= tau_n, rewards, tau_n * jnp.sign(rewards))
        else:
            r_tilde = rewards
        
        # 2. Build feature matrix phi (N, d*K)
        # Row-wise kronecker product of context and one-hot action
        num_train = contexts.shape[0]
        p = self.hparams.context_dim * self.hparams.num_actions
        eye_actions = jax.nn.one_hot(actions.ravel(), self.hparams.num_actions)
        
        # Vectorized row-wise kronecker
        phi = jax.vmap(lambda c, a: jnp.kron(c, a))(contexts, eye_actions)
        
        # 3. Solve Linear Regression (Ridge)
        self.Sigma_hat = self.hparams.lambd0 * jnp.eye(p) + phi.T @ phi
        self.y_hat = phi.T @ r_tilde.reshape(-1, 1)
        
        c, low = cho_factor(self.Sigma_hat)
        self.theta_hat = cho_solve((c, low), self.y_hat)
        
        # 4. Residual Risk Estimation on RAW rewards (Translation Invariance)
        predictions = (phi @ self.theta_hat).ravel()
        historical_residuals = rewards.ravel() - predictions
        self.rho_residuals = self._compute_risk_functional(historical_residuals)

    def sample_action(self, contexts):
        """Select actions using Risk-Aware LCB."""
        assert self.theta_hat is not None, "Call train_offline_batch() first."
        
        num_contexts = contexts.shape[0]
        num_actions = self.hparams.num_actions
        beta = self.hparams.beta
        L_rho = self._get_risk_lipschitz_factor()
        
        c, low = cho_factor(self.Sigma_hat)
        
        all_actions = []
        chunk_size = getattr(self.hparams, 'chunk_size', 500)
        
        for i in range(0, num_contexts, chunk_size):
            batch_ctx = contexts[i : i + chunk_size]
            B = batch_ctx.shape[0]
            lcbs = []
            
            for a in range(num_actions):
                # Build phi for this action
                eye_a = jax.nn.one_hot(jnp.ones(B, dtype=int) * a, num_actions)
                phi_a = jax.vmap(lambda ctx, act: jnp.kron(ctx, act))(batch_ctx, eye_a)
                
                # Mean Prediction
                mu_a = (phi_a @ self.theta_hat).ravel()
                
                # Risk Translation
                risk_a = mu_a + self.rho_residuals
                
                # Uncertainty Penalty R_a = sqrt(phi_a^T Sigma^-1 phi_a)
                x = cho_solve((c, low), phi_a.T) # (p, B)
                R_a = jnp.sqrt(jnp.sum(phi_a * x.T, axis=1))
                
                # Final Policy Selection
                policy_type = getattr(self.hparams, 'policy_type', 'risk-aware')
                if policy_type == 'standard-lcb':
                    lcb_a = mu_a - beta * R_a
                else:
                    # Risk-Aware LCB (Default)
                    lcb_a = risk_a - beta * L_rho * R_a
                
                lcbs.append(lcb_a.reshape(-1, 1))
                
            batch_LCBs = jnp.hstack(lcbs)
            all_actions.append(jnp.argmax(batch_LCBs, axis=1))
            
        return jnp.concatenate(all_actions, axis=0)
