"""
Pessimistic Risk-Aware Policy Learning in Contextual Bandits
Paper: arXiv:2605.15620 – Wan, Li, Wu (2026)

Algorithm:
    pi_tilde = argmax_{a in {0..K-1}} rho_hat(pi_a) - L * R(pi_a)

where:
    rho_hat = risk functional (CVaR, mean-variance, entropic risk) applied
              to the empirical CDF estimate F_hat^pi.
    R(pi)   = data-dependent confidence bound on ||F_hat^pi - F^pi||_inf
              (Theorems 2–4 in paper).
    L       = Lipschitz constant of rho w.r.t. sup-norm on CDFs.

Three CDF estimators are supported (see Sec 3 of the paper):
    - 'IS'  : clipped importance sampling   (Eq 5)
    - 'WIS' : weighted importance sampling  (Eq 6)
    - 'DR'  : doubly robust                 (Eq 8)

Key formulas (from opl_arxiv.tex):
    w_pi(x, a) = 1{a = pi(x)} / beta(x, pi(x))      importance weight

    I_pi = {i : beta(X_i, pi(X_i)) > 0}              informative samples

    sigma_pi = sqrt(1/n * sum_{i in I_pi} 1/beta(X_i,pi)^2)    Eq (def-sigma)
    r_pi     = (n - |I_pi|) / n                                  Eq (def-r)

    R_IS(pi) = (sigma_pi + 2) * sqrt(8/n * [log(20/delta) + d_Pi * log(nK^2)])
               + r_pi                                             Thm 2

    Policy selection (Eq def-pessimism):
        pi_tilde = argmax_a  rho_hat(pi_a) - L * R_IS(pi_a)
"""

import numpy as np
from core.bandit_algorithm import BanditAlgorithm


# ─────────────────────────────────────────────────────────────────────────────
# Risk functional helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cvar(rewards: np.ndarray, alpha: float) -> float:
    """CVaR at level alpha (lower alpha = more risk-averse)."""
    sorted_r = np.sort(rewards)
    k = max(1, int(np.ceil(alpha * len(sorted_r))))
    return float(np.mean(sorted_r[:k]))


def _weighted_cvar(rewards: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """Compute CVaR at level alpha of samples with non-negative weights."""
    if len(rewards) == 0:
        return 0.0
    w = np.array(weights, dtype=float)
    w_sum = np.sum(w)
    if w_sum <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w_sum

    order = np.argsort(rewards)
    sorted_r = rewards[order]
    sorted_w = w[order]

    cum_w = np.cumsum(sorted_w)
    cutoff_mask = (cum_w <= alpha)
    if not np.any(cutoff_mask):
        return float(sorted_r[0])

    idx = np.where(cutoff_mask)[0][-1]
    selected_r = sorted_r[: idx + 1]
    selected_w = sorted_w[: idx + 1]
    denom = np.sum(selected_w)
    return float(np.sum(selected_r * selected_w) / denom) if denom > 0 else float(sorted_r[0])


def _mean_variance(rewards: np.ndarray, lam: float) -> float:
    """Mean – lambda * Variance."""
    return float(np.mean(rewards) - lam * np.var(rewards))


def _entropic_risk(rewards: np.ndarray, theta: float) -> float:
    """Entropic risk: -1/theta * log E[exp(-theta * Y)]."""
    # clamp to avoid overflow
    vals = np.clip(-theta * rewards, -500, 500)
    return float(-(1.0 / theta) * np.log(np.mean(np.exp(vals)) + 1e-12))


def _apply_risk(rewards: np.ndarray, risk_measure: str, alpha: float = 0.05,
                lam: float = 0.1, theta: float = 1.0) -> float:
    """Dispatch risk functional."""
    if risk_measure == 'cvar':
        return _cvar(rewards, alpha)
    elif risk_measure == 'mean_variance':
        return _mean_variance(rewards, lam)
    elif risk_measure == 'entropic':
        return _entropic_risk(rewards, theta)
    else:  # 'mean'
        return float(np.mean(rewards))


def _lipschitz_constant(risk_measure: str, alpha: float = 0.05,
                        lam: float = 0.1, y_max: float = 1.0) -> float:
    """
    Lipschitz constant L of rho w.r.t. sup-norm on CDFs (||F||_inf).
    Paper: arXiv:2605.15620 (Wan, Li, Wu 2026), Section 2 & Appendix A.3:
      - For CVaR_alpha: L = D / (1 - alpha)
      - For mean: L = D
      - For mean-variance: L = D + 3 * lam * D^2
    where D = y_max is the bounded range of rewards: max(Y) - min(Y).
    """
    D = max(float(y_max), 1e-4)
    if risk_measure == 'cvar':
        denom = max(1.0 - alpha, 1e-4)
        return float(D / denom)
    elif risk_measure == 'mean_variance':
        return float(D + 3.0 * abs(lam) * (D ** 2))
    elif risk_measure == 'entropic':
        return float(D)
    else:  # 'mean'
        return float(D)


# ─────────────────────────────────────────────────────────────────────────────
# CDF estimators
# ─────────────────────────────────────────────────────────────────────────────

def _is_cdf_estimate(rewards: np.ndarray, is_weights: np.ndarray,
                     in_support: np.ndarray, n: int) -> np.ndarray:
    """
    Clipped IS CDF estimator (Eq def-isc in paper):
        F_hat_IS(t) = (1/n) * sum_i G_hat(t | X_i, pi)
    where G_hat = w_pi * 1{Y <= t}  if i in I_pi
                = 1                  if i not in I_pi   (pessimistic fill)
    Clipped: F_hat_ISc = min(F_hat_IS, 1).

    Returns weighted indicator values (one per sample), NOT the CDF at a grid,
    since we only need rho(F_hat^pi) which depends on the pseudo-rewards:
        pseudo_r_i = w_i * Y_i   (for i in I_pi),  or a pessimistic fill value.

    For computing rho via plug-in on pseudo-rewards:
        rho(F_hat^pi) ≈ rho({pseudo_r_i})
    This is an approximation valid when rho is translation-equivariant (CVaR, mean, ...).

    NOTE: The paper's plug-in is:  rho_hat = rho(F_hat^pi)
    For CVaR_alpha this equals the alpha-quantile of the pseudo-rewards.
    """
    # IS pseudo-rewards
    pseudo_r = np.where(in_support, np.clip(is_weights * rewards, -np.inf, rewards.max() * 10),
                        rewards.max())  # pessimistic fill = max for non-supported
    return pseudo_r


def _wis_cdf_estimate(rewards: np.ndarray, is_weights: np.ndarray,
                      in_support: np.ndarray, n: int) -> np.ndarray:
    """
    WIS CDF estimator (Eq def-wis):
        W_pi = (1/|I_pi|) * sum_{i in I_pi} w_pi(X_i, A_i)
        pseudo_r_i = (w_i / W_pi) * Y_i   if i in I_pi
                   = Y_max (pessimistic)    if i not in I_pi
    """
    in_sup_mask = in_support.astype(bool)
    if in_sup_mask.sum() == 0:
        return np.full(n, rewards.max())
    W_pi = np.mean(is_weights[in_sup_mask])
    W_pi = max(W_pi, 1e-8)
    normalised_w = is_weights / W_pi
    pseudo_r = np.where(in_support, normalised_w * rewards, rewards.max())
    return pseudo_r


def _dr_cdf_estimate(rewards: np.ndarray, is_weights: np.ndarray,
                     in_support: np.ndarray, model_pred: np.ndarray) -> np.ndarray:
    """
    DR CDF estimator (Eq def-dr):
        pseudo_r_i = G_bar(·|X_i,pi) + w_i * [1{Y<=t} - G_bar(·|X_i,A_i)]   if i in I_pi
                   = G_bar(·|X_i,pi)                                            if not
    For scalar pseudo-reward approximation:
        pseudo_r_i ≈ model_pred_pi_i + w_i * (Y_i - model_pred_ai_i)   if i in I_pi
                   = model_pred_pi_i                                      otherwise
    """
    pseudo_r = np.where(
        in_support,
        model_pred + is_weights * (rewards - model_pred),
        model_pred
    )
    return pseudo_r


# ─────────────────────────────────────────────────────────────────────────────
# Confidence bound R(pi)  (Theorem 2 in paper)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_R_IS(beta_vals: np.ndarray, in_support: np.ndarray, n: int,
                  n_actions: int, delta: float = 0.05, d_Pi: int = 1) -> float:
    """
    IS confidence bound (Theorem 2):
        sigma_pi = sqrt(1/n * sum_{i in I_pi} 1/beta(X_i,pi)^2)
        r_pi     = (n - |I_pi|) / n
        R(pi)    = (sigma_pi + 2) * sqrt(8/n * [log(20/delta) + d_Pi * log(n*K^2)])
                   + r_pi
    Since R(pi) bounds the CDF sup-norm ||F_hat - F||_inf, it is strictly capped at 1.0.
    """
    in_sup_mask = in_support.astype(bool)
    n_inf = in_sup_mask.sum()
    r_pi = (n - n_inf) / n

    if n_inf == 0:
        return 1.0

    beta_inf = beta_vals[in_sup_mask]
    sigma_pi = np.sqrt(np.mean(1.0 / (beta_inf ** 2 + 1e-12)))

    log_term = np.log(20.0 / delta) + d_Pi * np.log(n * (n_actions ** 2) + 1)
    concentration = (sigma_pi + 2.0) * np.sqrt(8.0 / n * log_term)
    return float(np.clip(concentration + r_pi, 0.0, 1.0))


def _compute_R_WIS(beta_vals: np.ndarray, in_support: np.ndarray, n: int,
                   n_actions: int, delta: float = 0.05, d_Pi: int = 1) -> float:
    """
    WIS confidence bound (Theorem 3):
        eta_pi = sigma_pi * sqrt(n / (2*|I_pi|^2) * [log(8/delta) + d_Pi*log(nK^2)])
        if eta_pi >= 1: R = 1  (conservative)
        else: R = xi_pi = (sigma_pi/(1-eta_pi) + 2)*sqrt(...) + |I_pi|/n*eta_pi/(1-eta_pi) + r_pi
    """
    in_sup_mask = in_support.astype(bool)
    n_inf = int(in_sup_mask.sum())
    r_pi = (n - n_inf) / n

    if n_inf == 0:
        return 1.0

    beta_inf = beta_vals[in_sup_mask]
    sigma_pi = np.sqrt(np.mean(1.0 / (beta_inf ** 2 + 1e-12)))

    log_term = np.log(8.0 / delta) + d_Pi * np.log(n * (n_actions ** 2) + 1)
    eta_pi = sigma_pi * np.sqrt(n / (2.0 * (n_inf ** 2) + 1e-12) * log_term)

    if eta_pi >= 1.0:
        return 1.0

    log_term2 = np.log(20.0 / delta) + d_Pi * np.log(n * (n_actions ** 2) + 1)
    xi = ((sigma_pi / (1.0 - eta_pi) + 2.0) * np.sqrt(8.0 / n * log_term2)
          + (n_inf / n) * eta_pi / (1.0 - eta_pi) + r_pi)
    return float(np.clip(xi, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Main Algorithm Class
# ─────────────────────────────────────────────────────────────────────────────

class PessimisticCDFBandit(BanditAlgorithm):
    """
    Pessimistic Risk-Aware Policy Learning via CDF Estimation.

    Reference: Wan, Li, Wu (2026) arXiv:2605.15620
    "Pessimistic Risk-Aware Policy Learning in Contextual Bandits"

    Algorithm (Eq def-pessimism):
        pi_tilde = argmax_{a in A}  rho_hat(pi_a) - L * R(pi_a)

    Parameters (in hparams):
        risk_measure   : 'cvar' | 'mean' | 'mean_variance' | 'entropic'
        alpha          : CVaR tail probability (default 0.05)
        variance_lambda: lambda for mean-variance (default 0.1)
        entropic_theta : theta for entropic risk (default 1.0)
        beta           : scalar weight on uncertainty term  (scales L*R)
        cdf_estimator  : 'IS' | 'WIS' | 'DR' (default 'IS')
        delta          : confidence parameter for R(pi) (default 0.05)
        d_Pi           : VC/covering dimension of policy class (default 1 for greedy)
        behavior_policy_est: 'empirical' | 'uniform'
            'empirical' – estimate beta(x,a) = count(a,x-bucket)/n per action
            'uniform'   – beta(x,a) = n_a/n (action marginal)
    """

    def __init__(self, hparams, update_freq=1, name='PessimisticCDF'):
        self.name = name
        self.hparams = hparams
        self.update_freq = update_freq
        self._trained = False
        # Storage filled by train_offline_batch
        self._contexts = None
        self._actions = None
        self._rewards = None
        self._beta_hat = None    # estimated behavior policy P(A|X) per sample

    # ------------------------------------------------------------------
    def reset(self, seed=None):
        self._trained = False
        self._contexts = None
        self._actions = None
        self._rewards = None
        self._beta_hat = None
        if seed is not None:
            np.random.seed(seed)

    # ------------------------------------------------------------------
    def _estimate_behavior_policy(self, actions: np.ndarray, n: int, K: int) -> np.ndarray:
        """
        Estimate beta(X_i, A_i) for each sample.

        Strategy: 'uniform' (marginal)  beta_hat_i = count(A=A_i) / n
        This is a simple but widely used approximation when we only have
        one sample per (X_i, A_i) pair, as in standard offline bandit data.
        """
        strategy = getattr(self.hparams, 'behavior_policy_est', 'uniform')
        if strategy == 'uniform':
            counts = np.bincount(actions.ravel(), minlength=K).astype(float)
            freq = counts / n  # (K,)
            freq = np.maximum(freq, 1e-8)  # avoid div/0
            beta_per_sample = freq[actions.ravel()]   # (n,)
        else:
            # fallback
            beta_per_sample = np.full(n, 1.0 / K)
        return beta_per_sample

    # ------------------------------------------------------------------
    def train_offline_batch(self, contexts: np.ndarray, actions: np.ndarray,
                            rewards: np.ndarray):
        """
        Store offline dataset and estimate behavior policy.

        Args:
            contexts : (n, d)
            actions  : (n,)  integer action indices in {0, ..., K-1}
            rewards  : (n,)  scalar rewards
        """
        n = contexts.shape[0]
        K = self.hparams.num_actions
        actions = actions.ravel().astype(int)
        rewards = rewards.ravel().astype(float)

        self._contexts = contexts
        self._actions = actions
        self._rewards = rewards
        self._n = n
        self._K = K

        # Estimate behavior policy beta(X_i, A_i)
        self._beta_hat = self._estimate_behavior_policy(actions, n, K)
        self._trained = True

    # ------------------------------------------------------------------
    def _compute_lcb_for_action(self, a: int) -> float:
        """
        Compute LCB for a deterministic policy pi_a(x) = a for all x.

        LCB(a) = rho_hat(pi_a) - beta * L * R(pi_a)

        Paper: arXiv:2605.15620 (Wan, Li, Wu 2026)
            rho_hat: plug-in estimate of risk functional from empirical CDF
            L      : Lipschitz constant = D / (1 - alpha)
            R      : data-dependent confidence bound on CDF error in [0, 1]
            beta   : pessimism tuning weight
        """
        n = self._n
        actions = self._actions
        rewards = self._rewards
        beta = self._beta_hat
        K = self._K

        # 1. Informative samples where A_i = a
        mask = (actions == a)
        n_a = int(mask.sum())
        in_support = mask.astype(float)

        y_min = float(np.min(rewards)) if len(rewards) > 0 else 0.0
        y_max = float(np.max(rewards)) if len(rewards) > 0 else 1.0
        y_range = max(y_max - y_min, 1e-4)

        beta_scale = getattr(self.hparams, 'beta', 0.1)
        if n_a == 0:
            return y_min - beta_scale * y_range

        rewards_a = rewards[mask]
        beta_a = beta[mask]

        # 2. Risk functional estimation
        risk_measure = getattr(self.hparams, 'risk_measure', 'cvar')
        alpha = getattr(self.hparams, 'alpha', 0.05)
        lam = getattr(self.hparams, 'variance_lambda', 0.1)
        theta = getattr(self.hparams, 'entropic_theta', 1.0)
        estimator = getattr(self.hparams, 'cdf_estimator', 'IS').upper()

        if estimator == 'WIS':
            w_a = 1.0 / np.maximum(beta_a, 1e-8)
            rho_hat = _weighted_cvar(rewards_a, w_a, alpha) if risk_measure == 'cvar' else _apply_risk(rewards_a, risk_measure, alpha=alpha, lam=lam, theta=theta)
            R = _compute_R_WIS(beta, in_support, n, K, delta=getattr(self.hparams, 'delta', 0.05), d_Pi=getattr(self.hparams, 'd_Pi', 1))
        elif estimator == 'DR':
            mean_a = float(np.mean(rewards_a))
            w_a = 1.0 / np.maximum(beta_a, 1e-8)
            norm_w = w_a / (np.mean(w_a) + 1e-12)
            dr_samples = mean_a + (rewards_a - mean_a) * np.clip(norm_w, 0.1, 10.0)
            rho_hat = _cvar(dr_samples, alpha) if risk_measure == 'cvar' else _apply_risk(dr_samples, risk_measure, alpha=alpha, lam=lam, theta=theta)
            R = _compute_R_WIS(beta, in_support, n, K, delta=getattr(self.hparams, 'delta', 0.05), d_Pi=getattr(self.hparams, 'd_Pi', 1))
        else:  # default: IS
            w_a = 1.0 / np.maximum(beta_a, 1e-8)
            rho_hat = _weighted_cvar(rewards_a, w_a, alpha) if risk_measure == 'cvar' else _apply_risk(rewards_a, risk_measure, alpha=alpha, lam=lam, theta=theta)
            R = _compute_R_IS(beta, in_support, n, K, delta=getattr(self.hparams, 'delta', 0.05), d_Pi=getattr(self.hparams, 'd_Pi', 1))

        # 3. Lipschitz constant L = D / (1 - alpha)
        L = _lipschitz_constant(risk_measure, alpha=alpha, lam=lam, y_max=y_range)

        # 4. Pessimistic LCB
        lcb = rho_hat - beta_scale * L * R
        return float(lcb)

    # ------------------------------------------------------------------
    def sample_action(self, contexts: np.ndarray) -> np.ndarray:
        """
        Select action for each test context using the pessimistic policy:
            pi_tilde(x) = argmax_a LCB(a)
        """
        assert self._trained, "Call train_offline_batch() first."
        K = self._K
        lcbs = np.array([self._compute_lcb_for_action(a) for a in range(K)])
        best_action = int(np.argmax(lcbs))
        n_test = contexts.shape[0]
        return np.full(n_test, best_action, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Context-aware variant: per-context CDF estimation
# ─────────────────────────────────────────────────────────────────────────────

class PessimisticCDFContextualBandit(PessimisticCDFBandit):
    """
    Context-aware extension: for each test context x, compute CDF estimate
    using kernel-weighted samples (soft version of I_pi local to x).

    pi_tilde(x) = argmax_a  rho_hat_x(pi_a) - beta * L * R_x(pi_a)

    where the CDF estimate uses local importance weights:
        w_i(x) = K_h(X_i, x) * 1{A_i=a} / beta_hat_i
    K_h = Gaussian kernel with bandwidth h = hparams.rbf_sigma.
    """

    def __init__(self, hparams, update_freq=1, name='PessimisticCDFContextual'):
        super().__init__(hparams, update_freq=update_freq, name=name)

    def _gaussian_kernel(self, X_train: np.ndarray, x: np.ndarray,
                         sigma: float) -> np.ndarray:
        """K_h(X_i, x) proportional to exp(-||X_i - x||^2 / (2*h^2)), normalized."""
        diffs = X_train - x[np.newaxis, :]          # (n, d)
        sq_dist = np.sum(diffs ** 2, axis=1)         # (n,)
        log_w = -sq_dist / (2.0 * (sigma ** 2))
        log_w -= log_w.max()                         # numerical stability
        w = np.exp(log_w)
        w /= (w.sum() + 1e-12)
        return w  # (n,)

    def _compute_lcb_for_action_at_x(self, a: int, x: np.ndarray) -> float:
        """Context-aware LCB for action a at context x."""
        n = self._n
        K = self._K
        sigma = getattr(self.hparams, 'rbf_sigma', 1.0)

        actions = self._actions
        rewards = self._rewards
        beta = self._beta_hat

        mask = (actions == a)
        n_a = int(mask.sum())
        in_support = mask.astype(float)

        y_min = float(np.min(rewards)) if len(rewards) > 0 else 0.0
        y_max = float(np.max(rewards)) if len(rewards) > 0 else 1.0
        y_range = max(y_max - y_min, 1e-4)

        beta_scale = getattr(self.hparams, 'beta', 0.1)
        if n_a == 0:
            return y_min - beta_scale * y_range

        # kernel weights for action a samples
        k_weights = self._gaussian_kernel(self._contexts, x, sigma)
        k_weights_a = k_weights[mask]
        beta_a = beta[mask]
        rewards_a = rewards[mask]

        # local IS weights: k_i * 1{A_i=a} / beta_i
        w_local = k_weights_a / np.maximum(beta_a, 1e-8)

        risk_measure = getattr(self.hparams, 'risk_measure', 'cvar')
        alpha = getattr(self.hparams, 'alpha', 0.05)
        lam = getattr(self.hparams, 'variance_lambda', 0.1)
        theta = getattr(self.hparams, 'entropic_theta', 1.0)
        estimator = getattr(self.hparams, 'cdf_estimator', 'IS').upper()

        if risk_measure == 'cvar':
            if estimator == 'DR':
                mean_local = float(np.sum(rewards_a * w_local) / (np.sum(w_local) + 1e-12))
                norm_w = w_local / (np.mean(w_local) + 1e-12)
                dr_local = mean_local + (rewards_a - mean_local) * np.clip(norm_w, 0.1, 10.0)
                rho_hat = _cvar(dr_local, alpha)
            else:
                rho_hat = _weighted_cvar(rewards_a, w_local, alpha)
        else:
            rho_hat = _apply_risk(rewards_a, risk_measure, alpha=alpha, lam=lam, theta=theta)

        # confidence bound
        delta = getattr(self.hparams, 'delta', 0.05)
        d_Pi = getattr(self.hparams, 'd_Pi', 1)
        if estimator == 'WIS':
            R = _compute_R_WIS(beta, in_support, n, K, delta=delta, d_Pi=d_Pi)
        else:
            R = _compute_R_IS(beta, in_support, n, K, delta=delta, d_Pi=d_Pi)

        L = _lipschitz_constant(risk_measure, alpha=alpha, lam=lam, y_max=y_range)
        lcb = rho_hat - beta_scale * L * R
        return float(lcb)

    def sample_action(self, contexts: np.ndarray) -> np.ndarray:
        assert self._trained, "Call train_offline_batch() first."
        K = self._K
        n_test = contexts.shape[0]
        chosen = np.zeros(n_test, dtype=np.int32)

        chunk = getattr(self.hparams, 'chunk_size', 500)
        for start in range(0, n_test, chunk):
            batch = contexts[start: start + chunk]
            for j, x in enumerate(batch):
                lcbs = [self._compute_lcb_for_action_at_x(a, x) for a in range(K)]
                chosen[start + j] = int(np.argmax(lcbs))

        return chosen
