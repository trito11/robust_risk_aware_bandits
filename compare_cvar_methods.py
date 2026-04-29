import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from absl import app, flags

from data.robust_synthetic_data import RobustSyntheticData
from algorithms.neural_offline_bandit import RobustOfflineBatchNeuraLCB

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_contexts', 2000, 'Number of training contexts')
flags.DEFINE_integer('num_test', 50, 'Number of test contexts for MILP (keep small)')
flags.DEFINE_integer('context_dim', 5, 'Context dimension')
flags.DEFINE_integer('num_actions', 10, 'Number of actions')
flags.DEFINE_float('alpha', 0.1, 'CVaR tail probability')

# Giả lập class hparams để nạp vào thuật toán
class HParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def compute_gt_marginal_cvar(test_mean, actions, true_noise, alpha):
    """Computes exact Marginal CVaR using I * M combinations."""
    sel_vals = test_mean[np.arange(test_mean.shape[0]), actions.ravel()]
    # Broadcast: (I, 1) + (1, M) -> Matrix (I, M)
    all_combinations = sel_vals.reshape(-1, 1) + true_noise.reshape(1, -1)
    all_combinations = all_combinations.ravel()
    
    sorted_vals = np.sort(all_combinations)
    tail_prob = alpha if alpha < 0.5 else (1.0 - alpha)
    k = max(1, int(tail_prob * len(sorted_vals)))
    return np.mean(sorted_vals[:k])
import gurobipy as gp
from gurobipy import GRB

def get_oracle_optimal_actions(test_mean, true_noise, alpha):
    I, K = test_mean.shape
    M = len(true_noise)
    
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model("Oracle_CVaR_MILP", env=env)
    
    # Khai báo biến
    x = model.addMVar((I, K), vtype=GRB.BINARY, name="x")
    var_nu = model.addVar(lb=-GRB.INFINITY, name="nu")
    z = model.addMVar((I, M), lb=0.0, name="z")
    
    # Ràng buộc: Mỗi context chọn đúng 1 action
    model.addConstr(x.sum(axis=1) == 1, name="OneAction")
    
    # Ràng buộc tính CVaR trên giá trị THẬT (test_mean + true_noise)
    for i in range(I):
        for m in range(M):
            # Chọn test_mean[i, k] thật sự
            chosen_mean = gp.quicksum(x[i, k] * test_mean[i, k] for k in range(K))
            actual_reward = chosen_mean + true_noise[m]
            model.addConstr(z[i, m] >= var_nu - actual_reward)
            
    # Hàm mục tiêu: Maximize CVaR
    tail_prob = alpha if alpha < 0.5 else (1.0 - alpha)
    cvar_expr = var_nu - (1.0 / (tail_prob * I * M)) * z.sum()
    model.setObjective(cvar_expr, GRB.MAXIMIZE)
    
    model.optimize()
    
    if model.Status == GRB.OPTIMAL:
        return np.argmax(x.X, axis=1)
    return np.zeros(I, dtype=int)

def main(_):
    hparams = HParams(
        context_dim=FLAGS.context_dim,
        num_actions=FLAGS.num_actions,
        layer_sizes=[64, 64],
        lr=1e-3,
        num_steps=200,
        buffer_s=FLAGS.num_contexts + 10,
        lambd0=0.1,
        m=64, # neural net width
        beta=0.1,
        alpha=FLAGS.alpha,
        tau_n=10.0,
        risk_measure='cvar',
        verbose=True,
        chunk_size=500,
        seed=42,
        s_init=1.0,
        debug_mode='none',
        layer_n=False,
        activation=jax.nn.relu,
        batch_size=2000,
        data_rand=True,
        lambd=0.1,
        freq_summary=1000
    )

    print("1. Generating Synthetic Data...")
    data = RobustSyntheticData(
        num_contexts=FLAGS.num_contexts,
        num_test_contexts=FLAGS.num_test,
        context_dim=FLAGS.context_dim,
        num_actions=FLAGS.num_actions,
        function_type='cosine',
        noise_type='student-t'
    )
    contexts, actions, full_rewards, test_ctx, test_mean = data.reset_data(0)
    
    # Lấy rewards cho các actions đã chọn
    rewards = full_rewards[np.arange(full_rewards.shape[0]), actions.ravel()]

    print("\n2. Training RobustOfflineBatchNeuraLCB...")
    algo = RobustOfflineBatchNeuraLCB(hparams, name='RobustAgent')
    algo.reset(42)
    algo.train_offline_batch(contexts, actions, rewards)

    print("\n3. Extracting Test Contexts and Generating True Noise...")
    # Lấy tập test nhỏ (I = 50) để giải MILP không bị chậm
    test_ctx = test_ctx[:FLAGS.num_test]
    test_mean = test_mean[:FLAGS.num_test]
    
    # Sinh mẫu nhiễu thật (M = 100)
    M_samples = 500
    true_noise = data.generate_noise((M_samples,))

    print("\n4. Action Selection: LOCAL (Point-wise argmax)")
    t0 = time.time()
    local_actions = algo.sample_action(test_ctx)
    local_time = time.time() - t0
    
    print("\n5. Action Selection: GLOBAL (Marginal MILP)")
    t0 = time.time()
    global_actions = algo.sample_action_milp(test_ctx, noise_samples=true_noise)
    global_time = time.time() - t0

    print("\n=======================================================")
    print("                 COMPARISON RESULTS                    ")
    print("=======================================================")
    
    # Đếm số hành động khác biệt giữa 2 phương pháp
    local_actions = np.array(local_actions)
    global_actions = np.array(global_actions)
    diff_count = np.sum(local_actions != global_actions)
    print(f"-> Policy Difference: {diff_count} / {FLAGS.num_test} actions are different.")
    
    # Tính Ground Truth Marginal CVaR của 2 policy (dùng chung tập I * M để công bằng)
    local_cvar = compute_gt_marginal_cvar(test_mean, local_actions, true_noise, FLAGS.alpha)
    global_cvar = compute_gt_marginal_cvar(test_mean, global_actions, true_noise, FLAGS.alpha)
    
    print(f"-> Local (Point-wise) GT CVaR  : {local_cvar:.4f}  (Time: {local_time:.3f}s)")
    print(f"-> Global (MILP) GT CVaR       : {global_cvar:.4f}  (Time: {global_time:.3f}s)")
    
    if global_cvar > local_cvar:
        print("\n=> KẾT LUẬN: Thuật toán MILP (Global) đã tìm ra tổ hợp hành động an toàn hơn (CVaR cao hơn) so với thuật toán Argmax (Local)!")
    elif global_cvar == local_cvar:
        print("\n=> KẾT LUẬN: Hai thuật toán cho ra Marginal CVaR tương đương nhau (MILP xác nhận Argmax đã tối ưu).")
    else:
        print("\n=> KẾT LUẬN: Point-wise ngẫu nhiên cho kết quả nhỉnh hơn (Hãy thử tăng số lượng Contexts hoặc điều chỉnh Beta).")
    # Tính Oracle Optimal Policy
    oracle_actions = get_oracle_optimal_actions(test_mean, true_noise, FLAGS.alpha)
    opt_cvar = compute_gt_marginal_cvar(test_mean, oracle_actions, true_noise, FLAGS.alpha)

    # Tính Suboptimality
    local_subopt = opt_cvar - local_cvar
    global_subopt = opt_cvar - global_cvar

    print(f"-> Oracle Optimal GT CVaR  : {opt_cvar:.4f}")
    print(f"   => Local Suboptimality  : {local_subopt:.4f}")
    print(f"   => Global Suboptimality : {global_subopt:.4f}")

if __name__ == '__main__':
    app.run(main)
