"""Mini main for testing algorithms. """ 

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import numpy as np 
import jax  
from jax import config
config.update("jax_debug_nans", False)
import jax.numpy as jnp 
from easydict import EasyDict as edict
import time 

from core.contextual_bandit import contextual_bandit_runner
from algorithms.neural_offline_bandit import ExactNeuraLCBV2, NeuralGreedyV2, ApproxNeuraLCBV2, RobustOfflineBatchNeuraLCB, NeuralRegressionOffline, RiskExactNeuraLCBV2, QuantileRiskNeuralBandit
from algorithms.risk_lin_lcb import RiskLinLCB
from algorithms.lin_lcb import LinLCB 
from algorithms.kern_lcb import KernLCB 
from algorithms.uniform_sampling import UniformSampling
from algorithms.neural_lin_lcb import ExactNeuralLinLCBV2, ExactNeuralLinGreedyV2, ApproxNeuralLinLCBV2, ApproxNeuralLinGreedyV2, \
    ApproxNeuralLinLCBJointModel, NeuralLinGreedyJointModel
import wandb
from data.realworld_data import *
from data.robust_synthetic_data import RobustSyntheticData
from data.npz_data import SimglucoseData

from absl import flags, app


FLAGS = flags.FLAGS 

flags.DEFINE_string('data_type', 'mushroom', 'Dataset to sample from')
flags.DEFINE_string('policy', 'eps-greedy', 'Offline policy, eps-greedy/subset')
flags.DEFINE_float('eps', 0.1, 'Probability of selecting a random action in eps-greedy')
flags.DEFINE_float('subset_r', 0.5, 'The ratio of the action spaces to be selected in offline data')
flags.DEFINE_integer('num_contexts', 15000, 'Number of contexts for training.') 
flags.DEFINE_integer('num_test_contexts', 10000, 'Number of contexts for test.') 
flags.DEFINE_boolean('verbose', True, 'verbose') 
flags.DEFINE_boolean('debug', True, 'debug') 
flags.DEFINE_boolean('normalize', False, 'normalize the regret') 
flags.DEFINE_integer('update_freq', 1, 'Update frequency')
flags.DEFINE_integer('freq_summary', 10, 'Summary frequency')

flags.DEFINE_integer('test_freq', 10, 'Test frequency')
flags.DEFINE_string('algo_group', 'approx-neural', 'baseline/neural')
flags.DEFINE_integer('num_sim', 10, 'Number of simulations')
flags.DEFINE_float('noise_std', 0.1, 'Noise std')
flags.DEFINE_integer('context_dim', 20, 'Context dimension for synthetic data')
flags.DEFINE_integer('num_actions', 30, 'Number of actions for synthetic data')
flags.DEFINE_list('layer_sizes', ['100', '100'], 'Layer sizes for Neural Network')

flags.DEFINE_integer('chunk_size', 500, 'Chunk size')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('num_steps', 100, 'Number of steps to train NN.') 
flags.DEFINE_integer('buffer_s', -1, 'Size in the train data buffer.')
flags.DEFINE_bool('data_rand', True, 'Where randomly sample a data batch or  use the latest samples in the buffer' )

flags.DEFINE_float('rbf_sigma', 1, 'RBF sigma for KernLCB') # [0.1, 1, 10]

# Quantile Regression
flags.DEFINE_integer('num_quantiles', 100, 'Number of quantiles for Quantile Regression')
flags.DEFINE_float('huber_kappa', 1.0, 'Kappa parameter for Huber Loss')

# NeuraLCB 
flags.DEFINE_float('beta', 0.1, 'confidence paramter') # [0.01, 0.05, 0.1, 0.5, 1, 5, 10] 
flags.DEFINE_float('lr', 1e-3, 'learning rate') 
flags.DEFINE_float('lambd0', 0.1, 'minimum eigenvalue') 
flags.DEFINE_float('lambd', 1e-4, 'regularization parameter')

# RobustOfflineBatchNeuraLCB params
flags.DEFINE_string('risk_measure', 'cvar', 'Risk measure: mean/cvar/entropic/mean_variance')
flags.DEFINE_float('tau_n', 1.0, 'Truncation threshold for Tofu loss')
flags.DEFINE_float('alpha', 0.05, 'CVaR alpha level')
flags.DEFINE_float('entropic_theta', 1.0, 'Theta for entropic risk')
flags.DEFINE_float('variance_lambda', 0.1, 'Lambda for mean-variance')
flags.DEFINE_string('truncation_mode', 'clip', 'clip or mask')
flags.DEFINE_string('noise_type', 'student-t', 'student-t, gaussian, binary-heavy')
flags.DEFINE_string('function_type', 'quadratic', 'linear, quadratic, quadratic2, cosine')
flags.DEFINE_string('policy_type', 'risk-aware', 'risk-aware or standard-lcb')
flags.DEFINE_string('save_model_path', 'results/model.pkl', 'Path to save weights after training')

# Logging
flags.DEFINE_boolean('use_wandb', False, 'Whether to use wandb for logging')
flags.DEFINE_string('wandb_project', 'offline_neural_bandits', 'wandb project name')
flags.DEFINE_string('wandb_entity', None, 'wandb entity')

# Evaluation
flags.DEFINE_string('agent_eval_method', 'local', 'How agent selects actions: local (argmax) or global (milp)')
flags.DEFINE_string('oracle_eval_method', 'global', 'How oracle selects actions: local (argmax) or global (milp)')

#================================================================
# Network parameters
#================================================================
def main(unused_argv): 
    print("Starting experiment script...")

    #=================
    # Data 
    #=================
    if FLAGS.policy == 'eps-greedy':
        policy_prefix = '{}{}'.format(FLAGS.policy, FLAGS.eps)
    elif FLAGS.policy == 'subset':
        policy_prefix = '{}{}'.format(FLAGS.policy, FLAGS.subset_r)
    elif FLAGS.policy == 'online':
        policy_prefix = '{}{}'.format(FLAGS.policy, FLAGS.eps) 
    else:
        raise NotImplementedError('{} not implemented'.format(FLAGS.policy))

    dataclasses = {'mushroom':MushroomData, 'jester':JesterData, 'statlog':StatlogData, 'covertype':CoverTypeData, 'stock': StockData,
            'adult': AdultData, 'census': CensusData, 'mnist': MnistData
    }
    
    if FLAGS.data_type in dataclasses:
        DataClass = dataclasses[FLAGS.data_type]
        data = DataClass(num_contexts=FLAGS.num_contexts, 
                    num_test_contexts=FLAGS.num_test_contexts,
                    pi = FLAGS.policy, 
                    eps = FLAGS.eps, 
                    subset_r = FLAGS.subset_r) 
    elif FLAGS.data_type == 'robust_syn':
        data = RobustSyntheticData(
            num_contexts=FLAGS.num_contexts,
            num_test_contexts=FLAGS.num_test_contexts,
            context_dim=FLAGS.context_dim,
            num_actions=FLAGS.num_actions,
            function_type=FLAGS.function_type,
            noise_type=FLAGS.noise_type,
            pi=FLAGS.policy,
            eps=FLAGS.eps
        )
    elif FLAGS.data_type == 'simglucose':
        data = SimglucoseData(
            path='data/simglucose_offline.npz',
            num_contexts=FLAGS.num_contexts
        )
    else:
        raise NotImplementedError

    if FLAGS.data_type == 'mnist': # Use 1000 test points for mnist 
        FLAGS.num_test_contexts = 1000  
        FLAGS.test_freq = 100
        FLAGS.chunk_size = 1
    dataset = data.reset_data()
    context_dim = dataset[0].shape[1] 
    num_actions = data.num_actions 
    
    # Process layer_sizes flag: convert list of strings to list of ints
    layer_sizes = [int(s) for s in FLAGS.layer_sizes]
    
    hparams = edict({
        'layer_sizes': layer_sizes, 
        's_init': 1, 
        'activation': jax.nn.relu, 
        'layer_n': True,
        'seed': 0,
        'context_dim': context_dim, 
        'num_actions': num_actions, 
        'beta': FLAGS.beta, # [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        'lambd': FLAGS.lambd, # regularization param: [0.1m, m, 10 m  ]
        'lr': FLAGS.lr, 
        'lambd0': FLAGS.lambd0, # shoud be lambd/m in theory but we fix this at 0.1 for simplicity and mainly focus on tuning beta 
        'verbose': FLAGS.verbose, 
        'batch_size': FLAGS.batch_size,
        'freq_summary': FLAGS.freq_summary, 
        'chunk_size': FLAGS.chunk_size, 
        'num_steps': FLAGS.num_steps, 
        'buffer_s': FLAGS.buffer_s, 
        'data_rand': FLAGS.data_rand,
        'debug_mode': 'full', # simple/full
        'risk_measure': FLAGS.risk_measure,
        'tau_n': FLAGS.tau_n,
        'alpha': FLAGS.alpha,
        'entropic_theta': FLAGS.entropic_theta,
        'variance_lambda': FLAGS.variance_lambda,
        'truncation_mode': FLAGS.truncation_mode,
        'policy_type': FLAGS.policy_type
    })

    lin_hparams = edict(
        {
            'context_dim': hparams.context_dim, 
            'num_actions': hparams.num_actions, 
            'lambd0': hparams.lambd0, 
            'beta': hparams.beta, 
            'rbf_sigma': FLAGS.rbf_sigma, # 0.1, 1, 10
            'max_num_sample': 1000,
            'risk_measure': hparams.risk_measure,
            'tau_n': hparams.tau_n,
            'alpha': hparams.alpha,
            'entropic_theta': hparams.entropic_theta,
            'variance_lambda': hparams.variance_lambda,
            'chunk_size': hparams.chunk_size,
            'policy_type': hparams.policy_type
        }
    )

    data_prefix = '{}_d={}_a={}_pi={}_std={}'.format(FLAGS.data_type, \
            context_dim, num_actions, policy_prefix, data.noise_std if hasattr(data, 'noise_std') else 'N/A')

    res_dir = os.path.join('results', data_prefix) 

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    #================================================================
    # Algorithms 
    #================================================================

    if FLAGS.algo_group == 'approx-neural':
        algos = [
                UniformSampling(lin_hparams),
                ApproxNeuraLCBV2(hparams, update_freq = FLAGS.update_freq)
            ]

        algo_prefix = 'approx-neural-gridsearch_epochs={}_m={}_layern={}_buffer={}_bs={}_lr={}_beta={}_lambda={}_lambda0={}'.format(
            hparams.num_steps, min(hparams.layer_sizes), hparams.layer_n, hparams.buffer_s, hparams.batch_size, hparams.lr, \
            hparams.beta, hparams.lambd, hparams.lambd0
        )

    eval_m = FLAGS.agent_eval_method
    oracle_m = FLAGS.oracle_eval_method
    if FLAGS.algo_group == 'robust-offline':
        algos = [
            RobustOfflineBatchNeuraLCB(hparams)
        ]
        layer_str = "-".join([str(s) for s in layer_sizes])
        algo_prefix = 'robust_{}_agent={}_oracle={}_risk={}_alpha={}_beta={}_n={}_layers={}'.format(
            FLAGS.data_type, eval_m, oracle_m, FLAGS.risk_measure, FLAGS.alpha, FLAGS.beta, FLAGS.num_contexts, layer_str
        )

    if FLAGS.algo_group == 'neural-regression':
        # Compare pure neural regression (no pessimism/risk) vs risk-aware LCB
        layer_str = "-".join([str(s) for s in layer_sizes])
        algos = [
            NeuralRegressionOffline(hparams),
            RobustOfflineBatchNeuraLCB(hparams),
        ]
        algo_prefix = 'neural_regression_{}_agent={}_oracle={}_risk={}_alpha={}_beta={}_n={}_layers={}'.format(
            FLAGS.data_type, eval_m, oracle_m, FLAGS.risk_measure, FLAGS.alpha, FLAGS.beta, FLAGS.num_contexts, layer_str
        )

    if FLAGS.algo_group == 'risk-exact':
        algos = [
            RiskExactNeuraLCBV2(hparams)
        ]
        layer_str = "-".join([str(s) for s in layer_sizes])
        algo_prefix = 'risk_exact_{}_agent={}_oracle={}_risk={}_alpha={}_beta={}_n={}_layers={}'.format(
            FLAGS.data_type, eval_m, oracle_m, FLAGS.risk_measure, FLAGS.alpha, FLAGS.beta, FLAGS.num_contexts, layer_str
        )

    if FLAGS.algo_group == 'quantile-risk':
        algos = [
            QuantileRiskNeuralBandit(hparams)
        ]
        layer_str = "-".join([str(s) for s in layer_sizes])
        algo_prefix = 'quantile_risk_{}_agent={}_oracle={}_alpha={}_beta={}_n={}_layers={}'.format(
            FLAGS.data_type, eval_m, oracle_m, FLAGS.alpha, FLAGS.beta, FLAGS.num_contexts, layer_str
        )

    if FLAGS.algo_group == 'risk-lin-lcb':
        algos = [
            RiskLinLCB(lin_hparams)
        ]
        algo_prefix = 'risk_lin_lcb_{}_agent={}_oracle={}_risk={}_alpha={}_beta={}_n={}'.format(
            FLAGS.data_type, eval_m, oracle_m, FLAGS.risk_measure, FLAGS.alpha, FLAGS.beta, FLAGS.num_contexts
        )

    #==============================
    # W&B Init
    #==============================
    if FLAGS.use_wandb is True:
        wandb.init(
            project=FLAGS.wandb_project,
            entity=FLAGS.wandb_entity,
            config=edict({**FLAGS.flag_values_dict(), **hparams}),
            name=algo_prefix
        )

    #==============================
    # Runner 
    #==============================
    file_name = os.path.join(res_dir, algo_prefix) + '.npz' 

    if FLAGS.algo_group in ('robust-offline', 'neural-regression', 'risk-exact', 'quantile-risk', 'risk-lin-lcb'):
        # -------------------------------------------------------
        # Generic offline batch runner – works for any group that
        # uses the train_offline_batch / sample_action interface.
        # Each algo in `algos` is evaluated independently on the
        # same dataset for each simulation seed.
        # -------------------------------------------------------
        num_algos = len(algos)
        algo_names = [a.name for a in algos]

        # Per-algo accumulators
        all_regrets   = [[] for _ in range(num_algos)]
        all_accs      = [[] for _ in range(num_algos)]
        all_gt_cvars  = [[] for _ in range(num_algos)]
        all_gt_means  = [[] for _ in range(num_algos)]
        all_gt_vars   = [[] for _ in range(num_algos)]
        all_times        = [[] for _ in range(num_algos)]
        all_oracle_cvars = []
        all_oracle_means = []
        all_oracle_vars  = []
        
        # --- Pre-calculate Oracle once for all simulations ---
        # Get test set and true means (fixed seed inside reset_data for test set)
        _, _, _, test_ctx_full, test_mean_full = data.reset_data(0)
        
        if FLAGS.data_type == 'robust_syn':
            oracle_state = np.random.RandomState(42)
            oracle_noise = data.generate_noise(test_ctx_full.shape[0]) 
            if hasattr(data, 'noise_type'):
                if data.noise_type == 'student-t':
                    oracle_noise = oracle_state.standard_t(df=2.1, size=(test_ctx_full.shape[0],))
                elif data.noise_type == 'binary-heavy':
                    delta = np.sqrt(1.0 / (40 * FLAGS.num_contexts))
                    oracle_noise = oracle_state.choice([-1/delta, 1/delta, -delta, delta], 
                                                    size=(test_ctx_full.shape[0],), p=[0.01, 0.01, 0.49, 0.49])
                else:
                    oracle_noise = oracle_state.normal(0, FLAGS.noise_std, size=(test_ctx_full.shape[0],))
            
            if FLAGS.oracle_eval_method == 'global':
                print(f"[Oracle] Solving Marginal MILP once for N={FLAGS.num_contexts}...")
                oracle_actions = data.get_oracle_optimal_actions_milp(test_ctx_full, FLAGS.alpha, noise_samples=oracle_noise)
            else:
                oracle_actions = data.get_oracle_optimal_actions(test_ctx_full, oracle_noise, FLAGS.alpha)
            
            opt_vals_full = test_mean_full[np.arange(test_mean_full.shape[0]), oracle_actions.ravel()]
            
            # Marginal CVaR for Oracle
            I_test = test_ctx_full.shape[0]
            M_noise = oracle_noise.shape[0]
            expanded_opt_vals = np.repeat(opt_vals_full, M_noise)
            expanded_noise = np.tile(oracle_noise, I_test)
            oracle_noisy_r = expanded_opt_vals + expanded_noise
            
            oracle_mean = np.mean(oracle_noisy_r)
            oracle_var  = np.var(oracle_noisy_r)
            sorted_o = np.sort(oracle_noisy_r)
            oracle_cvar = np.mean(sorted_o[:int(max(1, FLAGS.alpha * len(sorted_o)))])

            # --- Oracle Mean (For Standard Regret) ---
            # The absolute best expected reward regardless of risk
            mean_opt_vals_full = np.max(test_mean_full, axis=1) 
        else:
            oracle_actions = np.argmax(test_mean_full, axis=1)
            opt_vals_full = test_mean_full[np.arange(test_mean_full.shape[0]), oracle_actions.ravel()]
            oracle_mean = np.mean(opt_vals_full)
            oracle_var  = np.var(opt_vals_full)
            oracle_cvar = 0.0
            oracle_noise = None

        for sim in range(FLAGS.num_sim):
            print(f'Simulation: {sim + 1}/{FLAGS.num_sim}')
            
            # 1. Reset data (Train set varies by sim, Test set is fixed inside)
            contexts, actions, rewards, test_ctx, test_mean = data.reset_data(sim)
            # Re-use the pre-calculated oracle context/mean to ensure 100% consistency
            test_ctx, test_mean = test_ctx_full, test_mean_full

            if len(rewards.shape) > 1:
                n = contexts.shape[0]
                beh_rewards = rewards[np.arange(n), actions.ravel()]
            else:
                beh_rewards = rewards

            all_oracle_means.append(oracle_mean)
            all_oracle_vars.append(oracle_var)
            all_oracle_cvars.append(oracle_cvar)
            
            opt_vals = opt_vals_full
            opt_actions = oracle_actions
            # 2. Train + evaluate each algo
            for ai, algo in enumerate(algos):
                t0 = time.time()
                algo.reset(sim * 1111)
                algo.train_offline_batch(contexts, actions, beh_rewards)

                if FLAGS.agent_eval_method == 'global' and hasattr(algo, 'sample_action_milp'):
                    if oracle_noise is None:
                        eval_noise = np.random.standard_t(df=2.1, size=(100,))
                    else:
                        eval_noise = oracle_noise
                    test_actions = algo.sample_action_milp(test_ctx, noise_samples=eval_noise)
                else:
                    test_actions = algo.sample_action(test_ctx)
                sel_vals = test_mean[np.arange(test_mean.shape[0]), test_actions.ravel()]
                
                # 1. Reward Regret: Loss in expected value vs Mean-Optimal policy
                # (Note: mean_opt_vals_full is sliced to match current test_mean if needed)
                current_mean_opt = mean_opt_vals_full[:test_mean.shape[0]]
                regret = np.mean(current_mean_opt - sel_vals)
                acc    = np.mean(test_actions.ravel() == opt_actions.ravel())

                # Ground-truth risk stats (Calculated Marginally for accuracy)
                if FLAGS.data_type == 'robust_syn' and oracle_noise is not None:
                    # Marginal Agent Evaluation: Union of samples across all contexts
                    I_test = test_ctx.shape[0]
                    M_noise = oracle_noise.shape[0]
                    expanded_sel_vals = np.repeat(sel_vals, M_noise)
                    expanded_noise = np.tile(oracle_noise, I_test)
                    agent_noisy_r = expanded_sel_vals + expanded_noise
                elif FLAGS.data_type == 'simglucose':
                    agent_noisy_r = sel_vals
                else:
                    agent_noisy_r = None

                if agent_noisy_r is not None:
                    gt_mean = np.mean(agent_noisy_r)
                    gt_var  = np.var(agent_noisy_r)
                    sorted_agent = np.sort(agent_noisy_r)
                    # Marginal CVaR
                    gt_cvar = np.mean(sorted_agent[:int(FLAGS.alpha * len(sorted_agent))])
                else:
                    gt_mean = gt_var = gt_cvar = 0.0

                subopt = oracle_cvar - gt_cvar
                elapsed = time.time() - t0
                print(f'  [{algo.name}] Regret={regret:.4f} | Acc={acc:.4f} '
                      f'| GT Mean/Var/CVaR={gt_mean:.2f}/{gt_var:.2f}/{gt_cvar:.2f} '
                      f'| Oracle CVaR={oracle_cvar:.2f} | Subopt={subopt:.2f} | t={elapsed:.1f}s')

                if FLAGS.use_wandb:
                    wandb.log({
                        "sim": sim, "algo": algo.name,
                        "test_regret": regret, "test_accuracy": acc,
                        "gt_mean": gt_mean, "gt_var": gt_var, "gt_cvar": gt_cvar,
                        "oracle_mean": oracle_mean, "oracle_var": oracle_var, "oracle_cvar": oracle_cvar,
                    })

                all_regrets[ai].append(regret)
                all_accs[ai].append(acc)
                all_gt_cvars[ai].append(gt_cvar)
                all_gt_means[ai].append(gt_mean)
                all_gt_vars[ai].append(gt_var)
                all_times[ai].append(elapsed)

        # 3. Save results – one array per algo, keyed by algo name
        save_dict = {
            "algo_names": np.array(algo_names),
            "oracle_cvars":  np.array(all_oracle_cvars),
            "oracle_means":  np.array(all_oracle_means),
            "oracle_vars":   np.array(all_oracle_vars),
        }
        for ai, name in enumerate(algo_names):
            safe = name.replace(' ', '_')
            save_dict[f"{safe}_regrets"] = np.array(all_regrets[ai], dtype=np.float32)
            save_dict[f"{safe}_accs"]    = np.array(all_accs[ai],    dtype=np.float32)
            save_dict[f"{safe}_gt_cvars"]= np.array(all_gt_cvars[ai],dtype=np.float32)
            save_dict[f"{safe}_gt_means"]= np.array(all_gt_means[ai],dtype=np.float32)
            save_dict[f"{safe}_gt_vars"] = np.array(all_gt_vars[ai], dtype=np.float32)
            save_dict[f"{safe}_times"]   = np.array(all_times[ai],   dtype=np.float32)
            # Thêm lưu suboptimality
            save_dict[f"{safe}_subopt"]  = np.array(all_oracle_cvars, dtype=np.float32) - np.array(all_gt_cvars[ai], dtype=np.float32)
        np.savez(file_name, **save_dict)

        # Backward-compat: also expose flat regrets/errs for the first algo
        regrets = np.array(all_regrets[0], dtype=np.float32).reshape(FLAGS.num_sim, 1, 1)
        errs    = (1.0 - np.array(all_accs[0], dtype=np.float32)).reshape(FLAGS.num_sim, 1, 1)
    else:
        regrets, errs = contextual_bandit_runner(algos, data, FLAGS.num_sim,
            FLAGS.update_freq, FLAGS.test_freq, FLAGS.verbose, FLAGS.debug, FLAGS.normalize, file_name)
        np.savez(file_name, regrets=regrets, errs=errs)


    if FLAGS.use_wandb:
        wandb.finish()


if __name__ == '__main__': 
    app.run(main)
