"""Mini main for testing algorithms. """ 

import numpy as np 
import jax  
import jax.numpy as jnp 
from easydict import EasyDict as edict
import os 
import time 

from core.contextual_bandit import contextual_bandit_runner
from algorithms.neural_offline_bandit import ExactNeuraLCBV2, NeuralGreedyV2, ApproxNeuraLCBV2, RobustOfflineBatchNeuraLCB
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
flags.DEFINE_string('save_model_path', 'results/model.pkl', 'Path to save weights after training')

# Logging
flags.DEFINE_boolean('use_wandb', False, 'Whether to use wandb for logging')
flags.DEFINE_string('wandb_project', 'offline_neural_bandits', 'wandb project name')
flags.DEFINE_string('wandb_entity', None, 'wandb entity')

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
        'truncation_mode': FLAGS.truncation_mode
    })

    lin_hparams = edict(
        {
            'context_dim': hparams.context_dim, 
            'num_actions': hparams.num_actions, 
            'lambd0': hparams.lambd0, 
            'beta': hparams.beta, 
            'rbf_sigma': FLAGS.rbf_sigma, # 0.1, 1, 10
            'max_num_sample': 1000 
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

    if FLAGS.algo_group == 'robust-offline':
        algos = [
            RobustOfflineBatchNeuraLCB(hparams)
        ]
        layer_str = "-".join([str(s) for s in layer_sizes])
        algo_prefix = 'robust_{}_risk={}_tau={}_beta={}_n={}_layers={}'.format(
            FLAGS.data_type, FLAGS.risk_measure, FLAGS.tau_n, FLAGS.beta, FLAGS.num_contexts, layer_str
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

    if FLAGS.algo_group == 'robust-offline':
        all_regrets = []
        all_accs = []
        all_times = []
        all_gt_cvars = []
        all_oracle_cvars = []
        all_gt_means = []
        all_gt_vars = []
        all_oracle_means = []
        all_oracle_vars = []
        
        for sim in range(FLAGS.num_sim):
            t0 = time.time()
            print(f'Simulation: {sim + 1}/{FLAGS.num_sim}')
            
            # 1. Reset data and algo
            contexts, actions, rewards, test_ctx, test_mean = data.reset_data(sim)
            
            # behavior rewards only for selected actions (for training)
            beh_rewards = rewards 
            
            algo = algos[0]
            algo.reset(sim * 1111)
            
            # 2. Train Batch
            algo.train_offline_batch(contexts, actions, beh_rewards)

            # 3. Policy Evaluation (Run action selection once)
            test_actions = algo.sample_action(test_ctx)
            opt_actions = np.argmax(test_mean, axis=1)
            
            # 3.1 Calculate True Regret and Accuracy
            opt_vals = test_mean[np.arange(test_mean.shape[0]), opt_actions.ravel()]
            sel_vals = test_mean[np.arange(test_mean.shape[0]), test_actions.ravel()]
            
            regret = np.mean(opt_vals - sel_vals)
            acc = np.mean(test_actions.ravel() == opt_actions.ravel())
            
            # 3.2 Calculate Ground Truth Risks (Oracle Evaluation)
            if FLAGS.data_type == 'robust_syn':
                # Agent 
                fresh_noise = data.generate_noise(sel_vals.shape)
                agent_noisy_r = sel_vals + fresh_noise
                # Oracle
                oracle_noise = data.generate_noise(opt_vals.shape)
                oracle_noisy_r = opt_vals + oracle_noise
            elif FLAGS.data_type == 'simglucose':
                # Simglucose already contains realized rewards in its matrix
                agent_noisy_r = sel_vals
                oracle_noisy_r = opt_vals
            else:
                agent_noisy_r, oracle_noisy_r = None, None

            if agent_noisy_r is not None:
                # Calculate Agent stats
                gt_mean = np.mean(agent_noisy_r)
                gt_var = np.var(agent_noisy_r)
                sorted_agent = np.sort(agent_noisy_r)
                gt_cvar = np.mean(sorted_agent[:int(FLAGS.alpha * len(sorted_agent))])

                # Calculate Oracle stats
                oracle_mean = np.mean(oracle_noisy_r)
                oracle_var = np.var(oracle_noisy_r)
                sorted_oracle = np.sort(oracle_noisy_r)
                oracle_cvar = np.mean(sorted_oracle[:int(FLAGS.alpha * len(sorted_oracle))])

                stat_str = f" | GT Mean/Var/CVaR: {gt_mean:.2f}/{gt_var:.2f}/{gt_cvar:.2f}"
                ora_str = f" | Oracle Mean/Var/CVaR: {oracle_mean:.2f}/{oracle_var:.2f}/{oracle_cvar:.2f}"
                print(f'Regret: {regret:.4f} | Acc: {acc:.4f}{stat_str}{ora_str}')
            else:
                gt_mean = gt_var = gt_cvar = 0.0
                oracle_mean = oracle_var = oracle_cvar = 0.0
                print(f'Regret: {regret:.4f} | Acc: {acc:.4f}')
            
            if FLAGS.use_wandb:
                log_data = {
                    "sim": sim, "test_regret": regret, "test_accuracy": acc,
                    "gt_mean": gt_mean, "gt_var": gt_var, "gt_cvar": gt_cvar,
                    "oracle_mean": oracle_mean, "oracle_var": oracle_var, "oracle_cvar": oracle_cvar
                }
                wandb.log(log_data)
            
            all_regrets.append(regret)
            all_accs.append(acc)
            all_times.append(time.time() - t0)
            all_gt_cvars.append(gt_cvar)
            all_oracle_cvars.append(oracle_cvar)
            all_gt_means.append(gt_mean)
            all_gt_vars.append(gt_var)
            all_oracle_means.append(oracle_mean)
            all_oracle_vars.append(oracle_var)
            
        regrets = np.array(all_regrets, dtype=np.float32).reshape(FLAGS.num_sim, 1, 1)
        errs = (1.0 - np.array(all_accs, dtype=np.float32)).reshape(FLAGS.num_sim, 1, 1)
        
        save_dict = {
            "regrets": regrets,
            "errs": errs,
            "times": np.array(all_times),
            "gt_cvars": np.array(all_gt_cvars),
            "oracle_cvars": np.array(all_oracle_cvars),
            "gt_means": np.array(all_gt_means),
            "gt_vars": np.array(all_gt_vars),
            "oracle_means": np.array(all_oracle_means),
            "oracle_vars": np.array(all_oracle_vars)
        }
        np.savez(file_name, **save_dict)
    else:
        regrets, errs = contextual_bandit_runner(algos, data, FLAGS.num_sim, 
            FLAGS.update_freq, FLAGS.test_freq, FLAGS.verbose, FLAGS.debug, FLAGS.normalize, file_name)
        np.savez(file_name, regrets=regrets, errs=errs)

    if FLAGS.use_wandb:
        wandb.finish()


if __name__ == '__main__': 
    app.run(main)
