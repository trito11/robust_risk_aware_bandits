"""Define neural offline bandit algorithms. """

import math
import jax 
import jax.numpy as jnp
import optax
import numpy as np 
from tqdm import tqdm
from joblib import Parallel, delayed
from core.bandit_algorithm import BanditAlgorithm 
from core.bandit_dataset import BanditDataset
from core.utils import inv_sherman_morrison, inv_sherman_morrison_single_sample, vectorize_tree
from algorithms.neural_bandit_model import NeuralBanditModel, NeuralBanditModelV2

class ExactNeuraLCBV2(BanditAlgorithm):
    """NeuraLCB using exact confidence matrix and NeuralBanditModelV2. """
    def __init__(self, hparams, update_freq=1, name='ExactNeuraLCBV2'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModelV2(opt, hparams, '{}-net'.format(name))
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, '{}-data'.format(name))

        self.Lambda_inv = jnp.array(
            [
                jnp.eye(self.nn.num_params)/hparams.lambd0 for _ in range(hparams.num_actions)
            ]
        ) # (num_actions, p, p)

    def reset(self, seed): 
        self.Lambda_inv = jnp.array(
            [
                jnp.eye(self.nn.num_params)/ self.hparams.lambd0 for _ in range(self.hparams.num_actions)
            ]
        ) # (num_actions, p, p)

        self.nn.reset(seed) 
        self.data.reset()

    def sample_action(self, contexts):
        """
        Args:
            context: (None, self.hparams.context_dim)
        """
        cs = self.hparams.chunk_size
        num_chunks = math.ceil(contexts.shape[0] / cs)
        acts = []
        for i in range(num_chunks):
            ctxs = contexts[i * cs: (i+1) * cs,:] 
            lcb = []
            for a in range(self.hparams.num_actions):
                actions = jnp.ones(shape=(ctxs.shape[0],)) * a 

                f = self.nn.out(self.nn.params, ctxs, actions) # (num_samples, 1)
                # g = self.nn.grad_out(self.nn.params, convoluted_contexts) / jnp.sqrt(self.nn.m) # (num_samples, p)
                g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m)
                gA = g @ self.Lambda_inv[a,:,:] # (num_samples, p)
                
                gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_samples, )
                cnf = jnp.sqrt(gAg) # (num_samples,)

                lcb_a = f.ravel() - self.hparams.beta * cnf.ravel()  # (num_samples,)
                lcb.append(lcb_a.reshape(-1,1)) 
            lcb = jnp.hstack(lcb) 
            acts.append( jnp.argmax(lcb, axis=1)) 
        return jnp.hstack(acts)

    
    def update_buffer(self, contexts, actions, rewards): 
        self.data.add(contexts, actions, rewards)
        
    def update(self, contexts, actions, rewards):
        """Update the network parameters and the confidence parameter.
        
        Args:
            contexts: An array of d-dimensional contexts
            actions: An array of integers in [0, K-1] representing the chosen action 
            rewards: An array of real numbers representing the reward for (context, action)
        
        """

        # self.data.add(contexts, actions, rewards)
        self.nn.train(self.data, self.hparams.num_steps)

        # Update confidence parameter over all samples in the batch
        # convoluted_contexts = self.nn.action_convolution(contexts, actions)  
        # u = self.nn.grad_out(self.nn.params, convoluted_contexts) / jnp.sqrt(self.nn.m)  # (num_samples, p)
        u = self.nn.grad_out(self.nn.params, contexts, actions) / jnp.sqrt(self.nn.m)
        for i in range(contexts.shape[0]):
            jax.ops.index_update(self.Lambda_inv, actions[i], \
                inv_sherman_morrison_single_sample(u[i,:], self.Lambda_inv[actions[i],:,:]))

    def monitor(self, contexts=None, actions=None, rewards=None):
        norm = jnp.hstack(( jnp.ravel(param) for param in jax.tree_util.tree_leaves(self.nn.params)))

        preds = self.nn.out(self.nn.params, contexts, actions) # (num_samples,)

        cnfs = []
        for a in range(self.hparams.num_actions):
            actions_tmp = jnp.ones(shape=(contexts.shape[0],)) * a 

            f = self.nn.out(self.nn.params, contexts, actions_tmp) # (num_samples, 1)
            g = self.nn.grad_out(self.nn.params, contexts, actions_tmp) / jnp.sqrt(self.nn.m) # (num_samples, p)
            gA = g @ self.Lambda_inv[a,:,:] # (num_samples, p)
            
            gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_samples, )
            cnf = jnp.sqrt(gAg) # (num_samples,)

            cnfs.append(cnf) 
        cnf = jnp.hstack(cnfs) 

        cost = self.nn.loss(self.nn.params, contexts, actions, rewards)
        a = int(actions.ravel()[0])
        if self.hparams.debug_mode == 'simple':
            print('     r: {} | a: {} | f: {} | cnf: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], a, \
                preds.ravel()[0], \
                cnf.ravel()[a], cost, jnp.mean(jnp.square(norm))))
        else:
            print('     r: {} | a: {} | f: {} | cnf: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], \
                a, preds.ravel(), \
                cnf.ravel(), cost, jnp.mean(jnp.square(norm))))


class ApproxNeuraLCBV2(BanditAlgorithm):
    """NeuraLCB using exact confidence matrix and NeuralBanditModelV2. """
    def __init__(self, hparams, update_freq=1, name='ApproxNeuraLCBV2'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModelV2(opt, hparams, '{}-net'.format(name))
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, '{}-data'.format(name))

        self.diag_Lambda = [
                jnp.ones(self.nn.num_params)* hparams.lambd0 for _ in range(hparams.num_actions)
            ]
         # (num_actions, p)

    def reset(self, seed): 
        self.diag_Lambda = [
                jnp.ones(self.nn.num_params) * self.hparams.lambd0 for _ in range(self.hparams.num_actions)
            ]
         # (num_actions, p)

        self.nn.reset(seed) 
        self.data.reset()

    def sample_action(self, contexts):
        """
        Args:
            context: (None, self.hparams.context_dim)
        """
        cs = self.hparams.chunk_size
        num_chunks = math.ceil(contexts.shape[0] / cs)
        acts = []
        for i in range(num_chunks):
            ctxs = contexts[i * cs: (i+1) * cs,:] 
            lcb = []
            for a in range(self.hparams.num_actions):
                actions = jnp.ones(shape=(ctxs.shape[0],)) * a 

                f = self.nn.out(self.nn.params, ctxs, actions) # (num_samples, 1)
                # g = self.nn.grad_out(self.nn.params, convoluted_contexts) / jnp.sqrt(self.nn.m) # (num_samples, p)
                g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m)
                gAg = jnp.sum( jnp.square(g) / self.diag_Lambda[a][:], axis=-1) # (None, p) -> (None,)

                cnf = jnp.sqrt(gAg) # (num_samples,)
                lcb_a = f.ravel() - self.hparams.beta * cnf.ravel()  # (num_samples,)
                lcb.append(lcb_a.reshape(-1,1)) 
            lcb = jnp.hstack(lcb) 
            # print(lcb)
            acts.append( jnp.argmax(lcb, axis=1)) 
        return jnp.hstack(acts)

        
        # def process(i):
        #     ctxs = contexts[i * cs: (i+1) * cs,:] 
        #     lcb = []
        #     for a in range(self.hparams.num_actions):
        #         actions = jnp.ones(shape=(ctxs.shape[0],)) * a 

        #         f = self.nn.out(self.nn.params, ctxs, actions) # (num_samples, 1)
        #         # g = self.nn.grad_out(self.nn.params, convoluted_contexts) / jnp.sqrt(self.nn.m) # (num_samples, p)
        #         g = self.nn.grad_out(self.nn.params, ctxs, actions) / jnp.sqrt(self.nn.m)
        #         gAg = jnp.sum( jnp.square(g) / self.diag_Lambda[a][:], axis=-1) # (None, p) -> (None,)

        #         cnf = jnp.sqrt(gAg) # (num_samples,)
        #         lcb_a = f.ravel() - self.hparams.beta * cnf.ravel()  # (num_samples,)
        #         lcb.append(lcb_a.reshape(-1,1)) 
        #     lcb = jnp.hstack(lcb) 
        #     # print(lcb)
        #     return jnp.argmax(lcb, axis=1)
    
        # acts = Parallel(n_jobs=50,prefer="threads")(delayed(process)(i) for i in range(num_chunks))
        # return jnp.hstack(acts)

    def update_buffer(self, contexts, actions, rewards): 
        self.data.add(contexts, actions, rewards)

    def update(self, contexts, actions, rewards):
        """Update the network parameters and the confidence parameter.
        
        Args:
            contexts: An array of d-dimensional contexts
            actions: An array of integers in [0, K-1] representing the chosen action 
            rewards: An array of real numbers representing the reward for (context, action)
        
        """

        # Should run self.update_buffer before self.update to update the model in the latest data. 
        self.nn.train(self.data, self.hparams.num_steps)

        # Update confidence parameter over all samples in the batch
        # convoluted_contexts = self.nn.action_convolution(contexts, actions)  
        # u = self.nn.grad_out(self.nn.params, convoluted_contexts) / jnp.sqrt(self.nn.m)  # (num_samples, p)
        u = self.nn.grad_out(self.nn.params, contexts, actions) / jnp.sqrt(self.nn.m)
        for i in range(contexts.shape[0]):
            # jax.ops.index_update(self.diag_Lambda, actions[i], \
            #     jnp.square(u[i,:]) + self.diag_Lambda[actions[i],:])  
            self.diag_Lambda[actions[i]] = self.diag_Lambda[actions[i]] + jnp.square(u[i,:])

    def monitor(self, contexts=None, actions=None, rewards=None):
        norm = jnp.hstack(( jnp.ravel(param) for param in jax.tree_util.tree_leaves(self.nn.params)))

        preds = self.nn.out(self.nn.params, contexts, actions) # (num_samples,)

        cnfs = []
        for a in range(self.hparams.num_actions):
            actions_tmp = jnp.ones(shape=(contexts.shape[0],)) * a 

            f = self.nn.out(self.nn.params, contexts, actions_tmp) # (num_samples, 1)
            g = self.nn.grad_out(self.nn.params, contexts, actions_tmp) / jnp.sqrt(self.nn.m) # (num_samples, p)
            gAg = jnp.sum( jnp.square(g) / self.diag_Lambda[a][:], axis=-1)
            cnf = jnp.sqrt(gAg) # (num_samples,)

            cnfs.append(cnf) 
        cnf = jnp.hstack(cnfs) 

        cost = self.nn.loss(self.nn.params, contexts, actions, rewards)
        a = int(actions.ravel()[0])
        if self.hparams.debug_mode == 'simple':
            print('     r: {} | a: {} | f: {} | cnf: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], a, \
                preds.ravel()[0], \
                cnf.ravel()[a], cost, jnp.mean(jnp.square(norm))))
        else:
            print('     r: {} | a: {} | f: {} | cnf: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], \
                a, preds.ravel(), \
                cnf.ravel(), cost, jnp.mean(jnp.square(norm))))


class NeuralGreedyV2(BanditAlgorithm):
    def __init__(self, hparams, update_freq=1, name='NeuralGreedyV2'):
        self.name = name 
        self.hparams = hparams 
        self.update_freq = update_freq 
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModelV2(opt, hparams, '{}-net'.format(name))
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, '{}-data'.format(name))

    def reset(self, seed): 
        self.nn.reset(seed) 
        self.data.reset()

    def sample_action(self, contexts):
        preds = []
        for a in range(self.hparams.num_actions):
            actions = jnp.ones(shape=(contexts.shape[0],)) * a 
            f = self.nn.out(self.nn.params, contexts, actions) # (num_samples, 1)
            preds.append(f) 
        preds = jnp.hstack(preds) 
        return jnp.argmax(preds, axis=1)

    def update_buffer(self, contexts, actions, rewards): 
        self.data.add(contexts, actions, rewards)

    def update(self, contexts, actions, rewards):
        """Update the network parameters and the confidence parameter.
        
        Args:
            context: An array of d-dimensional contexts
            action: An array of integers in [0, K-1] representing the chosen action 
            reward: An array of real numbers representing the reward for (context, action)
        
        """

        # self.data.add(contexts, actions, rewards)
        self.nn.train(self.data, self.hparams.num_steps)

    def monitor(self, contexts=None, actions=None, rewards=None):
        norm = jnp.hstack(( jnp.ravel(param) for param in jax.tree_util.tree_leaves(self.nn.params)))

        convoluted_contexts = self.nn.action_convolution(contexts, actions)

        preds = self.nn.out(self.nn.params, contexts, actions) # (num_samples,)

        preds = []
        for a in range(self.hparams.num_actions):
            actions_tmp = jnp.ones(shape=(contexts.shape[0],)) * a 
            f = self.nn.out(self.nn.params, contexts, actions_tmp) # (num_samples, 1)
            preds.append(f) 
        preds = jnp.hstack(preds) 

        cost = self.nn.loss(self.nn.params, contexts, actions, rewards)

        a = int(actions.ravel()[0])
        if self.hparams.debug_mode == 'simple':
            print('     r: {} | a: {} | f: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], \
                a, preds.ravel()[a % preds.size], \
                cost, jnp.mean(jnp.square(norm))))
        else:
            print('     r: {} | a: {} | f: {} | loss: {} | param_mean: {}'.format(rewards.ravel()[0], \
                a, preds.ravel(), \
                cost, jnp.mean(jnp.square(norm))))
#===================================================================================================

class NeuraLCB(BanditAlgorithm):
    """NeuraLCB using diag approximation for confidence matrix. """
    def __init__(self, hparams, name='NeuraLCB'):
        self.name = name 
        self.hparams = hparams 
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModel(opt, hparams, 'nn')
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, 'bandit_data')

        self.diag_Lambda = jnp.array(
                [hparams.lambd0 * jnp.ones(self.nn.num_params) for _ in range(self.hparams.num_actions) ]
            ) # (num_actions, p)

    def sample_action(self, contexts):
        """
        Args:
            contexts: (None, self.hparams.context_dim)
        """
        n = contexts.shape[0] 
        
        if n <= self.hparams.max_test_batch:
            f = self.nn.out(self.nn.params, contexts) # (num_samples, num_actions)
            g = self.nn.grad_out(self.nn.params, contexts) # (num_actions, num_samples, p)

            cnf = jnp.sqrt( jnp.sum(jnp.square(g) / self.diag_Lambda.reshape(self.hparams.num_actions,1,-1), axis=-1) ) / jnp.sqrt(self.nn.m)
            lcb = f - self.hparams.beta * cnf.T   # (num_samples, num_actions)
            return jnp.argmax(lcb, axis=1)
        else: # Break contexts in batches if it is large.
            inv = int(n / self.hparams.max_test_batch)
            acts = []
            for i in range(inv):
                c = contexts[i*self.hparams.max_test_batch:self.hparams.max_test_batch*(i+1),:]
                f = self.nn.out(self.nn.params, c) # (num_samples, num_actions)
                g = self.nn.grad_out(self.nn.params, c) # (num_actions, num_samples, p)

                cnf = jnp.sqrt( jnp.sum(jnp.square(g) / self.diag_Lambda.reshape(self.hparams.num_actions,1,-1), axis=-1) ) / jnp.sqrt(self.nn.m)
                lcb = f - self.hparams.beta * cnf.T   # (num_samples, num_actions)
                acts.append(jnp.argmax(lcb, axis=1).ravel())
            return jnp.array(acts)
            

    def update(self, contexts, actions, rewards):
        """Update the network parameters and the confidence parameter.
        
        Args:
            contexts: An array of d-dimensional contexts
            actions: An array of integers in [0, K-1] representing the chosen action 
            rewards: An array of real numbers representing the reward for (context, action)
        
        """

        self.data.add(contexts, actions, rewards)
        self.nn.train(self.data, self.hparams.num_steps)

        # Update confidence parameter over all samples in the batch
        g = self.nn.grad_out(self.nn.params, contexts)  # (num_actions, num_samples, p)
        g = jnp.square(g) / self.nn.m 
        for i in range(g.shape[1]): 
            self.diag_Lambda += g[:,i,:]

    def monitor(self, contexts=None, actions=None, rewards=None):
        params = jnp.hstack(( jnp.ravel(param) for param in jax.tree_util.tree_leaves(self.nn.params)))

        f = self.nn.out(self.nn.params, contexts) # (num_samples, num_actions)
        g = self.nn.grad_out(self.nn.params, contexts) # (num_actions, num_samples, p)

        cnf = jnp.sqrt( jnp.sum(jnp.square(g) / self.diag_Lambda.reshape(self.hparams.num_actions,1,-1), axis=-1) ) / jnp.sqrt(self.nn.m)
        
        # action and reward fed here are in vector forms, we convert them into an array of one-hot vectors
        action_hot = jax.nn.one_hot(actions.ravel(), self.hparams.num_actions) 
        reward_hot = action_hot * rewards.reshape(-1,1)

        cost = self.nn.loss(self.nn.params, contexts, action_hot, reward_hot)
        print('     r: {} | a: {} | f: {} | cnf: {} | param_mean: {}'.format(rewards.ravel()[0], actions.ravel()[0], f.ravel(), \
            cnf.ravel(), jnp.mean(jnp.square(params))))

class ExactNeuraLCB(BanditAlgorithm):
    """NeuraLCB using exact confidence matrix. """
    def __init__(self, hparams, name='ExactNeuraLCB'):
        self.name = name 
        self.hparams = hparams 
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModel(opt, hparams, 'nn')
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, 'bandit_data')

        # self.diag_Lambda = jnp.array(
        #         [hparams.lambd0 * jnp.ones(self.nn.num_params) for _ in range(self.hparams.num_actions) ]
        #     ) # (num_actions, p)

        self.Lambda_inv = jnp.array(
            [
                np.eye(self.nn.num_params)/hparams.lambd0 for _ in range(hparams.num_actions)
            ]
        ) # (num_actions, p, p)

    def sample_action(self, contexts):
        """
        Args:
            context: (None, self.hparams.context_dim)
        """
        n = contexts.shape[0] 
        
        if n <= self.hparams.max_test_batch:
            f = self.nn.out(self.nn.params, contexts) # (num_samples, num_actions)
            g = self.nn.grad_out(self.nn.params, contexts) / jnp.sqrt(self.nn.m) # (num_actions, num_samples, p)
            gA = jnp.sum(jnp.multiply(g[:,:,None,:], self.Lambda_inv[:, None, :,:]), axis=-1) # (num_actions, num_samples, p)
            gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_actions, num_samples)
            cnf = jnp.sqrt(gAg).T # (num_samples, num_actions)

            lcb = f - self.hparams.beta * cnf   # (num_samples, num_actions)
            return jnp.argmax(lcb, axis=1)
        else: # Break contexts in batches if it is large.
            inv = int(n / self.hparams.max_test_batch)
            acts = []
            for i in range(inv):
                c = contexts[i*self.hparams.max_test_batch:self.hparams.max_test_batch*(i+1),:]
                f = self.nn.out(self.nn.params, c) # (num_samples, num_actions)
                # No change to g?
                g = self.nn.grad_out(self.nn.params, c) # (num_actions, num_samples, p)

                gA = jnp.sum(jnp.multiply(g[:,:,None,:], self.Lambda_inv[:, None, :,:]), axis=-1) # (num_actions, num_samples, p)
                gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_actions, num_samples)
                cnf = jnp.sqrt(gAg).T # (num_samples, num_actions)

                lcb = f - self.hparams.beta * cnf   # (num_samples, num_actions)
                acts.append(jnp.argmax(lcb, axis=1).ravel())
            return jnp.array(acts)
            

    def update(self, contexts, actions, rewards):
        """Update the network parameters and the confidence parameter.
        
        Args:
            contexts: An array of d-dimensional contexts, (None, context_dim)
            actions: An array of integers in [0, K-1] representing the chosen action, (None,)
            rewards: An array of real numbers representing the reward for (context, action), (None, )
        
        """

        self.data.add(contexts, actions, rewards)
        self.nn.train(self.data, self.hparams.num_steps)

        # Update confidence parameter over all samples in the batch
        u = self.nn.grad_out(self.nn.params, contexts) / jnp.sqrt(self.nn.m)  # (num_actions, num_samples, p)
        for i in range(u.shape[1]): 
            self.Lambda_inv =  inv_sherman_morrison(u[:,i,:], self.Lambda_inv)

    def monitor(self, contexts=None, actions=None, rewards=None):
        params = jnp.hstack(( jnp.ravel(param) for param in jax.tree_util.tree_leaves(self.nn.params)))

        f = self.nn.out(self.nn.params, contexts) # (num_samples, num_actions)
        g = self.nn.grad_out(self.nn.params, contexts) # (num_actions, num_samples, p)

        gA = jnp.sum(jnp.multiply(g[:,:,None,:], self.Lambda_inv[:, None, :,:]), axis=-1) # (num_actions, num_samples, p)
        gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) # (num_actions, num_samples)
        cnf = jnp.sqrt(gAg).T # (num_samples, num_actions)
        
        # action and reward fed here are in vector forms, we convert them into an array of one-hot vectors for computing loss
        action_hot = jax.nn.one_hot(actions.ravel(), self.hparams.num_actions) 
        reward_hot = action_hot * rewards.reshape(-1,1)

        cost = self.nn.loss(self.nn.params, contexts, action_hot, reward_hot)
        print('     r: {} | a: {} | f: {} | cnf: {} | param_mean: {}'.format(rewards.ravel()[0], actions.ravel()[0], f.ravel(), \
            cnf.ravel(), jnp.mean(jnp.square(params))))


class NeuralGreedy(BanditAlgorithm):
    def __init__(self, hparams, name='NeuralGreedy'):
        self.name = name 
        self.hparams = hparams 
        opt = optax.adam(hparams.lr)
        self.nn = NeuralBanditModel(opt, hparams, 'nn')
        self.data = BanditDataset(hparams.context_dim, hparams.num_actions, hparams.buffer_s, 'bandit_data')

    def sample_action(self, contexts):
        f = self.nn.out(self.nn.params, contexts) # (num_contexts, num_actions)
        return jnp.argmax(f, axis=1)

    def update(self, contexts, actions, rewards):
        """Update the network parameters and the confidence parameter.
        
        Args:
            context: An array of d-dimensional contexts
            action: An array of integers in [0, K-1] representing the chosen action 
            reward: An array of real numbers representing the reward for (context, action)
        
        """

        self.data.add(contexts, actions, rewards)
        self.nn.train(self.data, self.hparams.num_steps)

    def monitor(self, contexts=None, actions=None, rewards=None):
        params = jnp.hstack(( jnp.ravel(param) for param in jax.tree_util.tree_leaves(self.nn.params)))
        f = self.nn.out(self.nn.params, contexts) # (num_samples, num_actions)

        # action and reward fed here are in vector forms, we convert them into an array of one-hot vectors
        action_hot = jax.nn.one_hot(actions.ravel(), self.hparams.num_actions) 
        reward_hot = action_hot * rewards.reshape(-1,1)
        cost = self.nn.loss(self.nn.params, contexts, action_hot, reward_hot)
        print('     r: {} | a: {} | f: {} | param_mean: {}'.format(rewards.ravel()[0], actions.ravel()[0], f.ravel(), jnp.mean(jnp.square(params))))


# ============================================================
# ============================================================
# RobustOfflineBatchNeuraLCB
# ============================================================

class RobustOfflineBatchNeuraLCB(BanditAlgorithm):
    """Risk-Aware Offline Bandit using Tofu Loss, Risk Functionals, and Pessimistic LCB.

    This algorithm is designed for the *pure offline batch* regime.
    Key features:
    1. Tofu Loss: Reward truncation for robustness against heavy-tailed noise.
    2. Empirical Residuals: Simulation of return distributions using historical noise.
    3. Risk Functionals: Support for Mean, CVaR, Entropic, and Mean-Variance measures.
    4. Exact Covariance (Z): Direct inversion of the full covariance matrix for pessimism.

    Required hparams:
        risk_measure (str): 'mean', 'cvar', 'entropic', or 'mean_variance'.
        tau_n (float): Truncation threshold for Tofu reward clipping.
        alpha (float): CVaR tail probability (e.g., 0.05).
        beta (float): Scaling factor for the uncertainty penalty.
        lambd0 (float): Regularization coefficient for matrix Z.
        num_steps (int): Gradient steps for offline training.
    """

    def __init__(self, hparams, update_freq=1, name='RobustOfflineBatchNeuraLCB'):
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

    # ------------------------------------------------------------------
    # Risk Functional Helper
    # ------------------------------------------------------------------

    def _compute_risk_functional(self, Y, axis=-1):
        """Computes the requested risk measure on the sample distribution Y.
        
        Supported hparams.risk_measure:
          - 'mean': Standard arithmetic mean.
          - 'cvar': Conditional Value at Risk at level alpha.
          - 'entropic': Entropic Risk Measure with parameter theta.
          - 'mean_variance': Mean adjusted by a variance penalty.
        """
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

    def _get_risk_lipschitz_factor(self):
        """Returns the Lipschitz constant L_rho of the selected risk functional.
        
        This ensures the uncertainty penalty (R_a) is correctly scaled relative
        to the risk measure's sensitivity.
        """
        measure = getattr(self.hparams, 'risk_measure', 'cvar')
        if measure == 'cvar':
            # Handle alpha depending on notation (0.95 confidence level vs 0.05 tail)
            tail_prob = self.hparams.alpha if self.hparams.alpha < 0.5 else (1.0 - self.hparams.alpha)
            return 1.0 / tail_prob
        elif measure == 'mean':
            return 1.0
        elif measure == 'entropic':
            return 1.0
        elif measure == 'mean_variance':
            return 1.0
        return 1.0

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed):
        """Reset network weights and stored offline statistics."""
        self.nn.reset(seed)
        self.data.reset()
        self.historical_residuals = None
        self.Z_inv = None

    # ------------------------------------------------------------------
    # Offline Batch Training
    # ------------------------------------------------------------------

    def train_offline_batch(self, contexts, actions, rewards):
        """Train the network on the full offline dataset and compute Z_inv.

        Performs:
          1. Tofu truncation of rewards.
          2. Offline training for num_steps.
          3. Calculation of residuals for return simulation.
          4. Construction and inversion of the exact covariance matrix Z.
        
        Note: 'mask' mode generates dynamic shapes which may be incompatible with @jax.jit. 
        Use 'clip' mode (default) for full JAX JIT compatibility.
        """
        # Step 1: Reward Truncation (Tofu Loss)
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
            # Mask out samples exceeding the threshold
            mask = (jnp.abs(rewards) <= tau_n).ravel()
            train_ctx, train_act, train_rew = contexts[mask], actions[mask], rewards[mask]
            r_tilde = train_rew
        else:
            train_ctx, train_act, train_rew = contexts, actions, rewards
            r_tilde = rewards

        # Step 2: Training
        self.data.reset()
        self.data.add(train_ctx, train_act.reshape(-1, 1), train_rew.reshape(-1, 1))
        self.nn.train(self.data, self.hparams.num_steps)

        # Step 3: Empirical Residuals
        # Residuals are calculated using RAW rewards to preserve heavy-tail info for risk evaluation.
        f_hist = self.nn.out(self.nn.params, contexts, actions).ravel()
        self.historical_residuals = rewards.ravel() - f_hist

        # Step 4: Exact Covariance Matrix Z
        # Use chunked updates to stay within 8GB RAM for large parameter spaces
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
            # Force deletion to help memory?
            del g_chunk
            if self.hparams.verbose:
                pbar.update(end_idx - i)

        if self.hparams.verbose:
            pbar.close()

        self.Z_inv = jnp.linalg.inv(Z)
        del Z # Free memory immediately after inversion

        # Step 5: Residual Risk for Translation Invariance
        # Use Translation Invariance Property: rho(mu + residual) = mu + rho(residual)
        # Pre-compute rho(residual) once to avoid large matrix creation in test-time
        self.rho_residuals = self._compute_risk_functional(self.historical_residuals, axis=0)

        if self.hparams.verbose:
            print(f'[{self.name}] Batch Training: mean |resid|={jnp.mean(jnp.abs(self.historical_residuals)):.4f} | rho(resid)={self.rho_residuals:.4f}')

    # ------------------------------------------------------------------
    # Action Selection: Risk-Aware LCB
    # ------------------------------------------------------------------

    def sample_action(self, contexts):
        """Chooses actions using Point-wise Risk-Aware Lower Confidence Bounds.
        
        Contexts are processed in chunks to prevent Out-Of-Memory (OOM) errors 
        when broadcasting large historical residuals.
        """
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
                # Prediction
                actions_tmp = jnp.ones(shape=(B,)) * a
                mu_a = self.nn.out(self.nn.params, batch_contexts, actions_tmp).ravel()

                # Risk Measurement (Mean, CVaR, etc.)
                # Optimized using Translation Invariance Property
                rho_a = mu_a + self.rho_residuals

                # Uncertainty Penalty (Pessimism)
                g_test = self.nn.grad_out(self.nn.params, batch_contexts, actions_tmp) / jnp.sqrt(self.nn.m)
                R_a = jnp.sqrt(jnp.sum((g_test @ self.Z_inv) * g_test, axis=-1))

                L_rho = self._get_risk_lipschitz_factor()
                lcbs.append((rho_a - self.hparams.beta * L_rho * R_a).reshape(-1, 1))

            batch_LCBs = jnp.hstack(lcbs)
            all_actions.append(jnp.argmax(batch_LCBs, axis=1))
            
            if self.hparams.verbose:
                pbar.update(B)

        if self.hparams.verbose:
            pbar.close()

        return jnp.concatenate(all_actions, axis=0)

    def sample_action_milp(self, contexts, noise_samples=None):
        """Chooses actions by globally maximizing Marginal CVaR using MILP.
        
        Unlike `sample_action` which optimizes conditional point-wise CVaR, 
        this computes the global optimal policy coupling all contexts together.
        
        Args:
            contexts: A batch of contexts to evaluate (shape: I x dim).
            noise_samples: Optional true noise samples. If None, uses historical residuals.
        """
        import gurobipy as gp
        from gurobipy import GRB
        import numpy as np
        
        assert self.historical_residuals is not None, "Call train_offline_batch() first."
        assert self.Z_inv is not None, "Call train_offline_batch() first."
        
        if noise_samples is None:
            noise_samples = np.array(self.historical_residuals)
            
        I = contexts.shape[0]
        J = self.hparams.num_actions
        M = len(noise_samples)
        
        if self.hparams.verbose:
            print(f'[{self.name}] Extracting h_matrix for MILP (I={I}, J={J}, M={M})...')
            
        # 1. Build the matrices for mu and R
        mu_matrix = np.zeros((I, J))
        R_matrix = np.zeros((I, J))
        L_rho = self._get_risk_lipschitz_factor()
        chunk_size = getattr(self.hparams, 'chunk_size', 500)
        
        for a in range(J):
            mu_a_full = []
            R_a_full = []
            for i in range(0, I, chunk_size):
                batch_contexts = contexts[i : i + chunk_size]
                B = batch_contexts.shape[0]
                actions_tmp = jnp.ones(shape=(B,)) * a
                
                # Mean Prediction
                mu_a = self.nn.out(self.nn.params, batch_contexts, actions_tmp).ravel()
                
                # Uncertainty
                g_test = self.nn.grad_out(self.nn.params, batch_contexts, actions_tmp) / jnp.sqrt(self.nn.m)
                R_a = jnp.sqrt(jnp.sum((g_test @ self.Z_inv) * g_test, axis=-1))
                
                mu_a_full.append(np.array(mu_a))
                R_a_full.append(np.array(R_a))
                
            mu_matrix[:, a] = np.concatenate(mu_a_full)
            R_matrix[:, a] = np.concatenate(R_a_full)
            
        # 2. Formulate and solve the MILP
        if self.hparams.verbose:
            print(f'[{self.name}] Solving Exact Marginal CVaR LCB MILP...')
            
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 1 if self.hparams.verbose else 0)
        env.start()
        model = gp.Model("CVaR_LCB_MILP", env=env)
        
        z = model.addVars(I, J, vtype=GRB.BINARY, name="z")
        q = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="q")
        s = model.addVars(I, M, vtype=GRB.CONTINUOUS, lb=0.0, name="s")
        
        alpha = getattr(self.hparams, 'alpha', 0.9)
        tail_prob = alpha if alpha < 0.5 else (1.0 - alpha)
        
        # Objective: Exact Marginal CVaR - Uncertainty Penalty
        # CVaR = q - (1 / (I * M * tail_prob)) * sum(s_im)
        # Penalty = (beta * L_rho / I) * sum(z_ij * R_ij)
        beta = self.hparams.beta
        
        cvar_expr = q - (1.0 / (I * M * tail_prob)) * gp.quicksum(s[i, m] for i in range(I) for m in range(M))
        penalty_expr = (beta * L_rho / float(I)) * gp.quicksum(R_matrix[i, j] * z[i, j] for i in range(I) for j in range(J))
        
        model.setObjective(cvar_expr - penalty_expr, GRB.MAXIMIZE)
        
        for i in range(I):
            model.addConstr(gp.quicksum(z[i, j] for j in range(J)) == 1)
            
        for i in range(I):
            for m in range(M):
                # The random variable for CVaR is ONLY (mu + noise)
                mu_z_sum = gp.quicksum(mu_matrix[i, j] * z[i, j] for j in range(J))
                model.addConstr(s[i, m] >= q - (mu_z_sum + noise_samples[m]))
                
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            optimal_policy = np.zeros(I, dtype=int)
            for i in range(I):
                for j in range(J):
                    if z[i, j].X > 0.5:
                        optimal_policy[i] = j
                        break
            if self.hparams.verbose:
                print(f'[{self.name}] MILP solved. Max CVaR LCB = {model.ObjVal:.4f}')
            return jnp.array(optimal_policy)
        else:
            print(f'[{self.name}] MILP failed to find optimal solution. Fallback to point-wise LCB.')
            return self.sample_action(contexts)

    # ------------------------------------------------------------------
    # Global Policy Evaluation
    # ------------------------------------------------------------------

    def evaluate_offline_policy(self, contexts, pi_actions):
        """Evaluates the Global Marginal Risk of a specified policy.
        
        Warning: This creates a large flattened distribution array (M * N).
        If memory is limited, evaluate on a representative subset of contexts.
        """
        assert self.historical_residuals is not None, "Call train_offline_batch() first."
        
        # 1. Marginal Return Distribution
        mu = self.nn.out(self.nn.params, contexts, pi_actions).ravel()
        # Optimized using Translation Invariance Property
        marginal_risk = jnp.mean(mu) + self.rho_residuals
        
        # 2. Uncertainty Penalty
        g_test = self.nn.grad_out(self.nn.params, contexts, pi_actions) / jnp.sqrt(self.nn.m)
        pointwise_R = jnp.sqrt(jnp.sum((g_test @ self.Z_inv) * g_test, axis=-1))
        
        mean_R_pi = jnp.mean(pointwise_R)
        L_rho = self._get_risk_lipschitz_factor()
        lcb_marginal = marginal_risk - self.hparams.beta * L_rho * mean_R_pi
        
        return {
            "marginal_risk": marginal_risk,
            "mean_R_pi": mean_R_pi,
            "lcb_marginal": lcb_marginal,
            "pointwise_R": pointwise_R
        }

    def save_model(self, path):
        """Saves weights and algorithm statistics to a pickle file."""
        import pickle
        data = {
            'nn_params': self.nn.params,
            'nn_opt_state': self.nn.opt_state,
            'Z_inv': self.Z_inv,
            'historical_residuals': self.historical_residuals,
            'rho_residuals': self.rho_residuals
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        if self.hparams.verbose:
            print(f'[{self.name}] Model saved to {path}')

    def load_model(self, path):
        """Loads weights and algorithm statistics from a pickle file."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.nn.params = data['nn_params']
            self.nn.opt_state = data['nn_opt_state']
            self.Z_inv = data['Z_inv']
            self.historical_residuals = data['historical_residuals']
            self.rho_residuals = data.get('rho_residuals', None)
        if self.hparams.verbose:
            print(f'[{self.name}] Model loaded from {path}')

    # ------------------------------------------------------------------
    # Compatibility Stubs
    # ------------------------------------------------------------------

    def update_buffer(self, contexts, actions, rewards):
        self.data.add(contexts, actions.reshape(-1, 1), rewards.reshape(-1, 1))

    def update(self, contexts, actions, rewards):
        pass

    def monitor(self, contexts=None, actions=None, rewards=None):
        if self.historical_residuals is not None:
            print(f'[{self.name}] Residual Mean: {jnp.mean(jnp.abs(self.historical_residuals)):.4f}')
        else:
            print(f'[{self.name}] Model not trained.')


# ============================================================
# ============================================================
# NeuralRegressionOffline
# ============================================================

class NeuralRegressionOffline(BanditAlgorithm):
    """Pure Neural Regression baseline for offline contextual bandits.

    This is the simplest neural baseline in the offline batch setting.
    It trains a multi-layer perceptron to directly regress G(context, action)
    via MSE loss, using the same action-convoluted input as NeuralBanditModelV2:

        input = action_convolution(context, action)  # shape: (context_dim * num_actions,)
        G_hat = MLP(input)                           # scalar reward estimate

    Action selection is purely greedy:
        pi(context) = argmax_a  G_hat(context, a)

    No uncertainty quantification, no risk functional, no Tofu truncation.
    This serves as the *lower bound* baseline to compare against pessimistic
    and risk-aware algorithms (e.g. RobustOfflineBatchNeuraLCB).

    Required hparams (same schema as RobustOfflineBatchNeuraLCB):
        context_dim  (int):   Dimension of each context vector.
        num_actions  (int):   Number of discrete actions.
        layer_sizes  (list):  Hidden layer widths, e.g. [100, 100].
        lr           (float): Adam learning rate.
        lambd        (float): L2 regularization weight.
        num_steps    (int):   Gradient steps for offline training.
        batch_size   (int):   Mini-batch size per step.
        buffer_s     (int):   Replay buffer size (-1 = unlimited).
        data_rand    (bool):  Whether to sample batches randomly.
        verbose      (bool):  Print training progress.
    """

    def __init__(self, hparams, update_freq=1, name='NeuralRegressionOffline'):
        self.name = name
        self.hparams = hparams
        self.update_freq = update_freq

        opt = optax.adam(hparams.lr)
        # NeuralBanditModelV2 already implements:
        #   input  = action_convolution(context, action)  [context_dim * num_actions]
        #   output = MLP(input)                           [scalar]
        #   loss   = 0.5 * MSE + 0.5 * lambd * ||params||^2
        self.nn = NeuralBanditModelV2(opt, hparams, '{}-net'.format(name))

        self.data = BanditDataset(
            hparams.context_dim,
            hparams.num_actions,
            hparams.buffer_s,
            '{}-data'.format(name),
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed):
        """Reset network weights and replay buffer."""
        self.nn.reset(seed)
        self.data.reset()

    # ------------------------------------------------------------------
    # Offline Batch Training
    # ------------------------------------------------------------------

    def train_offline_batch(self, contexts, actions, rewards):
        """Train MLP regressor on the full offline dataset via MSE.

        Args:
            contexts: (N, context_dim)  – context vectors.
            actions:  (N,)              – integer action indices in [0, K-1].
            rewards:  (N,)              – observed scalar rewards.
        """
        self.data.reset()
        self.data.add(contexts, actions.reshape(-1, 1), rewards.reshape(-1, 1))
        self.nn.train(self.data, self.hparams.num_steps)

        if self.hparams.verbose:
            # Quick sanity check: in-sample MSE
            f_hist = self.nn.out(self.nn.params, contexts, actions.ravel()).ravel()
            mse = float(jnp.mean(jnp.square(f_hist - rewards.ravel())))
            print(f'[{self.name}] Training complete | in-sample MSE={mse:.4f}')

    # ------------------------------------------------------------------
    # Action Selection: Greedy Argmax
    # ------------------------------------------------------------------

    def sample_action(self, contexts):
        """Select action greedily: pi(x) = argmax_a G_hat(x, a).

        Args:
            contexts: (M, context_dim)

        Returns:
            actions: (M,)  integer action indices.
        """
        num_contexts = contexts.shape[0]
        chunk_size = getattr(self.hparams, 'chunk_size', 500)
        all_actions = []

        for i in range(0, num_contexts, chunk_size):
            batch_ctx = contexts[i: i + chunk_size]
            B = batch_ctx.shape[0]
            preds = []
            for a in range(self.hparams.num_actions):
                actions_tmp = jnp.ones(shape=(B,)) * a
                f_a = self.nn.out(self.nn.params, batch_ctx, actions_tmp).ravel()  # (B,)
                preds.append(f_a.reshape(-1, 1))
            pred_matrix = jnp.hstack(preds)   # (B, num_actions)
            all_actions.append(jnp.argmax(pred_matrix, axis=1))

        return jnp.concatenate(all_actions, axis=0)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_model(self, path):
        """Save network weights to a pickle file."""
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
        """Load network weights from a pickle file."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.nn.params = data['nn_params']
        self.nn.opt_state = data['nn_opt_state']
        if self.hparams.verbose:
            print(f'[{self.name}] Model loaded from {path}')

    # ------------------------------------------------------------------
    # Compatibility Stubs (online interface, unused in offline mode)
    # ------------------------------------------------------------------

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


class RiskExactNeuraLCBV2(ExactNeuraLCBV2):
    """
    Upgraded ExactNeuraLCBV2 with Tofu Loss (reward clipping) and Risk-Aware Action Selection.
    This exactly implements the formula: argmax { rho(F) - L * R(pi) }
    """
    def __init__(self, hparams, update_freq=1, name='RiskExactNeuraLCBV2'):
        super().__init__(hparams, update_freq, name)
        self.historical_residuals = None
        self.rho_residuals = None

    def _compute_risk_functional(self, Y):
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
        measure = getattr(self.hparams, 'risk_measure', 'cvar')
        if measure == 'cvar':
            tail_prob = self.hparams.alpha if self.hparams.alpha < 0.5 else (1.0 - self.hparams.alpha)
            return 1.0 / tail_prob
        return 1.0

    def update(self, contexts, actions, rewards):
        # 1. Tofu Loss: Clip rewards to tau_n to avoid outlier gradient explosions
        tau_n = getattr(self.hparams, 'tau_n', 1e6)
        r_tilde = jnp.where(jnp.abs(rewards) <= tau_n, rewards, tau_n * jnp.sign(rewards))

        self.data.add(contexts, actions, r_tilde)
        self.nn.train(self.data, self.hparams.num_steps)

        # 2. Risk Estimation: Calculate empirical residuals on RAW rewards
        f_hist = self.nn.out(self.nn.params, contexts, actions).ravel()
        self.historical_residuals = rewards.ravel() - f_hist
        self.rho_residuals = self._compute_risk_functional(self.historical_residuals)

        # 3. Update Confidence Matrix (Lambda_inv)
        u = self.nn.grad_out(self.nn.params, contexts, actions) / jnp.sqrt(self.nn.m)
        for i in range(contexts.shape[0]):
            self.Lambda_inv = self.Lambda_inv.at[actions[i]].set(
                inv_sherman_morrison_single_sample(u[i,:], self.Lambda_inv[actions[i],:,:])
            )

    def sample_action(self, contexts):
        assert self.rho_residuals is not None, "Call update() first."
        cs = getattr(self.hparams, 'chunk_size', 500)
        num_chunks = math.ceil(contexts.shape[0] / cs)
        acts = []
        L_rho = self._get_risk_lipschitz_factor()

        for i in range(num_chunks):
            ctxs = contexts[i * cs: (i+1) * cs,:] 
            lcb = []
            for a in range(self.hparams.num_actions):
                actions_tmp = jnp.ones(shape=(ctxs.shape[0],)) * a 

                f = self.nn.out(self.nn.params, ctxs, actions_tmp) 
                g = self.nn.grad_out(self.nn.params, ctxs, actions_tmp) / jnp.sqrt(self.nn.m)
                gA = g @ self.Lambda_inv[a,:,:] 
                
                gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) 
                cnf = jnp.sqrt(gAg) 

                # ========================================================
                # TRANSITION FROM MEAN TO RISK-AWARE: rho(F) - L * R(pi)
                # ========================================================
                # 1. Calculate risk of the predicted distribution (Translation Invariance)
                risk_a = f.ravel() + self.rho_residuals
                
                # 2. Calculate scaled uncertainty penalty (Lipschitz bounded)
                penalty = self.hparams.beta * L_rho * cnf.ravel()
                
                # 3. Final Risk-Aware LCB
                lcb_a = risk_a - penalty
                lcb.append(lcb_a.reshape(-1,1)) 
                
            lcb = jnp.hstack(lcb) 
            acts.append(jnp.argmax(lcb, axis=1)) 
        return jnp.hstack(acts)

    def sample_action_milp(self, contexts, noise_samples=None):
        """Chooses actions by globally maximizing Marginal CVaR using MILP.
        
        Unlike `sample_action` which optimizes conditional point-wise CVaR, 
        this computes the global optimal policy coupling all contexts together.
        """
        import gurobipy as gp
        from gurobipy import GRB
        import numpy as np
        
        assert self.rho_residuals is not None, "Call update() first."
        
        if noise_samples is None:
            noise_samples = np.array(self.historical_residuals)
            
        I = contexts.shape[0]
        J = self.hparams.num_actions
        M = len(noise_samples)
        
        if getattr(self.hparams, 'verbose', False):
            print(f'[{self.name}] Extracting mu and R matrices for MILP (I={I}, J={J}, M={M})...')
            
        mu_matrix = np.zeros((I, J))
        R_matrix = np.zeros((I, J))
        L_rho = self._get_risk_lipschitz_factor()
        chunk_size = getattr(self.hparams, 'chunk_size', 500)
        
        for a in range(J):
            mu_a_full = []
            R_a_full = []
            for i in range(0, I, chunk_size):
                batch_contexts = contexts[i : i + chunk_size]
                B = batch_contexts.shape[0]
                actions_tmp = jnp.ones(shape=(B,)) * a
                
                # Mean Prediction
                mu_a = self.nn.out(self.nn.params, batch_contexts, actions_tmp).ravel()
                
                # Uncertainty with Action-Specific Lambda_inv
                g = self.nn.grad_out(self.nn.params, batch_contexts, actions_tmp) / jnp.sqrt(self.nn.m)
                gA = g @ self.Lambda_inv[a, :, :] 
                gAg = jnp.sum(jnp.multiply(gA, g), axis=-1) 
                R_a = jnp.sqrt(gAg)
                
                mu_a_full.append(np.array(mu_a))
                R_a_full.append(np.array(R_a))
                
            mu_matrix[:, a] = np.concatenate(mu_a_full)
            R_matrix[:, a] = np.concatenate(R_a_full)
            
        if getattr(self.hparams, 'verbose', False):
            print(f'[{self.name}] Solving Exact Marginal CVaR LCB MILP...')
            
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 1 if getattr(self.hparams, 'verbose', False) else 0)
        env.start()
        model = gp.Model("RiskExact_CVaR_LCB_MILP", env=env)
        
        z = model.addVars(I, J, vtype=GRB.BINARY, name="z")
        q = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="q")
        s = model.addVars(I, M, vtype=GRB.CONTINUOUS, lb=0.0, name="s")
        
        alpha = getattr(self.hparams, 'alpha', 0.9)
        tail_prob = alpha if alpha < 0.5 else (1.0 - alpha)
        beta = self.hparams.beta
        
        cvar_expr = q - (1.0 / (I * M * tail_prob)) * gp.quicksum(s[i, m] for i in range(I) for m in range(M))
        penalty_expr = (beta * L_rho / float(I)) * gp.quicksum(R_matrix[i, j] * z[i, j] for i in range(I) for j in range(J))
        
        model.setObjective(cvar_expr - penalty_expr, GRB.MAXIMIZE)
        
        for i in range(I):
            model.addConstr(gp.quicksum(z[i, j] for j in range(J)) == 1)
            
        for i in range(I):
            for m in range(M):
                mu_z_sum = gp.quicksum(mu_matrix[i, j] * z[i, j] for j in range(J))
                model.addConstr(s[i, m] >= q - (mu_z_sum + noise_samples[m]))
                
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            optimal_policy = np.zeros(I, dtype=int)
            for i in range(I):
                for j in range(J):
                    if z[i, j].X > 0.5:
                        optimal_policy[i] = j
                        break
            if getattr(self.hparams, 'verbose', False):
                print(f'[{self.name}] MILP solved. Max CVaR LCB = {model.ObjVal:.4f}')
            return jnp.array(optimal_policy)
        else:
            print(f'[{self.name}] MILP failed. Fallback to point-wise LCB.')
            return self.sample_action(contexts)
