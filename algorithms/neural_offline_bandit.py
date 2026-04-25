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
            # For CVaR at level alpha, the Lipschitz constant is 1/alpha.
            return 1.0 / self.hparams.alpha
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
