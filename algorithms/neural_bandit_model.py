"""Define  neural network models for (contextual) bandits.  

This model takes a context as input and predicts the rewards of all actions:

                            f(x,a) = < phi(x), w_a > 
"""

import numpy as np 
from core.nn import NeuralNetwork 
from core.utils import action_convolution 
import haiku as hk
import jax 
import jax.numpy as jnp
import optax
from tqdm import tqdm

class NeuralBanditModel(NeuralNetwork):
    """Build a neural network model for bandits.
    
    This model takes a context as input and predict the expected rewards of multiple actions. 
    """

    def __init__(self, optimizer, hparams, name):
        self.optimizer = optimizer 
        self.hparams = hparams 
        self.name = name 
        self.m = min(self.hparams.layer_sizes)
        self.build_model()
        print('{} has {} parameters.'.format(name, self.num_params))

    def build_model(self):
        """Transform impure functions into pure functions and apply JAX tranformations."""

        self.nn = hk.without_apply_rng(hk.transform(self.net_impure_fn))
        self.out = jax.jit(self.out_impure_fn) 
        self.grad_out = jax.jit(self.grad_out_impure_fn)
        self.loss = jax.jit(self.loss_impure_fn)
        self.update = jax.jit(self.update_impure_fn)

        # Initialize network parameters and opt states. 
        self.init()

        # Compute number of network parameters 
        p = (self.hparams.context_dim + 1) * self.hparams.layer_sizes[0] 
        for i in range(1, len(self.hparams.layer_sizes)):
            p += (self.hparams.layer_sizes[i-1] + 1) * self.hparams.layer_sizes[i]
        p += self.hparams.layer_sizes[-1] + 1 
        if self.hparams.layer_n: 
            p += sum(2 * self.hparams.layer_sizes)
        self.num_params = p 

    def net_impure_fn(self, context):
        """
        Args:
            context: context, (None, self.hparams.context_dim)
        """
        net_structure = []
        for num_units in self.hparams.layer_sizes:
            net_structure.append(
                hk.Linear(
                    num_units, w_init=hk.initializers.UniformScaling(self.hparams.s_init) 
                    )
                ) 
            if self.hparams.layer_n: 
                net_structure.append(hk.LayerNorm(axis=1, create_scale=True, create_offset=True))

            net_structure.append(self.hparams.activation) 

        net_structure.append(
                hk.Linear(self.hparams.num_actions, w_init=hk.initializers.UniformScaling(self.hparams.s_init) )
            )
        
        mlp = hk.Sequential(net_structure) 

        return mlp(context)
    
    def out_impure_fn(self, params, context):
        return self.nn.apply(params, context)

    def grad_out_impure_fn(self, params, context):
        """Memory efficient per-sample gradients with chunking.
        
        Return:
            action_grad_params: (num_actions, num_samples, p)
        """
        num_samples = context.shape[0]
        chunk_size = 1000
        all_action_grads = [[] for _ in range(self.hparams.num_actions)]
        
        for i in range(0, num_samples, chunk_size):
            c_chunk = context[i : i + chunk_size]
            
            def single_sample_grad(c):
                return jax.jacobian(lambda p: self.nn.apply(p, c[None, :]).ravel())(params)
            
            jac_tree_chunk = jax.vmap(single_sample_grad)(c_chunk)
            
            for a in range(self.hparams.num_actions):
                batch_leaf_flat = []
                for leaf in jax.tree_util.tree_leaves(jac_tree_chunk):
                    batch_leaf_flat.append(leaf[:, a, :].reshape(c_chunk.shape[0], -1))
                all_action_grads[a].append(jnp.hstack(batch_leaf_flat))
        
        final_action_grads = [jnp.concatenate(act_list, axis=0) for act_list in all_action_grads]
        return jnp.array(final_action_grads)


    def loss_impure_fn(self, params, context, action, reward):
        """
        Args:
            context: An array of context, (None, self.hparams.context_dim)
            action: An array of one-hot action vectors, 1 for selected action and 0 other wise, (None, self.hparams.num_actions)
            reward: An array of reward vectors, (None, self.hparams.num_actions)
        """
        preds = self.out(params, context) 

        squared_loss = 0.5 * jnp.mean(jnp.sum(action * jnp.square(preds - reward), axis=1), axis=0)
        reg_loss = 0.5 * self.hparams.lambd * sum(
                jnp.sum(jnp.square(param)) for param in jax.tree_util.tree_leaves(params) 
            )

        return squared_loss + reg_loss 

    def update_impure_fn(self, params, opt_state, context, action, reward):
        """
        Args:
            context: An array of context, (None, self.hparams.context_dim)
            action: An array of one-hot action vectors, 1 for selected action and 0 other wise, (None, self.hparams.num_actions)
            reward: An array of reward vectors, (None, self.hparams.num_actions)
        """
        grads = jax.grad(self.loss)(params, context, action, reward)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    def init(self): 
        key = jax.random.PRNGKey(self.hparams.seed)
        key, subkey = jax.random.split(key)

        context = jax.random.normal(key, (1, self.hparams.context_dim))
        self.params = self.nn.init(subkey, context) 
        self.opt_state = self.optimizer.init(self.params)

    def train(self, data, num_steps):
        if self.hparams.verbose:
            print('Training {} for {} steps.'.format(self.name, num_steps)) 
            pbar = tqdm(total=num_steps, desc=f"Training {self.name}", unit="steps")
            
        params, opt_state = self.params, self.opt_state 
        for step in range(num_steps):
            x,w,y = data.get_batch_with_weights(self.hparams.batch_size) #(None,d), (None, num_actions), (None, num_actions)
            # print('DEBUG', x.shape, w.shape, y.shape)
            params, opt_state = self.update(params, opt_state, x,w,y) 

            if step % self.hparams.freq_summary == 0 and self.hparams.verbose:
                cost = self.loss(params, x,w,y)
                pbar.set_postfix({"loss": f"{cost:.4f}"})

            if self.hparams.verbose:
                pbar.update(1)
        
        if self.hparams.verbose:
            pbar.close()
        
        self.params, self.opt_state = params, opt_state

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({'params': self.params, 'opt_state': self.opt_state}, f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.params = data['params']
            self.opt_state = data['opt_state']


class NeuralBanditModelV2(NeuralBanditModel):
    """Build a neural network model V2 for bandits.
    
    This model takes an action-convoluted context as input and predict the expected reward of the context. 
    """

    def __init__(self, optimizer, hparams, name='NeuralBanditModelV2'):
        self.optimizer = optimizer 
        self.hparams = hparams 
        self.name = name 
        self.m = min(self.hparams.layer_sizes)
        self.build_model()
        print('{} has {} parameters.'.format(name, self.num_params))

    def build_model(self):
        """Transform impure functions into pure functions and apply JAX tranformations."""

        self.nn = hk.without_apply_rng(hk.transform(self.net_impure_fn))
        self.out = jax.jit(self.out_impure_fn) 
        self.grad_out = jax.jit(self.grad_out_impure_fn)
        self.action_convolution = jax.jit(self.action_convolution_impure_fn)
        self.phi_and_out = jax.jit(hk.without_apply_rng(hk.transform(self.phi_and_out_impure_fn)).apply)
        self.loss = jax.jit(self.loss_impure_fn)
        self.update = jax.jit(self.update_impure_fn)

        # Initialize network parameters and opt states. 
        self.init(self.hparams.seed)

        # Compute number of network parameters accurately from initialized params
        self.num_params = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
    def reset(self, seed):
        self.init(seed)

    def net_impure_fn(self, contexts, actions):
        """Returns only the scalar reward prediction."""
        phi, out = self.phi_and_out_impure_fn(contexts, actions)
        return out

    def phi_and_out_impure_fn(self, contexts, actions):
        """Returns both last layer features (phi) and the prediction."""
        convoluted_contexts = self.action_convolution_impure_fn(contexts, actions)
        
        x = convoluted_contexts
        for i, num_units in enumerate(self.hparams.layer_sizes):
            x = hk.Linear(num_units, name=f"layer_{i}")(x)
            if self.hparams.layer_n:
                x = hk.LayerNorm(axis=1, create_scale=True, create_offset=True)(x)
            x = self.hparams.activation(x)
        
        phi = x # Last hidden layer
        out = hk.Linear(1, name="output_layer")(phi)
        return phi, out

    def out_impure_fn(self, params, contexts, actions):
        return self.nn.apply(params, contexts, actions)

    def grad_out_impure_fn(self, params, contexts, actions):
        """Memory efficient per-sample gradients with chunking.
        
        Return:
            grad_params: (num_samples, p)
        """
        num_samples = contexts.shape[0]
        chunk_size = 100
        all_grads = []
        
        def single_sample_grad(c, a):
            return jax.grad(lambda p: self.nn.apply(p, c[None, :], a[None]).ravel()[0])(params)

        for i in range(0, num_samples, chunk_size):
            c_chunk = contexts[i : i + chunk_size]
            a_chunk = actions[i : i + chunk_size]
            
            per_sample_grads = jax.vmap(single_sample_grad)(c_chunk, a_chunk)
            
            flat_grads = []
            for leaf in jax.tree_util.tree_leaves(per_sample_grads):
                flat_grads.append(leaf.reshape(c_chunk.shape[0], -1))
            all_grads.append(jnp.hstack(flat_grads))
        
        return jnp.concatenate(all_grads, axis=0)

    def action_convolution_impure_fn(self, contexts, actions):
        return action_convolution(contexts, actions, self.hparams.num_actions)

    def loss_impure_fn(self, params, contexts, actions, rewards):
        """
        Args:
            contexts: An array of context, (None, self.hparams.context_dim)
            actions: An array of actions, (None,)
            rewards: An array of rewards for the chosen actions, (None,)
        """
        preds = self.out(params, contexts, actions) 

        squared_loss = 0.5 * jnp.mean( jnp.square(preds.ravel() - rewards.ravel()) )
        reg_loss = 0.5 * self.hparams.lambd * sum(
                jnp.sum(jnp.square(param)) for param in jax.tree_util.tree_leaves(params) 
            )

        return squared_loss + reg_loss 

    def init(self, seed): 
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)

        context = jax.random.normal(key, (1, self.hparams.context_dim))
        action = jax.random.randint(subkey, shape=(1,), minval=0, maxval=self.hparams.num_actions) 

        self.params = self.nn.init(subkey, context, action) 
        self.opt_state = self.optimizer.init(self.params)

    
    def update_impure_fn(self, params, opt_state, contexts, actions, rewards):
        """
        Args:
            contexts: An array of contexts, (None, self.hparams.context_dim)
            actions: An array of actions, (None, )
            rewards: An array of rewards for the chosen actions, (None,)
        """
        grads = jax.grad(self.loss)(params, contexts, actions, rewards)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    
    def train(self, data, num_steps):
        if self.hparams.verbose:
            print('Training {} for {} steps.'.format(self.name, num_steps)) 
            pbar = tqdm(total=num_steps, desc=f"Training {self.name}")

        params, opt_state = self.params, self.opt_state 
        for step in range(num_steps):
            x,a,y = data.get_batch(self.hparams.batch_size, self.hparams.data_rand) #(None,d), (None,), (None,)
            params, opt_state = self.update(params, opt_state, x,a,y) 

            if step % self.hparams.freq_summary == 0 and self.hparams.verbose:
                cost = self.loss(params, x,a,y)
                pbar.set_postfix({"loss": f"{cost:.4f}"})
            
            if self.hparams.verbose:
                pbar.update(1)
        
        if self.hparams.verbose:
            pbar.close()
        
        self.params, self.opt_state = params, opt_state


class QuantileNeuralBanditModel(NeuralBanditModelV2):
    """Neural Network model for Quantile Regression in Bandits."""

    def __init__(self, optimizer, hparams, name='QuantileNeuralBanditModel'):
        self.num_quantiles = getattr(hparams, 'num_quantiles', 100)
        self.kappa = getattr(hparams, 'huber_kappa', 1.0)
        super().__init__(optimizer, hparams, name)

    def build_model(self):
        """Transform impure functions into pure functions and apply JAX tranformations."""
        self.nn = hk.without_apply_rng(hk.transform(self.net_impure_fn))
        self.out = jax.jit(self.out_impure_fn)
        # We need both quantiles and features
        self.phi_and_out = jax.jit(hk.without_apply_rng(hk.transform(self.phi_and_out_impure_fn)).apply)
        self.action_convolution = jax.jit(self.action_convolution_impure_fn)
        self.loss = jax.jit(self.loss_impure_fn)
        self.update = jax.jit(self.update_impure_fn)

        # Initialize
        self.init(self.hparams.seed)
        self.num_params = sum(x.size for x in jax.tree_util.tree_leaves(self.params))

    def net_impure_fn(self, contexts, actions):
        """Returns only the quantiles."""
        phi, quantiles = self.phi_and_out_impure_fn(contexts, actions)
        return quantiles

    def phi_and_out_impure_fn(self, contexts, actions):
        """Returns both last layer features (phi) and quantiles."""
        convoluted_contexts = self.action_convolution_impure_fn(contexts, actions)
        
        x = convoluted_contexts
        for i, num_units in enumerate(self.hparams.layer_sizes):
            x = hk.Linear(num_units, name=f"layer_{i}")(x)
            if self.hparams.layer_n:
                x = hk.LayerNorm(axis=1, create_scale=True, create_offset=True)(x)
            x = self.hparams.activation(x)
        
        phi = x # Last hidden layer
        quantiles = hk.Linear(self.num_quantiles, name="quantile_output")(phi)
        return phi, quantiles

    def loss_impure_fn(self, params, contexts, actions, rewards):
        """Quantile Huber Loss."""
        quantiles = self.out(params, contexts, actions) # (batch, num_quantiles)
        rewards = rewards.reshape(-1, 1) # (batch, 1)
        
        diff = rewards - quantiles # (batch, num_quantiles)
        abs_diff = jnp.abs(diff)
        
        # Huber Loss
        huber_loss = jnp.where(abs_diff <= self.kappa, 
                               0.5 * jnp.square(diff), 
                               self.kappa * (abs_diff - 0.5 * self.kappa))
        
        # Quantile thresholds
        tau = jnp.linspace(1.0 / (2.0 * self.num_quantiles), 
                          1.0 - 1.0 / (2.0 * self.num_quantiles), 
                          self.num_quantiles)
        
        # Quantile Huber Loss: |tau - I(diff < 0)| * huber_loss / kappa
        weight = jnp.abs(tau - (diff < 0).astype(jnp.float32))
        quantile_huber_loss = weight * huber_loss / self.kappa
        
        loss = jnp.mean(jnp.sum(quantile_huber_loss, axis=1))
        
        # L2 Regularization
        reg_loss = 0.5 * self.hparams.lambd * sum(
                jnp.sum(jnp.square(param)) for param in jax.tree_util.tree_leaves(params) 
            )
        return loss + reg_loss
