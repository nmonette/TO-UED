# Some code taken from https://github.com/DramaCow/jaxued/blob/main

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from typing import Any, Optional, Tuple, Sequence
from functools import partial

Carry = Any
Output = Any

class ResetRNN(nn.Module):
    """This is a wrapper around an RNN that automatically resets the hidden state upon observing a `done` flag. In this way it is compatible with the jax-style RL loop where episodes automatically end/restart.
    """
    cell: nn.RNNCellBase

    @nn.compact
    def __call__(
        self,
        inputs: Tuple[jax.Array, jax.Array],
        *,
        initial_carry: Optional[Carry] = None,
        reset_carry: Optional[Carry] = None,
    ) -> Tuple[Carry, Output]:
        # On episode completion, model resets to this
        if reset_carry is None:
            reset_carry = self.cell.initialize_carry(jax.random.PRNGKey(0), inputs[0].shape[1:])
        carry = initial_carry if initial_carry is not None else reset_carry

        def scan_fn(cell, carry, inputs):
            x, resets = inputs
            
            # --- Reset hidden state if environment is reset ---
            carry = jax.tree_map(
                lambda a, b: jnp.where(resets.reshape(-1, 1), a, b).squeeze(), reset_carry, carry
            )
            return cell(carry, x)

        scan = nn.scan(
            scan_fn,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )
        return scan(self.cell, carry, inputs)
    
# TODO: Convolutional Actor and Critic, also weight sharing

class ActorCritic(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs
        
        img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)
        
        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(5, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="scalar_embed")(dir_embed)
        
        embedding = jnp.append(img_embed, dir_embed, axis=-1)

        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)

        actor_mean = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor_mean)
        pi = jax.nn.softmax(actor_mean)

        critic = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs
        # --- Feature extraction ---
        img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)
        
        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(5, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="scalar_embed")(dir_embed)

        embedding = jnp.append(img_embed, dir_embed.reshape(-1, 5), axis=-1)

        # --- Do a scan through the recurrent layer ---
        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)

        # --- Pass through the rest of the feedforward layers ---
        actor_mean = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor_mean)
        pi = jax.nn.softmax(actor_mean)

        return hidden, pi
    
    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))
    

class Critic(nn.Module):

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs

        # --- Feature extraction ---
        img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)
        
        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(5, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="scalar_embed")(dir_embed)
        
        embedding = jnp.append(img_embed, dir_embed.reshape(-1, 5), axis=-1)

        # --- Do a scan through the recurrent layer ---
        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)

        # --- Pass through the rest of the feedforward layers ---
        critic = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)

        return hidden, jnp.squeeze(critic, axis=-1)
    
    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))
    
def eval_agent(rng, rollout_manager, env_params, actor_train_state, num_workers, init_hstate):
    """
        Evaluate episodic agent performance over multiple workers.
        This version takes hstate as an argument.
    """
    rng, _rng = jax.random.split(rng)
    env_obs, env_state = rollout_manager.batch_reset_single_env(_rng, env_params, num_workers)
    rng, _rng = jax.random.split(rng)
    _, _, _, _, _, tot_reward = rollout_manager.batch_rollout_single_env(
        _rng, actor_train_state, None, env_params, env_obs, env_state, init_hstate, init_hstate, eval=True
    )
    return tot_reward.mean()

def eval_agent_nomean(rng, rollout_manager, env_params, actor_train_state, num_workers, init_hstate):
    """
        Evaluate episodic agent performance over multiple workers.
        This version takes hstate as an argument.
    """
    rng, _rng = jax.random.split(rng)
    env_obs, env_state = rollout_manager.batch_reset_single_env(_rng, env_params, num_workers)
    rng, _rng = jax.random.split(rng)
    _, _, _, _, _, tot_reward = rollout_manager.batch_rollout_single_env(
        _rng, actor_train_state, None, env_params, env_obs, env_state, init_hstate, init_hstate, eval=True
    )
    return tot_reward


def _get_policy_model(n_actions):
    return Actor(n_actions)

def _get_critic_model():
    return Critic()