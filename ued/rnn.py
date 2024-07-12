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

class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs
        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((obs, dones), initial_carry=hidden)

        actor_mean = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor_mean)
        pi = jax.nn.softmax(actor_mean)

        # jax.tree_map(jnp.squeeze, hidden)
        return hidden, pi
    
    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))
    

class Critic(nn.Module):

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs

        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((obs, dones), initial_carry=hidden)

        critic = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)

        return hidden, jnp.squeeze(critic, axis=-1)
    
    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))
    
def eval_agent(rng, rollout_manager, env_params, actor_train_state, num_workers, init_hstate):
    """Evaluate episodic agent performance over multiple workers."""
    rng, _rng = jax.random.split(rng)
    env_obs, env_state = rollout_manager.batch_reset(_rng, env_params, num_workers)
    rng, _rng = jax.random.split(rng)
    _, _, _, tot_reward = rollout_manager.batch_rollout(
        _rng, actor_train_state, env_params, env_obs, env_state, init_hstate, eval=True
    )
    return tot_reward.mean()

def _get_policy_model(n_actions):
    return Actor(n_actions)

def _get_critic_model():
    return Critic()