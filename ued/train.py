# Some code taken from https://github.com/DramaCow/jaxued/blob/main/examples/maze_plr.py
# Some code taken from https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py

import jax
import jax.numpy as jnp
import chex
from flax.training.train_state import TrainState

from util.data import Transition
from util.jax import mini_batch_vmap, map_reshape, map_swapaxes
from agents.agents import compute_val_adv_target
from .rnn import Actor

from typing import Any
from functools import partial

def agent_train_step(
    actor_state: TrainState,
    critic_state: TrainState,
    rollout: Transition,
    values: jnp.ndarray,
    advantages: jnp.ndarray,
    targets: jnp.ndarray,
    clip_eps: float,
    critic_coeff: float,
    entropy_coeff: float,
    hstate: jnp.ndarray,
):
    def selected_action_probs(all_action_probs, rollout_action):
        all_action_probs += 1e-8
        return jnp.take_along_axis(all_action_probs, rollout_action[..., None], -1).squeeze()
    
    def loss_fn(actor_params, critic_params):
        # --- Swap axes because RNN handles sequence length as dim 0 ---
        obs = map_swapaxes(rollout.obs, 0, 1)
        done = map_swapaxes(rollout.done, 0, 1)

        # --- Forward pass through policy network ---
        _, all_action_probs = actor_state.apply_fn({"params": actor_params}, (obs, done), hstate)
        all_action_probs = all_action_probs.swapaxes(0, 1)
        pi = jax.vmap(selected_action_probs)(all_action_probs, rollout.action)
        entropy = jax.scipy.special.entr(pi).mean()
        lp = jnp.log(pi)

        # --- Forward pass through value network ---    
        _, values_pred = critic_state.apply_fn({"params": critic_params}, (obs, done), hstate)
        values_pred = values_pred.swapaxes(0, 1)

        # --- Calculate value loss ---
        values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
        value_losses = jnp.square(values - targets)
        value_losses_clipped = jnp.square(values_pred_clipped - targets)
        value_loss = (
            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        )

        # --- Calculate actor loss ---
        ratio = jnp.exp(lp - rollout.log_prob)
        A = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        loss_actor1 = ratio * A
        loss_actor2 = (
            jnp.clip(
                ratio,
                1 - clip_eps,
                1 + clip_eps,
            )
            * A
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()

        loss = (
            loss_actor
            + critic_coeff * value_loss
            + entropy_coeff * entropy
        )

        metrics = {
            "ppo_loss": loss,
            "ppo_value_loss": value_loss,
            "unclipped_policy_loss": loss_actor1,
            "clipped_policy_loss": loss_actor2,
            "policy_entropy": entropy
        }

        return loss, metrics
    
    # --- Apply Gradients ---
    grad_fn = jax.grad(loss_fn, has_aux=True, argnums=(0, 1))
    (actor_grad, critic_grad), metrics = grad_fn(actor_state.params, critic_state.params)
    actor_state = actor_state.apply_gradients(grads=actor_grad)
    critic_state = critic_state.apply_gradients(grads=critic_grad)
    return actor_state, critic_state, metrics


def train_agent(
    rng: chex.PRNGKey,
    actor_state: TrainState,
    critic_state: TrainState,
    env_params,
    rollout_manager: Any,
    num_epochs: int,
    num_mini_batches: int,
    num_workers: int,
    gamma: float,
    gae_lambda: float, 
    clip_eps: float,
    critic_coeff: float,
    entropy_coeff: float
):
    # --- Perform Rollouts ---
    rng, reset_rng, rollout_rng = jax.random.split(rng, 3)

    # NOTE: batch_reset has been modified to accept a batch of env_params
    init_obs, init_state = rollout_manager.batch_reset(reset_rng, env_params)
    hstate = Actor.initialize_carry(init_state.time.shape[:-1])
    rollout, _, _, _ = rollout_manager.batch_rollout(
        rollout_rng, actor_state, env_params, init_obs, init_state, hstate
    )

    # --- Compute values, advantages, and targets ---
    value_fn = partial(compute_val_adv_target, critic_state, gamma=gamma, gae_lambda=gae_lambda)
    adv, values, target = jax.vmap(value_fn)(rollout=rollout, hstate=hstate) 
    values = values[:,:-1]
    
    agent_train_step_fn = partial(
        agent_train_step,
        clip_eps=clip_eps,
        critic_coeff=critic_coeff,
        entropy_coeff=entropy_coeff
    )

    def epoch(carry, _):
        rng, actor_state, critic_state, rollout, values, adv, target, hstate = carry

        def minibatch(carry, data):
            actor_state, critic_state = carry
            rollout, values, adv, target, hstate = data

            # --- Perform one update per minibatch ---
            actor_state, critic_state, metrics = agent_train_step_fn(
                actor_state=actor_state, 
                critic_state=critic_state,
                rollout=rollout, 
                values=values,
                advantages=adv,
                targets=target,
                hstate=hstate
            )
            return (actor_state, critic_state), metrics
        
        rng, _rng = jax.random.split(rng)
        perm = jax.random.permutation(_rng, rollout.action.shape[0])

        minibatch_fn = lambda x: jnp.take(
            x.reshape(-1, *x.shape[1:]), perm, axis=0) \
            .reshape(num_mini_batches, -1, *x.shape[1:])
        
        # --- Shuffle data and sort into minibatches ---
        minibatches = (
            jax.tree_util.tree_map(minibatch_fn, rollout),
            minibatch_fn(values),
            minibatch_fn(adv),
            minibatch_fn(target),
            jax.tree_util.tree_map(minibatch_fn, hstate)
        )

        (actor_state, critic_state), metrics = jax.lax.scan(
            minibatch, (actor_state, critic_state), minibatches
        ) 

        return (rng, actor_state, critic_state, rollout, values, adv, target, hstate), metrics
    
    carry_out, metrics = jax.lax.scan(
        epoch, (rng, actor_state, critic_state, rollout, values, adv, target, hstate), None, num_epochs
    )

    actor_state, critic_state = carry_out[1], carry_out[2]

    return actor_state, critic_state, jax.tree_util.tree_map(jnp.mean, metrics)



        
        



    



