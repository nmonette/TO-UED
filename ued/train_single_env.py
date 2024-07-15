# Some code taken from https://github.com/DramaCow/jaxued/blob/main/examples/maze_plr.py
# Some code taken from https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py

import jax
import jax.numpy as jnp
import chex
from flax.training.train_state import TrainState

from util.data import Transition
from agents.agents import compute_adv_target
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
        # --- Forward pass through policy network ---
        _, all_action_probs = jax.vmap(actor_state.apply_fn, in_axes=(None, 0, 0))({"params": actor_params}, (rollout.obs, rollout.done), hstate)
        entropy = jax.scipy.special.entr(all_action_probs).sum(-1).mean() 
        pi = jax.vmap(selected_action_probs)(all_action_probs, rollout.action)
        lp = jnp.log(pi)

        # --- Forward pass through value network ---    
        _, values_pred = jax.vmap(critic_state.apply_fn, in_axes=(None, 0, 0))({"params": critic_params}, (rollout.obs, rollout.done), hstate)
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
            - entropy_coeff * entropy
        )

        metrics = {
            "ppo_loss": loss,
            "ppo_value_loss": value_loss,
            "ppo_policy_loss": loss_actor,
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
    entropy_coeff: float, 
    hstate,
    value_hstate,
    init_obs,
    init_state,
):
    # --- Perform Rollouts ---
    rng, rollout_rng = jax.random.split(rng)
    rollout, end_obs, end_state, end_hstate, end_value_hstate, _ = rollout_manager.batch_rollout_single_env(
        rollout_rng, actor_state, critic_state, env_params, init_obs, init_state, hstate, value_hstate
    )

    # --- Compute values, advantages, and targets ---
    hstate = Actor.initialize_carry(init_state.time.shape)
    value_fn = partial(compute_adv_target, gamma=gamma, gae_lambda=gae_lambda)
    adv, values, target = jax.vmap(value_fn)(values=rollout.value, rollout=rollout) 
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
    return (actor_state, critic_state, end_hstate, end_value_hstate, end_obs, end_state), jax.tree_util.tree_map(jnp.mean, metrics)

def train_eval_agent(
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
    entropy_coeff: float,
    num_steps: int
):
    
    train_agent_fn = partial(
        train_agent,
        env_params=env_params,
        rollout_manager=rollout_manager,
        num_epochs=num_epochs,
        num_mini_batches=num_mini_batches,
        num_workers=num_workers,
        gamma=gamma,
        gae_lambda=gae_lambda, 
        clip_eps=clip_eps,
        critic_coeff=critic_coeff,
        entropy_coeff=entropy_coeff
    )
    

    def loop(carry, _):
        rng, actor_state, critic_state = carry

        rng, _rng = jax.random.split(rng)
        actor_state, critic_state, metrics = train_agent_fn(
            _rng, actor_state, critic_state
        )

        return (rng, actor_state, critic_state), metrics

        
    carry_out, metrics = jax.lax.scan(
        loop, (rng, actor_state, critic_state), None, num_steps
    )
    _, actor_state, critic_state, = carry_out

    return (actor_state, critic_state), metrics

    



