# Some code taken from https://github.com/DramaCow/jaxued/blob/main/examples/maze_plr.py
# Some code taken from https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py

import jax
import jax.numpy as jnp
import chex
from flax.training.train_state import TrainState

from util.data import Transition
from util.jax import mini_batch_vmap
from agents.agents import compute_val_adv_target
from .rnn import Actor, Critic


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
    actor_hstate: jnp.ndarray,
):
    gather = lambda p,idx: p[idx]
    def selected_action_probs(all_action_probs, rollout_action):
        all_action_probs += 1e-8
        return gather(all_action_probs, rollout_action)
    
    def loss_fn(actor_params, critic_params):
        # --- Forward pass through policy network ---
        _, all_action_probs = actor_state.apply_fn({"params": actor_params}, (rollout.obs.reshape(1, *rollout.obs.shape), rollout.done.reshape(1, *rollout.done.shape)), actor_hstate)

        pi = jax.vmap(selected_action_probs)(all_action_probs, rollout.action)
        entropy = jax.scipy.special.entr(pi).mean()
        lp = jnp.log(pi)

        # --- Forward pass through value network ---    
        _, values_pred = critic_state.apply_fn({"params": critic_params}, (rollout.obs.reshape(1, *rollout.obs.shape), rollout.done.reshape(1, *rollout.done.shape)), actor_hstate)

        # --- Calculate value loss ---
        values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
        value_losses = jnp.square(values - targets)
        value_losses_clipped = jnp.square(values_pred_clipped - targets)
        value_loss = (
            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        )

        # --- Calculate actor loss ---
        ratio = jnp.exp(lp - rollout.log_prob)[..., jnp.newaxis]
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
    rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
    reset_rng = jax.random.split(reset_rng, env_params.start_pos.shape[0])
    init_obs, init_state = jax.vmap(rollout_manager.batch_reset, in_axes=(0, 0, None))(
        reset_rng, env_params, num_workers
    )

    rollout_rng = jax.random.split(rollout_rng, env_params.start_pos.shape[0])
    rollout_fn = partial(rollout_manager.batch_rollout, train_state=actor_state)
    rollout, _, _, _ = mini_batch_vmap(rollout_fn, num_mini_batches=num_mini_batches)(
        rng=rollout_rng,
        env_params=env_params,
        init_obs=init_obs,
        init_state=init_state,
        init_hstates= Actor.initialize_carry(init_obs.shape[:-1])
    )
    critic_hstate = Critic.initialize_carry(rollout.obs.shape[:-2])
    value_fn = partial(compute_val_adv_target, critic_state, gamma=gamma, gae_lambda=gae_lambda)
    adv, values, target = jax.vmap(jax.vmap(value_fn))(rollout=rollout, hstate=critic_hstate) 
    values = values[:,:,:-1]

    print(rollout.obs.shape)

    actor_hstate = Actor.initialize_carry(rollout.obs.shape[:-1])
    
    agent_train_step_fn = partial(
        agent_train_step,
        clip_eps=clip_eps,
        critic_coeff=critic_coeff,
        entropy_coeff=entropy_coeff
    )

    def epoch(carry, _):
        rng, actor_state, critic_state, rollout, values, adv, target, actor_hstate = carry

        def minibatch(carry, data):
            actor_state, critic_state = carry
            rollout, values, adv, target, actor_hstate = data

            actor_state, critic_state, metrics = agent_train_step_fn(
                actor_state=actor_state, 
                critic_state=critic_state,
                rollout=rollout, 
                values=values,
                advantages=adv,
                targets=target,
                actor_hstate=actor_hstate
            )
            return (actor_state, critic_state), metrics
        
        rng, _rng = jax.random.split(rng)
        perm = jax.random.permutation(_rng, rollout.obs.shape[0] * rollout.obs.shape[1] * rollout.obs.shape[2])

        minibatch_fn = lambda x: jnp.take(
            x.reshape(-1, *x.shape[3:]),
            perm,
            axis=0
        ).reshape(num_mini_batches, -1, *x.shape[3:])

        minibatches = (
            jax.tree_util.tree_map(minibatch_fn, rollout),
            minibatch_fn(values),
            minibatch_fn(adv),
            minibatch_fn(target),
            jax.tree_util.tree_map(minibatch_fn, actor_hstate)
        )
        (actor_state, critic_state), metrics = jax.lax.scan(
            minibatch, (actor_state, critic_state), minibatches
        )
        
        return (rng, actor_state, critic_state, rollout, values, adv, target, actor_hstate), metrics
    
    carry_out, metrics = jax.lax.scan(
        epoch, (rng, actor_state, critic_state, rollout, values, adv, target, actor_hstate), None, num_epochs
    )

    actor_state, critic_state = carry_out[1], carry_out[2]

    return actor_state, critic_state, jax.tree_util.tree_map(jnp.mean, metrics)



        
        



    



