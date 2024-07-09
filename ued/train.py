# Some code taken from https://github.com/DramaCow/jaxued/blob/main/examples/maze_plr.py

import jax
import jax.numpy as jnp
import chex
from flax.training.train_state import TrainState

from util.data import Transition
from util.jax import mini_batch_vmap
from agents.agents import compute_val_adv_target, eval_agent

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
    entropy_coeff: float
):
    
    gather = lambda p,idx: p[idx]

    def selected_action_probs(all_action_probs, rollout_action):
        all_action_probs += 1e-8
        return gather(all_action_probs, rollout_action)
    
    def loss_fn(actor_params, critic_params):
        all_action_probs = actor_state.apply_fn({"params": actor_params}, rollout.obs)

        pi = jax.vmap(jax.vmap(jax.vmap(selected_action_probs)))(all_action_probs, rollout.action)
        entropy = jax.scipy.special.entr(pi).mean()
        values_pred = critic_state.apply_fn({"params": critic_params}, rollout.obs)
        lp = jnp.log(pi)
        
        ratio = jnp.exp(lp - rollout.log_prob)[..., jnp.newaxis]

        A = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        l_clip = (-jnp.minimum(ratio * A, jnp.clip(ratio, 1-clip_eps, 1 + clip_eps) * A)).mean()
    
        values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
        l_vf = 0.5 * jnp.maximum((values_pred - targets) ** 2, (values_pred_clipped - targets) ** 2).mean()

        loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy

        metrics = {
            "ppo_loss": loss,
            "ppo_value_loss": l_vf,
            "clipped_loss": l_clip,
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
        init_state=init_state
    )

    value_fn = partial(compute_val_adv_target, critic_state, gamma=gamma, gae_lambda=gae_lambda)
    adv, values, target = jax.vmap(jax.vmap(value_fn))(rollout) 
    values = values[:,:,:-1]
    
    agent_train_step_fn = partial(
        agent_train_step,
        clip_eps=clip_eps,
        critic_coeff=critic_coeff,
        entropy_coeff=entropy_coeff
    )

    def epoch(carry, _):
        rng, actor_state, critic_state, rollout, values, adv, target = carry

        def minibatch(carry, data):
            actor_state, critic_state = carry
            rollout, values, adv, target = data

            actor_state, critic_state, metrics = agent_train_step_fn(
                actor_state, 
                critic_state,
                rollout, 
                values,
                adv,
                target
            )
            return (actor_state, critic_state), metrics
        
        rng, _rng = jax.random.split(rng)
        perm = jax.random.permutation(_rng, rollout.obs.shape[1])

        # TODO: fix permutation
        minibatch_fn = lambda x: jnp.take(x, perm, axis=1) \
        .reshape(x.shape[0], num_mini_batches, -1, *x.shape[2:]) \
        .swapaxes(0, 1)

        minibatches = (
            jax.tree_map(minibatch_fn, rollout),
            minibatch_fn(values),
            minibatch_fn(adv),
            minibatch_fn(target),
        )
        (actor_state, critic_state), metrics = jax.lax.scan(
            minibatch, (actor_state, critic_state), minibatches
        )
        
        return (rng, actor_state, critic_state, rollout, values, adv, target), metrics
    
    carry_out, metrics = jax.lax.scan(
        epoch, (rng, actor_state, critic_state, rollout, values, adv, target), None, num_epochs
    )

    actor_state, critic_state = carry_out[1], carry_out[2]

    return actor_state, critic_state, jax.tree_util.tree_map(jnp.mean, metrics)



        
        



    



