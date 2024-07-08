# TODO: create_train_state function X - this is just create_agent
# TODO: make_lpg_train_step function
# TODO: train a model that is compatible 
#       with TrainState and AgentState

# Some code taken from https://github.com/DramaCow/jaxued/blob/main/examples/maze_plr.py

import jax
import jax.numpy as jnp
import chex
from flax.training.train_state import TrainState

from util.data import Transition
from util.jax import gather
from agents.agents import compute_val_adv_target

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
    
    def selected_action_probs(all_action_probs, rollout_action):
        all_action_probs += 1e-8
        return gather(all_action_probs, rollout_action)
    
    def loss_fn(actor_params, critic_params):
        all_action_probs = actor_state.apply_fn({"params": actor_params}, rollout.obs)
        pi = jax.vmap(selected_action_probs)(all_action_probs, rollout.action)
        entropy = pi.entropy().mean()
        values_pred = critic_state.apply_fn({"params": critic_params}, rollout.obs)

        lp = jnp.log(pi)
        ratio = jnp.exp(lp - rollout.log_probs)
        A = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        l_clip = (-jnp.minimum(ratio * A, jnp.clip(ratio, 1-clip_eps, 1 + clip_eps) * A)).mean()
    
        values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
        l_vf = 0.5 * jnp.maximum((values_pred - targets) ** 2, (values_pred_clipped - targets) ** 2).mean()

        loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy

        return loss, (loss, l_vf, l_clip, entropy)
    
    grad_fn = jax.grad(loss_fn, has_aux=True)
    grads, metrics = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, metrics


def train_agent(
    rng: chex.PRNGKey,
    actor_state: TrainState,
    critic_state: TrainState,
    rollout_manager: Any,
    num_train_steps: int,
    num_workers: int,
    agent_target_coeff: float,
    env_params,
    gamma: float,
    gae_lambda: float
):
    rng, reset_rng, rollout_rng = jax.random.split(rng)
    init_obs, init_state = rollout_manager.batch_reset(
        reset_rng, env_params, num_workers
    )
    rollout, _, _, _ = rollout_manager.batch_rollout(
        rollout_rng, 
        actor_state,
        env_params,
        init_obs,
        init_state
    )

    adv, values, target = compute_val_adv_target(
        critic_state, rollout, gamma, gae_lambda
    )
    
    agent_train_step_fn = partial(
        agent_train_step_fn,
        rollout,
        values,
        adv,
        target,

    )
    



