"""
Based on Gymnax experimental/rollout.py
"""
import jax
import jax.numpy as jnp

from typing import Optional
from environments.environments import get_env
from environments.jaxued.maze.env import EnvParams

from util import *


class RolloutWrapper:
    def __init__(
        self,
        env_name: str = "Pendulum-v1",
        train_rollout_len: Optional[int] = None,
        eval_rollout_len: Optional[int] = None,
        env_kwargs: dict = {},
        return_info: bool = False,
    ):
        """
        env_name (str): Name of environment to use.
        train_rollout_len (int): Number of steps to rollout during training.
        eval_rollout_len (int): Number of steps to rollout during evaluation.
        env_kwargs (dict): Static keyword arguments to pass to environment, same for all agents.
        return_info (bool): Return rollout information.
        """
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        # Define the RL environment & network forward function
        self.env = get_env(env_name, env_kwargs)

        self.train_rollout_len = train_rollout_len
        self.eval_rollout_len = eval_rollout_len
        self.return_info = return_info

    def batch_reset_single_env(self, rng, env_params, num_workers):
        """Reset a single environment for multiple workers, returning initial states and observations."""
        rng = jax.random.split(rng, num_workers)
        if self.env_name == "Maze-v0":
            batch_reset_fn = jax.vmap(
                partial(self.env.reset_env_to_level, params=self.env.default_params), in_axes=(0, None))
        else:   
            batch_reset_fn = jax.vmap(self.env.reset, in_axes=(0, None))

        return batch_reset_fn(rng, env_params)

    # --- ENVIRONMENT ROLLOUT ---
    def batch_rollout_single_env(
        self, rng, actor_state, critic_state, env_params, init_obs, init_state, actor_hstate, critic_hstate, eval=False
    ):
        """Evaluate an agent on a single environment over a batch of workers."""
        rng = jax.random.split(rng, init_obs.image.shape[0])
        return jax.vmap(self.single_rollout, in_axes=(0, None, None, None, 0, 0, 0, 0, None))(
            rng, actor_state, critic_state, env_params, init_obs, init_state, actor_hstate, critic_hstate, eval
        )

    # --- ENVIRONMENT RESET ---
    def batch_reset(self, rng, env_params):
        """Reset a single environment for multiple workers, returning initial states and observations."""
        
        if self.env_name == "Maze-v0":
            rng = jax.random.split(rng, env_params.width.shape[0])
            batch_reset_fn = jax.vmap(
                partial(self.env.reset_env_to_level, params=self.env.default_params)
            )
        else:
            rng = jax.random.split(rng, env_params.max_steps_in_episode.shape[0])
            batch_reset_fn = jax.vmap(self.env.reset)
        return batch_reset_fn(rng, env_params)

    # --- ENVIRONMENT ROLLOUT ---
    def batch_rollout(
        self, rng, actor_state, critic_state, env_params, init_obs, init_state, actor_hstate, critic_hstate,  eval=False
    ):
        """Evaluate an agent on a single environment over a batch of workers."""
        rng = jax.random.split(rng, init_state.time.shape[0])
        return jax.vmap(self.single_rollout, in_axes=(0, None, None, 0, 0, 0, 0, 0, None))(
            rng, actor_state, critic_state, env_params, init_obs, init_state, actor_hstate, critic_hstate, eval
        )

    def single_rollout(
        self, rng, actor_state, critic_state, env_params, init_obs, init_state, actor_hstate, critic_hstate, eval=False
    ):
        """Rollout an episode."""

        def policy_step(state_input, _):
            rng, obs, state, actor_state, critic_state, actor_hstate, critic_hstate, cum_reward, valid_mask, last_done = state_input
            rng, _rng = jax.random.split(rng)
            reshaped_obs = obs.replace(
                image = obs.image.reshape(1, *obs.image.shape)
            )
            # last_done.reshape(1,1)
            actor_hstate, action_probs = actor_state.apply_fn({"params": actor_state.params}, (reshaped_obs, jnp.full((1,1), False)), actor_hstate)
            
            value = None
            if critic_state is not None:
                # last_done.reshape(1,1)
                critic_hstate, value = critic_state.apply_fn({"params": critic_state.params}, (reshaped_obs, jnp.full((1,1), False)), critic_hstate)
    
            action = jax.random.choice(_rng, action_probs.shape[-1], p=action_probs.squeeze())
            rng, _rng = jax.random.split(rng)
            next_obs, next_state, reward, done, info = self.env.step(
                _rng, state, action, env_params
            )
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry = [
                rng,
                next_obs,
                next_state,
                actor_state,
                critic_state,
                actor_hstate,
                critic_hstate,
                new_cum_reward,
                new_valid_mask,
                done
            ]
            transition = Transition(obs, action, reward, next_obs, done, jnp.log(action_probs.squeeze()[action]), value)
            if self.return_info:
                return carry, (transition, info)
            return carry, transition

        # Scan over episode step loop
        carry_out, rollout = jax.lax.scan(
            policy_step,
            [
                rng,
                init_obs,
                init_state,
                actor_state,
                critic_state, 
                actor_hstate,
                critic_hstate,
                jnp.float32(0.0),
                jnp.float32(1.0),
                jnp.full((), False)
            ],
            (),
            self.eval_rollout_len if eval else self.train_rollout_len,
        )
        if self.return_info:
            rollout, info = rollout
        end_obs, end_state, actor_hstate, cum_return, critic_hstate = carry_out[1], carry_out[2], carry_out[5], carry_out[7], carry_out[6]

        # --- Add final value onto end of rollouts ---
        if critic_state is not None:
            reshaped_obs = end_obs.replace(
                image = end_obs.image.reshape(1, *end_obs.image.shape)
            )
            critic_hstate, value = critic_state.apply_fn({"params": critic_state.params}, (reshaped_obs, jnp.full((1,1), False)), critic_hstate)
            rollout = rollout.replace(
                value = jnp.append(rollout.value, value.reshape(1,1), axis=0).squeeze()
            )

        if self.return_info:
            return rollout, end_obs, end_state, actor_hstate, critic_hstate, cum_return, info
        return rollout, end_obs, end_state, actor_hstate, critic_hstate, cum_return

    def optimal_return(self, env_params, max_rollout_len, return_all):
        """Return the optimal expected return for the given set of environment parameters."""
        return jax.vmap(self.env.optimal_return, in_axes=(0, None, None))(
            env_params, max_rollout_len, return_all
        )

    @property
    def input_shape(self):
        """Get the shape of the observation."""
        rng = jax.random.PRNGKey(0)
        obs, state = self.env.reset(rng, self.env_params)
        return obs.shape
