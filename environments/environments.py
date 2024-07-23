import jax
import chex
import gymnax

import environments.gridworld.gridworld as grid
import environments.gridworld.configs as grid_conf
import environments.gymnax.configs as gym_conf

from environments.jaxued.maze import make_level_generator
import environments.jaxued.maze.env as maze
from environments.jaxued.maze.env_solved import MazeSolved
import environments.jaxued.maze.configs as maze_conf
from .jaxued.autoreplay import AutoReplayWrapper
from .jaxued.autoreset import AutoResetWrapper, AutoResetFiniteWrapper


def get_env(env_name: str, env_kwargs: dict):
    if env_name in gymnax.registered_envs:
        env, _ = gymnax.make(env_name, **env_kwargs)
    elif env_name in grid.registered_envs:
        env = grid.GridWorld(**env_kwargs)
    elif env_name in maze.registered_envs:
        env = AutoReplayWrapper(
            MazeSolved(**env_kwargs)
        )
        # env = AutoResetFiniteWrapper(
        #     MazeSolved(**env_kwargs), make_level_generator(
        #         env_kwargs["max_height"], env_kwargs["max_width"], 25
        #     )
        # )
    else:
        raise ValueError(
            f"Environment {env_name} not registered in any environment sources."
        )
    return env


def reset_env_params(rng: chex.PRNGKey, env_name: str, env_mode: str):
    """Reset environment parameters and agent lifetime."""
    if env_name in gymnax.registered_envs:
        env, _ = gymnax.make(env_name)
        params = env.default_params
        lifetime = None
        if env_name in gym_conf.configured_envs:
            # Select lifetime if mode configuration exists
            lifetime = gym_conf.reset_lifetime(env_name=env_name)
    elif env_name in grid.registered_envs:
        p_rng, l_rng = jax.random.split(rng)
        params = grid_conf.reset_env_params(p_rng, env_mode)
        lifetime = grid_conf.reset_lifetime(l_rng, env_mode)
    else:
        raise ValueError(f"Environment {env_name} has no parameter reset method.")
    return params, lifetime


def get_env_spec(env_name: str, env_mode: str):
    """Returns static environment parameters, rollout length and lifetime."""
    if env_name in [*gymnax.registered_envs]:
        kwargs = {}
        env = get_env(env_name, kwargs)
        max_rollout_len = env.default_params.max_steps_in_episode
        if env_name in gym_conf.configured_envs:
            max_lifetime = gym_conf.get_max_lifetime(env_name=env_name)
        else:
            max_lifetime = None
    elif env_name in grid.registered_envs:
        kwargs, max_rollout_len = grid_conf.get_env_spec(env_mode)
        max_lifetime = grid_conf.get_max_lifetime(env_mode)
    elif env_name in maze.registered_envs:
        kwargs, max_rollout_len = maze_conf.get_env_spec("maze")
        max_lifetime = maze_conf.get_max_lifetime("maze")
    else:
        raise ValueError(f"Environment {env_name} has no get env spec method.")
    return kwargs, max_rollout_len, max_lifetime


def get_agent_hypers(env_name: str, env_mode: str = None):
    if env_name in gym_conf.configured_envs:
        return gym_conf.get_agent_hypers(env_name)
    elif env_name in grid.registered_envs:
        return grid_conf.get_agent_hypers(env_mode)
    elif env_name in maze.registered_envs:
        return maze_conf.get_agent_hypers(env_mode)
    raise ValueError(f"Environment {env_name} has no get agent hyperparameters method.")
