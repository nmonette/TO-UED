import chex

def reset_lifetime(rng: chex.PRNGKey, env_mode: str):
    return _MAZE_LIFETIME


def get_env_spec(mode: str):
    """Returns static environment specification and maximum episode length."""
    return {k: v for k, v in ENV_MODE_KWARGS[mode].items()}, ENV_MODE_EPISODE_LEN[mode]


def get_max_lifetime(mode: str):
    """Returns maximum lifetime length."""
    return _MAZE_LIFETIME


def get_agent_hypers(mode: str):
    """Returns agent hyperparameters for a given mode."""
    return MODE_AGENT_HYPERS[mode]


ENV_MODE_EPISODE_LEN = {
    "maze":50
}

# Reference: lifetime = int(3e6 / (args.env_workers * args.train_rollout_len))
# Updates per LPG update (K) * LPG updates
_TABULAR_LIFETIME = 5 * 500
_RAND_LIFETIME = 10 * 5 * 500
_SMALL_LIFETIME = 5 * 50
_MEDIUM_LIFETIME = 5 * 200
_LARGE_LIFETIME = 5 * 500
_MAZE_LIFETIME = 5 * 500
_DEBUG_LIFETIME = 4

_TABULAR_HYPERS = {
    "actor_net": (),
    "actor_learning_rate": 1e-6,
    "critic_net": (),
    "critic_learning_rate": 1e-6,  # Reference: 4e+1
    "optimizer": "Adam",
    "max_grad_norm": 0.5,
}

_RAND_HYPERS = {
    "actor_net": (32,),
    "actor_learning_rate": 1e-3,
    "critic_net": (32,),
    "critic_learning_rate": 1e-3,
    "optimizer": "Adam",
    "max_grad_norm": 0.5,
}

# Convolution layers have form (features, kernel_width)
_TINY_HYPERS = {
    "actor_net": (32, 32, 32),
    "actor_learning_rate": 1e-3,
    "critic_net": (32, 32, 32),
    "critic_learning_rate": 1e-3,
    "optimizer": "Adam",
    "max_grad_norm": 0.5,
}

MODE_AGENT_HYPERS = {
    "maze":_TABULAR_HYPERS
}

ENV_MODE_KWARGS = {
    "maze": {
        "max_height": 13,
        "max_width": 13,
        "agent_view_size": 5,
        "see_agent": False,
        "normalize_obs": True,
        "fully_obs": False,
        "penalize_time": True
    }
}