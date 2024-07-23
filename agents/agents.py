import jax
import chex
from flax import struct

from util import *
from models.agent import Actor, ConvActor, Critic, ConvCritic
from models.optim import create_optimizer
from environments.environments import get_agent_hypers

from ued.rnn import (
    _get_critic_model as _get_rnn_critic,
    _get_policy_model as _get_rnn_policy,
)

@struct.dataclass
class AgentHyperparams:
    actor_net: tuple
    actor_learning_rate: float
    critic_net: tuple
    critic_learning_rate: float
    optimizer: str
    max_grad_norm: float
    # Number of critic dimensions, 1 for value critic, > 1 for target (LPG) critic
    critic_dims: int = 1
    convert_nchw: bool = False

    @staticmethod
    def from_args(args):

        def actor_linear_schedule(count):
            frac = (
                1.0
                - 0.1 * (count // (args.num_mini_batches * args.num_epochs))
                / args.train_steps
            )
            return args.actor_lr * frac

        def critic_linear_schedule(count):
            frac = (
                1.0
                - 0.1 * (count // (args.num_mini_batches * args.num_epochs))
                / args.train_steps
            )
            return args.critic_lr * frac
        
        agent_hypers_dict = {
            k: v for k, v in get_agent_hypers(args.env_name, args.env_mode).items()
        }
        # TODO: Make overrides here if tuning
        return AgentHyperparams(**agent_hypers_dict, critic_dims=args.lpg_target_width).replace(
            actor_learning_rate=critic_linear_schedule,
            critic_learning_rate=actor_linear_schedule
        )


def create_agent(
    rng: chex.PRNGKey, agent_params: AgentHyperparams, action_n: int, obs_shape: tuple
):
    """Initialise actor and critic train states for a single agent."""
    if type(obs_shape) is int:
        obs_shape = (obs_shape,)
    actor_rng, critic_rng = jax.random.split(rng)
    policy_model = _get_policy_model(agent_params, action_n, len(obs_shape))
    critic_model = _get_critic_model(agent_params, len(obs_shape))
    actor_train_state = _create_train_state(
        actor_rng,
        policy_model,
        obs_shape,
        agent_params.optimizer,
        agent_params.actor_learning_rate,
        agent_params.max_grad_norm,
    )
    critic_train_state = _create_train_state(
        critic_rng,
        critic_model,
        obs_shape,
        agent_params.optimizer,
        agent_params.critic_learning_rate,
        agent_params.max_grad_norm,
    )
    return actor_train_state, critic_train_state

def create_agent_rnn(
    rng: chex.PRNGKey, agent_params: AgentHyperparams, action_n: int, obs_shape: tuple
):
    """Initialise actor and critic train states for a single agent."""
    if type(obs_shape) is int:
        obs_shape = (obs_shape,)
    actor_rng, critic_rng = jax.random.split(rng)
    policy_model = _get_rnn_policy(action_n)
    critic_model = _get_rnn_critic()

    actor_train_state = _create_rnn_train_state(
        actor_rng,
        policy_model,
        obs_shape,
        agent_params.optimizer,
        agent_params.actor_learning_rate,
        agent_params.max_grad_norm,
        policy_model.initialize_carry(())
    )
    critic_train_state = _create_rnn_train_state(
        critic_rng,
        critic_model,
        obs_shape,
        agent_params.optimizer,
        agent_params.critic_learning_rate,
        agent_params.max_grad_norm,
        policy_model.initialize_carry(())
    )
    
    return actor_train_state, critic_train_state

def create_value_critic(
    rng: chex.PRNGKey, agent_params: AgentHyperparams, obs_shape: tuple
):
    """Initialise value critic train state."""
    if type(obs_shape) is int:
        obs_shape = (obs_shape,)
    agent_params = agent_params.replace(critic_dims=1)
    critic_model = _get_critic_model(agent_params, len(obs_shape))
    critic_train_state = _create_train_state(
        rng,
        critic_model,
        obs_shape,
        agent_params.optimizer,
        agent_params.critic_learning_rate,
        agent_params.max_grad_norm,
    )
    return critic_train_state


def _get_policy_model(agent_params, n_actions, obs_n_dims):
    if obs_n_dims > 2:  # CNN for 3D observations
        return ConvActor(agent_params.actor_net, n_actions, agent_params.convert_nchw)
    return Actor(agent_params.actor_net, n_actions)


def _get_critic_model(agent_params, obs_n_dims):
    if obs_n_dims > 2:  # CNN for 3D observations
        return ConvCritic(
            agent_params.actor_net, agent_params.critic_dims, agent_params.convert_nchw
        )
    return Critic(agent_params.actor_net, agent_params.critic_dims)


def _create_train_state(rng, model, obs_shape, optimizer, learning_rate, max_grad_norm):
    params = model.init(rng, jnp.ones(obs_shape))["params"]
    tx = create_optimizer(optimizer, learning_rate, max_grad_norm)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def _create_rnn_train_state(rng, model, obs_shape, optimizer, learning_rate, max_grad_norm, hidden):
    params = model.init(rng, (jax.tree_map(lambda x: jnp.ones((1, *x)), obs_shape, is_leaf=lambda x: type(x) is tuple), jnp.full((1, 1), False)), hidden)["params"]
    tx = create_optimizer(optimizer, learning_rate, max_grad_norm)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def eval_agent(rng, rollout_manager, env_params, actor_train_state, num_workers):
    """Evaluate episodic agent performance over multiple workers."""
    rng, _rng = jax.random.split(rng)
    env_obs, env_state = rollout_manager.batch_reset(_rng, env_params, num_workers)
    rng, _rng = jax.random.split(rng)
    _, _, _, tot_reward = rollout_manager.batch_rollout(
        _rng, actor_train_state, env_params, env_obs, env_state, eval=True
    )
    return tot_reward.mean()

def compute_advantage(critic_state, rollout, gamma, gae_lambda):
    """Compute semi-gradient critic MSE and advantage over a rollout"""
    all_obs = jnp.append(rollout.obs, jnp.expand_dims(rollout.next_obs[-1], 0), axis=0)
    value = critic_state.apply_fn({"params": critic_state.params}, all_obs)
    adv, target = jax.lax.stop_gradient(
        gae(value, rollout.reward, rollout.done, gamma, gae_lambda)
    )
    return jnp.mean(jnp.square(target - value[:-1])), adv

def compute_adv_target(values, rollout, gamma, gae_lambda):
    """Compute advantage over a rollout, also return values and target estimates"""
    adv, target = jax.lax.stop_gradient(
        gae(values, rollout.reward, rollout.done, gamma, gae_lambda)
    )
    return adv, values, target
