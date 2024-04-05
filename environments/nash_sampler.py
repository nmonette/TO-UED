from typing import List

import chex
import jax
import jax.numpy as jnp

from jax import random
from jax.tree_util import tree_map

from jaxopt.projection import projection_simplex

from flax import struct
from functools import partial
from typing import Callable

from util import *
from environments.environments import get_env, reset_env_params, get_env_spec
from environments.level_sampler import LevelSampler, LevelBuffer
from environments.rollout import RolloutWrapper
from agents.agents import (
    create_agent,
    create_value_critic,
    eval_agent,
    AgentHyperparams,
)
from agents.a2c import train_a2c_agent, A2CHyperparams
from agents.agents import eval_agent, compute_advantage
from meta.meta import create_lpg_train_state, make_lpg_train_step
from agents.lpg_agent import train_lpg_agent


# TODO: Reimplement positive_value_loss and l1_value_loss
SCORE_FUNCTIONS = ["random", "frozen", "alg_regret"]
SCORE_TRANSFORMS = ["proportional", "rank"]

@struct.dataclass
class Game:
    game: int
    x: int
    y: int

    def get_grads(self, x=False, y=False):
        if x:
            return jnp.dot(self.game,  self.y)
        elif y:
            return jnp.dot(self.x.T, self.game)
        # else:
        #     return self.x.T @ self.game @ self.y

def get_nash(game: Game, num_iters = 1000):
    x_strats = jnp.empty((num_iters + 1, game.x.shape[0])).at[0].set(game.x)
    y_strats = jnp.empty((num_iters + 1, game.y.shape[0])).at[0].set(game.y)

    def run_loop(carry, t):
        x_strats, y_strats, game = carry

        x_grad = game.grad(x=True)
        x = projection_simplex(x_strats[t-1] - 0.01 * x_grad)

        y_grad = game.grad(y=True)
        y = projection_simplex(y_strats[t-1] - 0.01 * y_grad)

        game = game.replace(x=x, y=y)
        
        return (x_strats.at[t].set(x), y_strats.at[t].set(y), game), None

    carry, _ = jax.lax.scan(run_loop, (x_strats, y_strats, game), jnp.arange(1, num_iters + 1), num_iters)

    return jnp.mean(carry[0], axis=0), jnp.mean(carry[1], axis=0)


class NashSampler(LevelSampler):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.lpg_hypers = LpgHyperparams.from_run_args(args)

    def _initialize_buffer(self, rng):
        """Creates a new level buffer, if used by the score function."""
        rng = jax.random.split(rng, self.buffer_size)
        random_params, random_lifetimes = self._sample_env_params(rng)
        return LevelBuffer.create_buffer(random_params, random_lifetimes).at[0].replace(active=True)
    
    def initialize_buffers(self, rng):
        rng, train_rng, eval_rng = jax.random.split(rng, 3)
        train_buffer = self._initialize_buffer(train_rng)
        eval_buffer = self._initialize_buffer(eval_rng)

        return train_buffer, eval_buffer
    
    def _train_agent(self, rng, level, agent_state, value_critic_state, lpg_train_state, agent_train_fn):
        """Perform K agent train steps then evaluate an agent."""
        # --- Perform K agent train steps ---
        rng, _rng = jax.random.split(rng)
        agent_state, agent_metrics = agent_train_fn(_rng, lpg_train_state, agent_state)

        # --- Rollout updated agent ---
        rng, _rng = jax.random.split(rng)
        eval_rollouts, env_obs, env_state, _ = self.rollout_manager.batch_rollout(
            _rng,
            agent_state.actor_state,
            agent_state.level.env_params,
            agent_state.env_obs,
            agent_state.env_state,
        )
        agent_state = agent_state.replace(
            env_obs=env_obs,
            env_state=env_state,
        )

        # --- Update value function ---
        def _compute_value_loss(critic_params):
            value_critic_state.replace(params=critic_params)
            value_loss, adv = jax.vmap(
                compute_advantage, in_axes=(None, 0, None, None)
            )(value_critic_state, eval_rollouts, self.args.gamma, self.args.gae_lambda)
            return value_loss.mean(), adv

        (value_loss, adv), value_critic_grad = jax.value_and_grad(
            _compute_value_loss, has_aux=True
        )(value_critic_state.params)
        value_critic_state = value_critic_state.apply_gradients(grads=value_critic_grad)
    
        return agent_state, value_critic_state
    
    def _train_lpg(self, rng, train_level):
        # --- Initialize LPG and training loop ---
        rng, lpg_rng = jax.random.split(rng)
        train_state = create_lpg_train_state(lpg_rng, self.args)
        lpg_train_step_fn = make_lpg_train_step(self.args, self.rollout_manager, single_env=True)

        temp_buffer = LevelBuffer(train_level, 0, True, True)

        # --- Initialize agent and value critic for training level ---
        ## there may be an issue that the _create_agent and create_value_critic are a single object instead of an array but we'll see
        rng, agent_rng, value_rng = random.split(rng, 3)
        agent_states = self._create_agent(agent_rng, train_level)
        value_critics = create_value_critic(value_rng, self.agent_hypers, self.obs_shape)

        def _meta_train_loop(carry, _):
            rng, train_state, agent_states, value_critic_states, level_buffer = carry

            # --- Update LPG ---
            rng, _rng = jax.random.split(rng)
            train_state, agent_states, value_critic_states, metrics = lpg_train_step_fn(
                rng=_rng,
                lpg_train_state=train_state,
                agent_states=agent_states,
                value_critic_states=value_critic_states,
            )
            carry = (rng, train_state, agent_states, value_critic_states, level_buffer)
            return carry, metrics

        carry = (rng, train_state, agent_states, value_critic_states, level_buffer)
        carry, metrics = jax.lax.scan(
            _meta_train_loop, carry, None, length=self.args.train_steps
        )
        rng, train_state, agent_states, value_critic_states, level_buffer = carry

        return train_state
    
    def _compute_algorithmic_regret(self, rng, train_level, eval_level, train_state=None):
        if train_level.active == False or eval_level.active == False:
            return None

        # --- Train LPG on train_level unless LPG is already provided ---
        if train_state is None:
            train_state = self._train_lpg(rng, train_level)

        # --- Train an on the eval environment using the new LPG ---

        ## --- Initialize agent and value critic for eval level ---
        rng, agent_rng, value_rng = random.split(rng, 3)
        agent_states = self._create_agent(agent_rng, eval_level)
        value_critics = create_value_critic(value_rng, self.agent_hypers, self.obs_shape)

        ## --- Train the agent ---
        agent_train_fn = partial(
            train_lpg_agent,
            rollout_manager=self.rollout_manager,
            num_train_steps=self.lpg_hypers.num_agent_updates,
            agent_target_coeff=self.lpg_hypers.agent_target_coeff,
        )
        rng, train_rng = jax.random.split(rng)
        agent_state, value_critic = self._train_agent(train_rng, eval_level, agent_states, value_critics, train_state, agent_train_fn)
        
        return super()._compute_algorithmic_regret(rng, agent_state)

    def get_payoff_matrix(self, rng, train_buffer, eval_buffer):
        # --- Train agents on each level in train buffer ---
        rng, *train_rng = jax.random.split(rng, jnp.sum(train_buffer.active.astype(int)) + 1)
        train_states = jax.vmap(self._train_lpg)(train_rng, train_buffer.level)

        rng, *ar_rng = jax.random.split(rng, jnp.sum(train_buffer.active.astype(int)) + 1)
        ar_fn = jax.vmap(jax.vmap(self._compute_algorithmic_regret, in_axes=(0, 0, None)), in_axes=(None, None, 0))

        return ar_fn(eval_buffer)(ar_rng, train_buffer.level, eval_buffer.level)
    
    def compute_nash(self, rng, train_buffer, eval_buffer):
        # --- Calculate Payoff Matrix ---
        matrix = self.get_payoff_matrix(rng, train_buffer, eval_buffer)
        
        rng, _rng = jax.random.split(rng)
        # --- Initialize random strategies ---
        strats = jax.nn.softmax(jax.random.uniform(_rng, (2, matrix.shape[0])), axis=1)
        # -- Calculate nash ---
        game = Game(matrix, strats[0], strats[1])
        return get_nash(game) # x,y = get_nash(game)

        # return jnp.pad(x, (0, self.buffer_size - len(x))), jnp.pad(y, (0, self.buffer_size - len(y)))
    
    def get_training_levels(self, rng, train_buffer, train_nash_dist):
        rng, _rng = jax.random.split(rng)
        envs = jax.random.categorical(_rng, jnp.nan_to_num(jnp.log(train_nash_dist)), shape=(self.args.buffer_size, ))

        rng, *_rng = jax.random.split(rng, self.args.buffer_size + 1)
        return jax.vmap(self._create_agent)(_rng, train_buffer.level[envs], not self.args.use_es)

    def _get_level(self, rng):
        """
        Sample an environment. 
        TODO: Add in an environment generator
        """
        # --- Sample a level ---
        rng, _rng = jax.random.split(rng)
        level, agent_state, value_critic_state = self.initial_sample(rng, not self.args.use_es)
        return level.replace(active=True)

    def get_train_br(self, rng, eval_nash, eval_buffer):
        """
        Sample a level, then collect the convex combination of 
        algorithmic regrets over the nash of evaluative 
        environments. If the regret
        is lower than the current lowest, replace the
        'current min' with the newly generated/sampled
        environment. If choosing to use a generator, 
        use the -regret as a reward for the generator.
        """
        rng, *_rng = jax.random.split(rng, self.args.br + 1)

        def _br_loop(rng):
            # --- Sample a level and train an LPG on it ---
            train_level = self._get_level(rng)

            # --- Collect Evaluative Regrets ---
            rng, *_rng = jax.random.split(rng, jnp.sum(eval_buffer.active.astype(int)) + 1)
            regrets = jax.vmap(partial(self._compute_algorithmic_regret, train_level=train_level))(rng=_rng, eval_level=eval_buffer.level)

            return train_level, jnp.dot(eval_nash, regrets)
        
        levels, regrets = jax.vmap(_br_loop)(_rng)
        
        return levels[jnp.argmin(regrets)].replace(active=True)
    
    def get_eval_br(self, rng, train_nash, train_buffer):
        """
        First, train an LPG on the training nash. Then, 
        repeatedly sample envs and evaluate regret on them.
        We will use super()._compute_algorithmic_regret here.
        """
        # --- Initialize agent states ---
        rng, _rng = jax.random.split(rng)
        agent_states = self.get_training_levels(_rng, train_buffer, train_nash)
        ## we won't create a value_critic here because we will use ES.

        lpg_train_step_fn = make_lpg_train_step(self.args, self.rollout_manager, single_env=True)

        def _meta_train_loop(carry, _):
            rng, train_state, agent_states, value_critic_states, level_buffer = carry

            # --- Update LPG ---
            rng, _rng = jax.random.split(rng)
            train_state, agent_states, value_critic_states, metrics = lpg_train_step_fn(
                rng=_rng,
                lpg_train_state=train_state,
                agent_states=agent_states,
                value_critic_states=value_critic_states,
            )
            carry = (rng, train_state, agent_states, value_critic_states, level_buffer)
            return carry, metrics

        carry = (rng, train_state, agent_states, None, level_buffer)
        carry, metrics = jax.lax.scan(
            _meta_train_loop, carry, None, length=self.args.train_steps
        )
        rng, train_state, agent_states, _, level_buffer = carry

        def _br_loop(rng):
            # --- Sample a level ---
            eval_level = self._get_level(rng)

            # --- Collect Evaluative Regrets ---
            rng, _rng = jax.random.split(rng)
            return eval_level, self._compute_algorithmic_regret(_rng, None, eval_level, train_state)
        
        levels, regrets = jax.vmap(_br_loop)(_rng)
        
        return levels[jnp.argmax(regrets)].replace(active=True)
    
