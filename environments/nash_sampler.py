
import jax
import jax.numpy as jnp

from jax import random

from flax import struct

from util import *
from environments.level_sampler import LevelSampler, LevelBuffer
from environments.environments import reset_env_params

from agents.agents import create_value_critic
from agents.agents import compute_advantage
from agents.lpg_agent import train_lpg_agent

from meta.meta import create_lpg_train_state, make_lpg_train_step

from functools import partial

# TODO: Reimplement positive_value_loss and l1_value_loss
SCORE_FUNCTIONS = ["random", "frozen", "alg_regret"]
SCORE_TRANSFORMS = ["proportional", "rank"]

@struct.dataclass
class Game:
    game: int
    x: int
    y: int

    def grad(self, x=False, y=False):
        if x:
            return jnp.dot(self.game,  self.y)
        elif y:
            return -jnp.dot(self.x.T, self.game)
        # else:
        #     return self.x.T @ self.game @ self.y

def get_nash(game: Game, x_nz, y_nz, num_iters = 10000):
    x_strats = jnp.empty((num_iters + 1, game.x.shape[0])).at[0].set(game.x)
    y_strats = jnp.empty((num_iters + 1, game.y.shape[0])).at[0].set(game.y)

    def run_loop(carry, t):
        x_strats, y_strats, game = carry

        x_grad = game.grad(x=True)
        x = projection_simplex(game.x - 0.01 * x_grad, x_nz)

        y_grad = game.grad(y=True)
        y = projection_simplex(game.y - 0.01 * y_grad, y_nz)

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
        buffer = LevelBuffer.create_buffer(random_params, random_lifetimes)
        return buffer.replace(active=buffer.active.at[0].set(True))
    
    def initialize_buffers(self, rng):
        rng, train_rng, eval_rng = jax.random.split(rng, 3)
        train_params, train_lifetime = reset_env_params(train_rng, self.env_name, self.env_mode)
        eval_params, eval_lifetime = reset_env_params(eval_rng, self.env_name, self.env_mode)
        train_buffer = [LevelBuffer(Level(train_params, train_lifetime, 0), 0.0, False, True)]
        eval_buffer = [LevelBuffer(Level(eval_params, eval_lifetime, 0), 0.0, False, True)]

        return train_buffer, eval_buffer
    
    def _train_agent(self, rng, level, agent_state, value_critic_state, lpg_train_state, agent_train_fn):
        """Perform K agent train steps then evaluate an agent."""
        rng, train_rng, rollout_rng = jax.random.split(rng, 3)

        # --- Perform K agent train steps ---
        agent_state, agent_metrics = agent_train_fn(train_rng, lpg_train_state, agent_state)

        # --- Rollout updated agent ---
        rng, _rng = jax.random.split(rng)
        eval_rollouts, env_obs, env_state, _ = self.rollout_manager.batch_rollout(
            rollout_rng,
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
    
    def _train_lpg(self, rng, train_level, train_state):
        # ---  LPG  training loop ---
        lpg_train_step_fn = make_lpg_train_step(self.args, self)

        # --- Initialize agent and value critic for training level ---
        rng, agent_rng, value_rng = random.split(rng, 3)
        agent_rng = random.split(agent_rng, self.args.num_agents)
        agent_states = jax.vmap(self._create_agent, in_axes=(0, None))(agent_rng, train_level)
        
        value_critic_states = None
        if not self.args.use_es:
            value_rng = random.split(value_rng, self.args.num_agents)
            value_critic_states = jax.vmap(create_value_critic, in_axes=(0, None, None))(value_rng, self.agent_hypers, self.obs_shape)

        def _meta_train_loop(carry, _):
            rng, train_state, agent_states, value_critic_states = carry

            # --- Update LPG ---
            rng, _rng = jax.random.split(rng)
            train_state, agent_states, value_critic_states, metrics = lpg_train_step_fn(
                rng=_rng,
                lpg_train_state=train_state,
                agent_states=agent_states,
                value_critic_states=value_critic_states,
            )
            carry = (rng, train_state, agent_states, value_critic_states)
            return carry, metrics

        carry = (rng, train_state, agent_states, value_critic_states)
        carry, metrics = jax.lax.scan(
            _meta_train_loop, carry, None, length=self.args.train_steps
        )
        rng, train_state, agent_states, value_critic_states = carry

        return train_state
    
    
    
    def _compute_algorithmic_regret(self, rng, train_level, eval_level, train_state=None):        
        """
        We are no longer going to check if it is active, because we will pass in the appropriately sized buffer pre-jit
        """
        # --- Train LPG on train_level unless train_level is not provided ---
        if train_level is not None:
            train_state = self._train_lpg(rng, train_level, train_state)

        # --- Train an agent on the eval environment using the new LPG ---

        ## --- Initialize agent for eval level ---
        rng, agent_rng, value_rng = random.split(rng, 3)
        agent_states = self._create_agent(agent_rng, eval_level)
        
        ## --- Train the agent --- 
        rng, train_rng = jax.random.split(rng)
        agent_state, _, _ = train_lpg_agent(train_rng, train_state.train_state, agent_states, self.rollout_manager, self.lpg_hypers.num_agent_updates, self.lpg_hypers.agent_target_coeff) 
        return super(NashSampler, self)._compute_algorithmic_regret(rng, agent_state)
    
    def get_payoff_matrix(self, rng, train_state, train_buffer, eval_buffer):
        # --- Train agents on each level in train buffer ---
        rng, *train_rng = jax.random.split(rng, self.buffer_size + 1)
        
        train_rng = jnp.array(train_rng)
        train_states = mini_batch_vmap(self._train_lpg, 10, in_axes=(0, 0, None,))(train_rng, train_buffer.level, train_state)

        rng, _rng = jax.random.split(rng)
        _rng = jax.random.split(_rng, (self.buffer_size, self.buffer_size))

        ar_fn = mini_batch_vmap(mini_batch_vmap(self._compute_algorithmic_regret, 10, in_axes=(0, None, 0, 0, 0, None)), 10, (0, 0, None, None, None, 0))
        return ar_fn(_rng, train_buffer.level, eval_buffer.level, train_states, train_buffer.active, eval_buffer.active)
    
    @jax.jit
    def compute_nash(self, rng, train_state, train_buffer, eval_buffer):
        # --- Calculate Payoff Matrix ---
        matrix = self.get_payoff_matrix(rng, train_state, train_buffer, eval_buffer)
        rng, _rng = jax.random.split(rng)
        # --- Initialize random strategies ---
        nz = jnp.sum(train_buffer.active)
        strats = jnp.where(jnp.arange(0, matrix.shape[0]) < nz, jax.random.uniform(_rng, (2, matrix.shape[0])), 0)
        x = projection_simplex(strats[0], nz)
        y = projection_simplex(strats[1], nz)
        # -- Calculate nash ---
        game = Game(matrix, x, y)
        x,y = get_nash(game, jnp.sum(train_buffer.active), jnp.sum(eval_buffer.active))

        return x, y, matrix
    
    def get_training_levels(self, rng, train_buffer, train_nash, num_agents=None, create_value_critic=True):
        if num_agents is None:
            num_agents = self.args.num_agents

        print(len(train_buffer))
        
        # --- Sample levels ---
        rng, _rng = jax.random.split(rng)
        idx = jax.random.choice(_rng, jnp.arange(0, train_nash.shape[0]), (num_agents, ), True, train_nash)
        envs = jax.tree_util.tree_map(lambda x:  x[idx], train_buffer.level)

        # --- Get agent states from levels ---
        rng, agent_rng, value_rng = jax.random.split(rng, 3)
        agent_rng = jax.random.split(agent_rng, num_agents)
        agent_states = jax.vmap(self._create_agent, in_axes=(0, 0, None))(agent_rng, envs, not self.args.use_es)

        value_critics = None
        if create_value_critic:
            value_rng = jax.random.split(value_rng, self.args.buffer_size)
            value_critics = jax.vmap(create_value_critic, in_axes=(0, None, None))(
                value_rng, self.agent_hypers, self.obs_shape
            )
        return agent_states, value_critics

    @jax.jit
    def get_train_br(self, rng, train_state, eval_nash, eval_buffer):
        """
        Sample a level, then collect the convex combination of 
        algorithmic regrets over the nash of evaluative 
        environments. If the regret
        is lower than the current lowest, replace the
        'current min' with the newly generated/sampled
        environment. If choosing to use a generator, 
        use the -regret as a reward for the generator.
        """
        def _br_loop(rng):
            # --- Sample a level to train a LPG on ---
            rng, _rng = jax.random.split(rng)
            params, lifetime = reset_env_params(_rng, self.env_name, self.env_mode)
            train_level = Level(params, lifetime, 0)

            # --- Compute Regrets --- 
            regrets = mini_batch_vmap(self._compute_algorithmic_regret, 10, in_axes=(None, None, 0, None, None, 0))(rng, train_level, eval_buffer.level, train_state, True, eval_buffer.active)

            # --- Return expected regret over the nash ---
            return train_level, jnp.dot(eval_nash, regrets)
        
        rng = jax.random.split(rng, self.args.br)
        levels, regrets = mini_batch_vmap(_br_loop, self.args.br // 20)(rng)
        
        idx = jnp.argmin(regrets)
        level = jax.tree_util.tree_map(lambda x: x[idx], levels)
        return level
    
    @jax.jit
    def get_eval_br(self, rng, train_state):
        """
        Take the trained LPG on the training nash. Then, 
        repeatedly sample envs and evaluate regret on them.
        """
        def _br_loop(rng):
            # --- Sample a level ---
            rng, _rng = jax.random.split(rng)
            params, lifetime = reset_env_params(_rng, self.env_name, self.env_mode)
            eval_level = Level(params, lifetime, 0)

            # --- Collect Evaluative Regrets ---
            rng, _rng = jax.random.split(rng)
            return eval_level, self._compute_algorithmic_regret(_rng, None, eval_level, train_state, True, True)
        
        rng = jax.random.split(rng, self.args.br)
        levels, regrets = mini_batch_vmap(_br_loop, self.args.br // 20)(rng)

        idx = jnp.argmax(regrets)
        level = jax.tree_util.tree_map(lambda x: x[idx], levels)

        return level, regrets[idx]
    
    def sample(self, rng, train_buffer, train_nash, old_agents, old_value_critics):
        # --- Check which agents' lifetimes are finished ---
        terminated_mask = old_agents.actor_state.step >= old_agents.level.lifetime
        term_mask_fn = lambda term_val, active_val: jax.vmap(jnp.where)(
            terminated_mask, term_val, active_val
        )
        # --- Sample new agents/value critics ---
        rng, _rng = jax.random.split(rng)
        agent_states, new_value_critics = self.get_training_levels(_rng, train_buffer, train_nash, terminated_mask.shape[0], not self.args.use_es)
    
        # --- Function mismatch fix trick ---
        agent_states = agent_states.replace(
            critic_state=agent_states.critic_state.replace(
                tx=old_agents.critic_state.tx, apply_fn=old_agents.critic_state.apply_fn
            ),
            actor_state=agent_states.actor_state.replace(
                tx=old_agents.actor_state.tx, apply_fn=old_agents.actor_state.apply_fn
            ),
        )
        if new_value_critics is not None:
            new_value_critics = new_value_critics.replace(
                tx=old_value_critics.tx, apply_fn=old_value_critics.apply_fn
            )
        
        agent_states = jax.tree_util.tree_map(term_mask_fn, agent_states, old_agents)
        value_critics = jax.tree_util.tree_map(term_mask_fn, new_value_critics, old_value_critics)
        return agent_states, value_critics