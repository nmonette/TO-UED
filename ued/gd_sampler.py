import jax
import jax.numpy as jnp
import chex
from ued.level_sampler import LevelBuffer

from .level_sampler import LevelSampler
from .rnn import eval_agent
from util import *
from util.jax import pmap

from functools import partial

class GDSampler(LevelSampler):
    def __init__(self, args, fixed_eval = None):
        super().__init__(args)
        self.args = args
        self.lpg_hypers = LpgHyperparams.from_run_args(args)
        self.fixed_eval = fixed_eval

    def _create_eval_agent(self, rng, level, actor_state, critic_state=None):
        """Initialise an agent on the given level."""
        env_obs, env_state = self.rollout_manager.batch_reset_single_env(
            rng, level.env_params, self.env_workers
        )
        return AgentState(
            actor_state=actor_state,
            critic_state=critic_state,
            level=level,
            env_obs=env_obs,
            env_state=env_state,
        )
    
    def _sample_actions(
        self, 
        rng, 
        buffer,
        batch_size
    ):
        # @partial(jax.grad, has_aux=True)
        def sample_action(policy, _rng, bern, buffer):
            unseen_total = buffer.new.sum()
            policy = jax.lax.select(jnp.logical_and(bern, unseen_total > 0), policy, jnp.where(buffer.new, 1 / unseen_total, 0.))

            action = jax.random.choice(_rng, jnp.arange(len(buffer)), p=policy)

            # Returning the gradient (see: REINFORCE)
            return action # jnp.log(policy[action] + 1e-6), action
        
        rng, _rng = jax.random.split(rng)        
        bern = jax.random.bernoulli(_rng, self.p_replay, shape=(batch_size, ))

        rng = jax.random.split(rng, batch_size)
        level_ids = jax.vmap(sample_action, in_axes=(None, 0, 0, None))(buffer.score, rng, bern, buffer)

        return level_ids
    
    def sample(
        self, 
        rng, 
        train_buffer, 
        eval_buffer,
        train_levels,
        eval_levels,
        actor_state, 
        critic_state,
        timestamp
    ):
        batch_size = self.args.num_agents
        rng, _rng, eval_rng = jax.random.split(rng, 3)
        score_rng = jax.random.split(_rng, batch_size)

        # --- Calculate regret scores (agents observe a reward) ---
        agent_rng = jax.random.split(eval_rng, batch_size)
    
        eval_agents = jax.vmap(self._create_eval_agent, in_axes=(0, 0, None, None))(agent_rng, eval_levels, actor_state, critic_state)

        rng, _rng = jax.random.split(rng)
        
        eval_regrets = pmap(
            self._compute_algorithmic_regret, self.num_mini_batches
        )(score_rng, eval_agents)
        
        eval_dist = jnp.unique(eval_levels.buffer_id, return_counts=True, size=len(eval_agents.level.buffer_id))[1]
        eval_dist = eval_dist / eval_dist.sum()

        eval_regret = jnp.dot(eval_dist, eval_regrets)

        # --- Update sampling distributions ---
        eta = jnp.pow(timestamp, -1/2)
        eps = jnp.pow(timestamp, -1/6)

        def argmin(carry, _):
            x, xt, g = carry

            @jax.grad
            def grad_fn(x, g):
                g = g / x + eps * jnp.log(x + 1e-8)
                return x.T @ g + (1 / eta) * kl_divergence(x, xt)

            return (projection_simplex_truncated(
                x - 0.01 * grad_fn(x, g), 1 / (len(train_buffer) * jnp.square(timestamp)),
            ), xt, g), None
        
        x_g = jnp.zeros(len(train_buffer)).at[train_levels.buffer_id].set(eval_regret)
        y_g = jnp.zeros(len(train_buffer)).at[eval_levels.buffer_id].set(-eval_regret)

        (x, _, _), _ = jax.lax.scan(argmin, (jnp.full_like(train_buffer.score, 1 / len(train_buffer)), train_buffer.score, x_g), length=1000)
        (y, _, _), _ = jax.lax.scan(argmin, (jnp.full_like(train_buffer.score, 1 / len(train_buffer)), train_buffer.score, y_g), length=1000)

        # --- Update buffers for next round of sampling ---
        train_buffer = train_buffer.replace(
            score = x,
            new = train_buffer.new.at[train_levels.buffer_id].set(False)
        ) 
        eval_buffer = eval_buffer.replace(
            score = y,
            new = train_buffer.new.at[train_levels.buffer_id].set(False)
        )

        # --- Sample levels ---
        rng, train_rng, eval_rng = jax.random.split(rng, 3)
        train_buffer = self._reset_lowest_scoring(train_rng, train_buffer, batch_size)
        eval_buffer = self._reset_lowest_scoring(eval_rng, eval_buffer, batch_size)

        rng, x_rng, y_rng = jax.random.split(rng, 3)
        x_level_ids = self._sample_actions(x_rng, train_buffer, batch_size)
        y_level_ids = self._sample_actions(y_rng, eval_buffer, batch_size)

        train_levels = jax.tree_util.tree_map(lambda x: x[x_level_ids], train_buffer.level)
        eval_levels = jax.tree_util.tree_map(lambda x: x[y_level_ids], eval_buffer.level)

        return train_buffer, eval_buffer, train_levels, eval_levels, eval_regret


        