import jax
import jax.numpy as jnp
import chex
import distrax

from ued.level_sampler import LevelBuffer

from .level_sampler import LevelSampler
from .rnn import eval_agent
from util import *
from util.jax import pmap

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
        @partial(jax.grad, has_aux=True)
        def sample_action(policy, _rng, bern, buffer):
            unseen_total = buffer.new.sum()
            policy = distrax.Categorical(
                probs=jax.lax.select(jnp.logical_and(bern, unseen_total > 0), policy, jnp.where(buffer.new, 1 / unseen_total, 0.))
            )

            action = policy.sample(seed=_rng)
            log_prob = policy.log_prob(action)
            return log_prob, action
        
        rng, _rng = jax.random.split(rng)        
        bern = jax.random.bernoulli(_rng, self.p_replay, shape=(batch_size, ))

        rng = jax.random.split(rng, batch_size)
        lp, level_ids = jax.vmap(sample_action, in_axes=(None, 0, 0, None))(buffer.score, rng, bern, buffer)

        return lp, level_ids
    
    def sample(
        self, 
        rng, 
        train_buffer, 
        eval_buffer,
        prev_x_grad,
        prev_y_grad,
        actor_state, 
    ):
        # --- Calculate train and eval distributions ---
        x = projection_simplex_truncated(train_buffer.score + self.args.ogd_learning_rate * prev_x_grad, self.args.ogd_trunc_size)
        y = projection_simplex_truncated(eval_buffer.score + self.args.ogd_learning_rate * prev_y_grad, self.args.ogd_trunc_size)
       
        batch_size = self.args.num_agents
        rng, _rng, eval_rng = jax.random.split(rng, 3)
        score_rng = jax.random.split(_rng, batch_size)

        new_train = train_buffer.replace(score=x)

        if self.fixed_eval is None:
            new_eval = eval_buffer.replace(
            score=y
        ) 
        else:
            new_eval = eval_buffer.replace(
                score=self.fixed_eval
            ) 
        
        # --- Sample levels ---
        rng, train_rng, eval_rng = jax.random.split(rng, 3)
        train_buffer = self._reset_lowest_scoring(train_rng, train_buffer, batch_size)
        eval_buffer = self._reset_lowest_scoring(eval_rng, eval_buffer, batch_size)

        train_buffer = train_buffer.replace(
            score = projection_simplex_truncated(train_buffer.score, self.args.ogd_trunc_size)
        )

        eval_buffer = eval_buffer.replace(
            score = projection_simplex_truncated(eval_buffer.score, self.args.ogd_trunc_size)
        )

        rng, x_rng, y_rng = jax.random.split(rng, 3)
        x_lp, x_level_ids = self._sample_actions(x_rng, new_train, batch_size)
        y_lp, y_level_ids = self._sample_actions(y_rng, new_eval, batch_size)

        train_levels = jax.tree_util.tree_map(lambda x: x[x_level_ids], train_buffer.level)
        eval_levels = jax.tree_util.tree_map(lambda x: x[y_level_ids], eval_buffer.level)

        agent_rng = jax.random.split(eval_rng, batch_size)
    
        eval_agents = jax.vmap(self._create_eval_agent, in_axes=(0, 0, None))(agent_rng, eval_levels, actor_state)

        rng, _rng = jax.random.split(rng)
        
        eval_regrets = pmap(
            self._compute_algorithmic_regret, self.num_mini_batches
        )(score_rng, eval_agents)
        
        eval_dist = jnp.unique(y_level_ids, return_counts=True, size=len(eval_agents.level.buffer_id))[1]
        eval_dist = eval_dist / eval_dist.sum()

        eval_regret = jnp.dot(eval_dist, eval_regrets)

        x_grad = -(x_lp * eval_regret).mean(axis=0)
        y_grad = (y_lp * eval_regret).mean(axis=0)

        # --- Update buffers for next round of sampling ---
        train_buffer = train_buffer.replace(
            score = projection_simplex_truncated(train_buffer.score + self.args.ogd_learning_rate * x_grad, self.args.ogd_trunc_size),
            new = train_buffer.new.at[x_level_ids].set(False)
        ) 
        eval_buffer = eval_buffer.replace(
            score = projection_simplex_truncated(eval_buffer.score + self.args.ogd_learning_rate * y_grad, self.args.ogd_trunc_size),
            new = eval_buffer.new.at[y_level_ids].set(False)
        )
        
        return train_buffer, eval_buffer, train_levels, eval_levels, x_grad, y_grad, eval_regret