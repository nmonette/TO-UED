import jax
import jax.numpy as jnp
import chex

from .level_sampler import LevelSampler, LevelBuffer
from util import *
from agents.lpg_agent import train_lpg_agent
from agents.agents import create_value_critic
from .environments import reset_env_params


class GDSampler(LevelSampler):
    def __init__(self, args, fixed_eval = None):
        super().__init__(args)
        self.args = args
        self.lpg_hypers = LpgHyperparams.from_run_args(args)
        self.fixed_eval = fixed_eval

    @partial(jax.vmap, in_axes=(None, 0, None))
    def _sample_env_params(self, rng, env_mode = None):
        """Sample a batch of environment parameters and agent lifetimes."""
        if env_mode is None:
            env_mode = self.env_mode
        return reset_env_params(rng, self.env_name, env_mode)
    
    def _sample_random_levels(self, rng: chex.PRNGKey, batch_size: int):
        rng = jax.random.split(rng, batch_size)
        new_params, new_lifetimes = self._sample_env_params(rng, None)
        # TODO: figure out why the buffer id's are all zeros here (idk)
        return Level(new_params, new_lifetimes, jnp.zeros(batch_size, dtype=int))

    def initialize_buffer(self, rng):
        """Creates a new level buffer, if used by the score function."""
        train_rng, eval_rng = jax.random.split(rng)
        train_rng = jax.random.split(train_rng, self.buffer_size)
        random_params, random_lifetimes = self._sample_env_params(train_rng, self.args.env_mode)
        train_buffer = LevelBuffer.create_buffer(random_params, random_lifetimes)

        eval_rng = jax.random.split(eval_rng, self.buffer_size)
        random_params, random_lifetimes = self._sample_env_params(eval_rng, self.args.eval_env_mode)
        eval_buffer = LevelBuffer.create_buffer(random_params, random_lifetimes)

        return train_buffer, eval_buffer

    def _sample_actions(
        self, 
        rng, 
        buffer,
        batch_size
    ):
        
        @partial(jax.grad, has_aux=True)
        def sample_action(policy, _rng, bern, buffer):
            unseen_total = buffer.new.sum()
            policy = jax.lax.select(jnp.logical_and(bern, unseen_total > 0), jnp.where(buffer.new, 1 / unseen_total, 0.), policy)
            policy = jax.lax.select(jnp.logical_and(bern, unseen_total == 0), jnp.full_like(policy, 1 / len(buffer)), policy)

            action = jax.random.choice(_rng, jnp.arange(len(buffer)), p=policy)

            # Returning the gradient (see: REINFORCE)
            return jnp.log(policy[action] + 1e-6), action
        
        rng, _rng = jax.random.split(rng)        
        bern = jax.random.bernoulli(_rng, self.p_replay, shape=(batch_size, ))

        rng = jax.random.split(rng, batch_size)
        lp, level_ids = jax.vmap(sample_action, in_axes=(None, 0, 0, None))(buffer.score, rng, bern, buffer)

        return lp, level_ids
    
    def sample(
        self, 
        rng, 
        train_state,
        train_buffer, 
        eval_buffer,
        prev_x_grad,
        prev_y_grad,
        old_agents, 
        old_value_critics = None
    ):
        # --- Calculate train and eval distributions ---
        x = projection_simplex_truncated(train_buffer.score + self.args.ogd_learning_rate * prev_x_grad, self.args.ogd_trunc_size)
        y = projection_simplex_truncated(eval_buffer.score + self.args.ogd_learning_rate * prev_y_grad, self.args.ogd_trunc_size)
       
        batch_size = self.args.num_agents
        rng, _rng, eval_rng = jax.random.split(rng, 3)
        score_rng = jax.random.split(_rng, batch_size)

        new_train = train_buffer.replace(score=x)
        new_eval = eval_buffer.replace(
            score=jax.lax.select(self.fixed_eval is None, y, self.fixed_eval)
        ) 
        
        # --- Sample levels ---
        rng, x_rng, y_rng = jax.random.split(rng, 3)
        x_lp, x_level_ids = self._sample_actions(x_rng, new_train, batch_size)
        y_lp, y_level_ids = self._sample_actions(y_rng, new_eval, batch_size)

        train_levels = jax.tree_util.tree_map(lambda x: x[x_level_ids], train_buffer.level)
        eval_levels = jax.tree_util.tree_map(lambda x: x[y_level_ids], eval_buffer.level)

        agent_rng = jax.random.split(eval_rng, batch_size)
        eval_agents = jax.vmap(self._create_agent)(agent_rng, eval_levels)

        rng, _rng = jax.random.split(rng)
        train_rng = jax.random.split(rng, batch_size)

        # --- Get eval regret scores ---
        agents, _, _  = mini_batch_vmap(
            lambda r, a: train_lpg_agent(
                r,
                train_state,
                a,
                self.rollout_manager,
                self.lpg_hypers.num_agent_updates,
                self.lpg_hypers.agent_target_coeff), 
            self.num_mini_batches,)(
            train_rng, 
            eval_agents,
        )
        
        eval_regrets = mini_batch_vmap(
            self._compute_algorithmic_regret, self.num_mini_batches
        )(score_rng, agents)
        
        eval_dist = jnp.unique(y_level_ids, return_counts=True, size=len(agents.level.buffer_id))[1]
        eval_dist = eval_dist / eval_dist.sum()

        eval_regret = jnp.dot(eval_dist, eval_regrets)

        x_grad = -(x_lp * eval_regret).mean(axis=0)
        y_grad = (y_lp * eval_regret).mean(axis=0)

        # --- Update buffers for next round of sampling ---
        train_buffer = train_buffer.replace(
            score = projection_simplex_truncated(train_buffer.score + self.args.ogd_learning_rate * x_grad, self.args.ogd_trunc_size),
            new = train_buffer.new.at[train_levels.buffer_id].set(False)
        ) 
        eval_buffer = eval_buffer.replace(
            score = projection_simplex_truncated(eval_buffer.score + self.args.ogd_learning_rate * y_grad, self.args.ogd_trunc_size),
            new = train_buffer.new.at[eval_levels.buffer_id].set(False)
        )

        # --- Initialise new agents and environment workers ---
        rng, _rng = jax.random.split(rng)
        _rng = jax.random.split(_rng, batch_size)
        agent_states = jax.vmap(self._create_agent)(_rng, train_levels)

        # --- Initialise new value critics (if required) ---
        new_value_critics = None
        if old_value_critics is not None:
            rng, _rng = jax.random.split(rng)
            _rng = jax.random.split(_rng, batch_size)
            new_value_critics = jax.vmap(create_value_critic, in_axes=(0, None, None))(
                _rng, self.agent_hypers, self.obs_shape
            )

        # --- Hack to fix function mismatch ---
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

        return train_buffer, eval_buffer, x_grad, y_grad, agent_states, new_value_critics