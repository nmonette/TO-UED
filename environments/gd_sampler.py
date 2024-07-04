import jax
import jax.numpy as jnp

from .level_sampler import LevelSampler
from ..util import *
from agents.lpg_agent import train_lpg_agent


class GDSampler(LevelSampler):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.lpg_hypers = LpgHyperparams.from_run_args(args)

    def _replay_from_buffer(
        self, rng, buffer, batch_size: int
    ):
        dist = jnp.where(buffer.seen, buffer.score, 0.)
        dist /= dist.sum()

        level_ids = jax.random.choice(
            rng,
            jnp.arange(self.buffer_size),
            p=buffer.score,
            shape=(batch_size,),
            replace=True, # Original paper has replace=False
        )

        return jax.tree_util.tree_map(
            lambda x: x[level_ids], buffer.levels
        )

    def sample(
        self, 
        rng, 
        train_state,
        train_buffer, 
        eval_buffer,
        prev_x_grad,
        prev_y_grad,
        prev_train_dist
    ):
        batch_size = self.args.batch_size
        rng, _rng, eval_rng = jax.random.split(rng, 3)
        score_rng = jax.random.split(_rng, batch_size)
        
        # --- Sample eval levels ---
        rng, replay_rng, random_rng = jax.random.split(rng, 3)
        replay_levels = self._replay_from_buffer(
            replay_rng, eval_buffer, batch_size
        )
        random_levels = self._sample_random_from_buffer(
            random_rng, eval_buffer, batch_size
        )

        # --- Select replay vs random levels ---
        rng, _rng = jax.random.split(rng)
        n_to_replay = jnp.sum(
            jax.random.bernoulli(_rng, self.p_replay, shape=(batch_size,))
        )
        use_replay = jnp.arange(batch_size) < n_to_replay
        n_replayable = self.buffer_size - jnp.sum(
            jnp.logical_or(eval_buffer.new, eval_buffer.active)
        )
        # Replay only when there are enough inactive, evaluated levels in buffer
        use_replay = jnp.logical_and(use_replay, n_replayable >= batch_size)
        rng, _rng = jax.random.split(rng)
        # Shuffle to remove bias
        use_replay = jax.random.permutation(_rng, use_replay)
        select_fn = lambda x, y: jax.vmap(jnp.where)(use_replay, x, y)
        # Select new levels from replay or random sets
        new_levels = jax.tree_util.tree_map(select_fn, replay_levels, random_levels)

        # --- Update active status of new levels in buffer ---
        eval_buffer = eval_buffer.replace(
            active=train_buffer.active.at[new_levels.buffer_id].set(True)
        )

        agent_rng = jax.random.split(eval_rng, batch_size)
        eval_agents = jax.vmap(self._create_agent)(agent_rng, new_levels)

        rng, _rng = jax.random.split(rng)
        train_rng = jax.random.split(rng, batch_size)

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
        
        eval_dist = jnp.unique(agents.level.buffer_id, return_counts=True, size=len(agents.level.buffer_id))[1]
        eval_dist = eval_dist / eval_dist.sum()

        eval_regret = jnp.dot(eval_dist, eval_regrets)

        # --- Update train buffer ---
        rng, replay_rng, random_rng = jax.random.split(rng, 3)
        replay_levels = self._replay_from_buffer(
            replay_rng, train_buffer, batch_size
        )
        random_levels = self._sample_random_from_buffer(
            random_rng, train_buffer, batch_size
        )

         # --- Select replay vs random levels ---
        rng, _rng = jax.random.split(rng)
        n_to_replay = jnp.sum(
            jax.random.bernoulli(_rng, self.p_replay, shape=(batch_size,))
        )
        use_replay = jnp.arange(batch_size) < n_to_replay
        rng, _rng = jax.random.split(rng)
        # Shuffle to remove bias
        use_replay = jax.random.permutation(_rng, use_replay)
        select_fn = lambda x, y: jax.vmap(jnp.where)(use_replay, x, y)
        # Select new levels from replay or random sets
        new_levels = jax.tree_util.tree_map(select_fn, replay_levels, random_levels)

        train_dist = jnp.unique(new_levels.buffer_id, return_counts=True, size=len(agents.level.buffer_id))[1]
        train_dist = train_dist / train_dist.sum()
        
        train_buffer = train_buffer.replace(
            score=projection_simplex_truncated(train_buffer.score + 0.001 * (prev_x_grad), 1e-6)
            
        )

        eval_buffer = eval_buffer.replace(
            score=eval_buffer.score.at[agents.level.buffer_id].set(eval_regrets),
            new=eval_buffer.new.at[agents.level.buffer_id].set(False)
        )