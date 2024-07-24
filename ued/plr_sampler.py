import jax
import jax.numpy as jnp

from .gd_sampler import GDSampler
from util.jax import pmap


# NOTE: this is a variant that uses sampling with replacement. This may need to change.

class PLRSampler(GDSampler):

    def _replay_from_buffer(
        self, rng, level_buffer, batch_size: int, staleness_coeff:float, timestamp: int
    ):
        """
        Samples previously evaluated environment levels from the buffer.
        Levels are returned in sample order, which is significant for e.g. rank-based sampling.
        If there are not enough inactive, evaluated levels in the buffer, returns random levels.
        """
        invalid_levels = level_buffer.new # jnp.logical_or(level_buffer.new, level_buffer.active)

        scores = jnp.pow(1 / level_buffer.score, 1 / self.score_temperature)
        scores = jnp.where(invalid_levels, 0.0, scores)
        scores /= scores.sum()

        # Take mixture with staleness distribution
        staleness_logits = timestamp - level_buffer.last_sampled
        scores = (1 - staleness_coeff) * scores + staleness_coeff * (staleness_logits / staleness_logits.sum())
        
        # Return uniform (invalid) score when there aren't enough inactive, seen levels in buffer
        p_replay = jnp.where(
            self.buffer_size - jnp.sum(invalid_levels) < batch_size,
            jnp.ones_like(scores),
            scores,
        )
        rng, _rng = jax.random.split(rng)
        level_ids = jax.random.choice(_rng, len(level_buffer), (batch_size, ), p=p_replay)
        return jax.tree_map(lambda x: x[level_ids], level_buffer.level)

    def sample(
        self,
        rng, 
        level_buffer,
        train_levels,
        actor_state, 
        critic_state,
        timestamp
    ):
        batch_size = len(train_levels.buffer_id)
        rng, agent_rng, score_rng = jax.random.split(rng, 3)
        agent_rng = jax.random.split(agent_rng, batch_size)
    
        eval_agents = jax.vmap(self._create_eval_agent, in_axes=(0, 0, None, None))(agent_rng, train_levels, actor_state, critic_state)
        
        score_rng = jax.random.split(rng, batch_size)
        scores = pmap(
            self._compute_algorithmic_regret, self.num_mini_batches
        )(score_rng, eval_agents)

        swap_cond = level_buffer.score[train_levels.buffer_id] < scores
        old_levels = jax.tree_map(lambda x: x[train_levels.buffer_id], level_buffer.level)
        new_levels = jax.tree_map(
            lambda x, y: jax.vmap(jnp.where)(swap_cond, x, y), old_levels, train_levels
        )

        # --- Updating buffer from previous round ---
        level_buffer = level_buffer.replace(
            level = jax.tree_map(lambda x, y: x.at[new_levels.buffer_id].set(y), level_buffer.level, new_levels),
            score = level_buffer.score.at[train_levels.buffer_id].set(scores),
            active = jnp.full_like(level_buffer.active, False),
            last_sampled = level_buffer.last_sampled.at[train_levels.buffer_id].set(timestamp)
        )

        # --- Sample levels ---
        rng, _rng = jax.random.split(rng)
        replay_decision = jax.random.bernoulli(
            _rng, self.p_replay, shape=(batch_size, )
        )
        
        rng, replay_rng, rand_rng = jax.random.split(rng, 3)
        replay_levels = self._replay_from_buffer(replay_rng, level_buffer, batch_size, self.args.staleness_coeff, timestamp+1)
        random_levels = self._sample_random_from_buffer(rand_rng, level_buffer, batch_size)
        new_levels = jax.tree_map(
            lambda x, y: jax.vmap(jnp.where)(replay_decision, x, y), replay_levels, random_levels
        )
        return new_levels, level_buffer

        



