import jax
import jax.numpy as jnp

from .gd_sampler import GDSampler
from util.jax import pmap

class PLRSampler(GDSampler):

    def sample(
        self,
        rng, 
        level_buffer,
        train_levels,
        actor_state, 
        critic_state,
    ):
        batch_size = len(train_levels.buffer_id)
        rng, agent_rng, score_rng = jax.random.split(rng, 3)
        agent_rng = jax.random.split(agent_rng, batch_size)
    
        eval_agents = jax.vmap(self._create_eval_agent, in_axes=(0, 0, None, None))(agent_rng, train_levels, actor_state, critic_state)
        
        score_rng = jax.random.split(rng, batch_size)
        scores = pmap(
            self._compute_algorithmic_regret, self.num_mini_batches
        )(score_rng, eval_agents)

        # --- Replace regret scores in buffer ---
        level_buffer = level_buffer.replace(
            score = level_buffer.score.at[train_levels.buffer_id].set(scores),
            last_sampled = jnp.where(level_buffer.new, 0., level_buffer.last_sampled + 1)
        )

        # --- Calculate sample distribution
        total_num_sampled = len(level_buffer.new) - level_buffer.new.sum()
        staleness_probs = (
            (total_num_sampled - level_buffer.last_sampled)
            / (total_num_sampled - level_buffer.last_sampled).sum()
        )

        # --- Sample levels ---
        rng, _rng = jax.random.split(rng)
        replay_decision = jax.random.bernoulli(
            _rng, self.p_replay, shape=(batch_size, )
        )

        ranks = jnp.pow(1 / jax.scipy.stats.rankdata(level_buffer.score), 1 / self.score_temperature)
        sample_dist = ranks / ranks.sum()

        sample_dist = (1 - self.args.staleness_coeff) * sample_dist \
        + self.args.staleness_coeff * staleness_probs

        rng, sample_rng, rand_rng = jax.random.split(rng, 3)
        sampled_idxs = jax.random.choice(sample_rng, len(level_buffer), (batch_size, ), p=sample_dist)
        
        rand_dist = jnp.where(level_buffer.new, 1.0, 0.0)
        rand_dist = rand_dist / rand_dist.sum()
        random_idxs = jax.random.choice(rand_rng, len(level_buffer), (batch_size, ), p=rand_dist)

        level_idxs = jnp.where(replay_decision, sampled_idxs, random_idxs)
        new_levels = jax.tree_map(
            lambda x: x[level_idxs], level_buffer.level
        )

        level_buffer = level_buffer.replace(
            active = jnp.full_like(level_buffer.active, False).at[level_idxs].set(True),
            new = level_buffer.new.at[level_idxs].set(False),
            last_sampled = level_buffer.last_sampled.at[level_idxs].set(0.)
        )

        return new_levels, level_buffer

        



