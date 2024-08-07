import jax
import jax.numpy as jnp
import chex
import distrax

from ued.level_sampler import LevelBuffer

from .level_sampler import LevelSampler
from environments.environments import get_env
from .rnn import eval_agent
from .rollout import RolloutWrapper
from util import *
from util.jax import pmap

class GDSampler(LevelSampler):
    def __init__(self, args, fixed_eval = None, train_dist = None, levels = None):
        super().__init__(args)
        self.args = args
        self.lpg_hypers = LpgHyperparams.from_run_args(args)
        self.fixed_eval = fixed_eval

        if train_dist is not None and args.env_reset_method != "reset":
            self.env = get_env(self.env_name, self.env_kwargs, train_dist, levels, replay = (args.env_reset_method == "replay") )
            self.rollout_manager = RolloutWrapper(
                self.env_name, 
                self.args.train_rollout_len, 
                self.max_rollout_len, 
                env_kwargs=self.env_kwargs, 
                env=self.env
            )

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
    
    def _reset_lowest_scoring(
        self, rng: chex.PRNGKey, level_buffer: LevelBuffer, minimum_new: int
    ):
        """
        Reset the lowest scoring levels in the buffer.
        Ensures there are at least minimum_new new, inactive levels.
        """
        # --- Identify lowest scoring levels ---
        level_scores = jnp.where(level_buffer.new, -jnp.inf, level_buffer.score)
        level_scores = jnp.where(level_buffer.active, jnp.inf, level_scores)
        reset_ids = jnp.argsort(level_scores)[:minimum_new]
        rng = jax.random.split(rng, minimum_new)
        new_params, new_lifetimes = self._sample_env_params(rng)
        new_levels = Level(new_params, new_lifetimes, reset_ids)

        # --- Reset lowest scoring levels ---
        reset_fn = lambda x, y: x.at[reset_ids].set(y)
        return level_buffer.replace(
            level=jax.tree_util.tree_map(reset_fn, level_buffer.level, new_levels),
            score=projection_simplex_truncated(level_buffer.score.at[reset_ids].set(0.0), self.args.ogd_trunc_size),
            active=level_buffer.active.at[reset_ids].set(False),
            new=level_buffer.new.at[reset_ids].set(True),
        )
    
    def sample_step(
        self, 
        rng, 
        train_buffer, 
        eval_buffer, 
        x,
        y
    ):
        """
        Line 1, then replenish buffers, and then line 2

        NOTE: a lot of the operations for x/the train buffer here are not actually used, this is just an artifact of the past.
        TODO: make x and y the sample dist instead of xhat and yhat
        """
        batch_size = self.args.num_agents
        
        # --- Sample new levels ---
        new_train = train_buffer.replace(score=x)
        if self.fixed_eval is None:
            new_eval = eval_buffer.replace(
                score=y
            ) 
        else:
            new_eval = eval_buffer.replace(
                score=self.fixed_eval
            ) 

        rng, y_rng = jax.random.split(rng)
        y_lp, y_level_ids = self._sample_actions(y_rng, new_eval, batch_size)

        eval_levels = jax.tree_util.tree_map(lambda x: x[y_level_ids], eval_buffer.level)

        # --- Update buffers back to xhat, yhat, but with the new levels ---
        train_buffer = new_train.replace(
            score = projection_simplex_truncated(
                jnp.where(new_train.new, 0., train_buffer.score), self.args.ogd_trunc_size
            ),
            # new = new_train.new.at[train_levels.buffer_id].set(False)
        )

        eval_buffer = new_eval.replace(
            score = projection_simplex_truncated(
                jnp.where(new_eval.new, 0., eval_buffer.score), self.args.ogd_trunc_size
            ),
            new = new_eval.new.at[eval_levels.buffer_id].set(False)
        )

        return train_buffer, eval_buffer, eval_levels, new_train.score, y_lp.mean(axis=0)
        

    def eval_step(
        self, 
        rng, 
        actor_state, 
        eval_levels,
        train_buffer, 
        eval_buffer
    ):
        """
        Lines 4-6
        """
        # --- Calculate evaluative regret ---
        batch_size = self.args.num_agents
        rng, _rng = jax.random.split(rng)
        agent_rng = jax.random.split(_rng, batch_size)
    
        eval_agents = jax.vmap(self._create_eval_agent, in_axes=(0, 0, None))(agent_rng, eval_levels, actor_state)

        rng, _rng = jax.random.split(rng)
        score_rng = jax.random.split(_rng, batch_size)

        eval_regret = pmap(
            self._compute_algorithmic_regret, self.num_mini_batches
        )(score_rng, eval_agents).mean()

        return train_buffer, eval_buffer, eval_regret