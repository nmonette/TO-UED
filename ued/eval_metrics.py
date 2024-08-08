import jax
import jax.numpy as jnp

from ued.rnn import eval_agent_nomean as eval_agent, Actor
from environments.jaxued.maze import Level as MazeLevel, prefabs
from util.data import Level

def make_eval(args):

    holdout_levels = Level(
        env_params = MazeLevel.load_prefabs([
            "SixteenRooms",
            "SixteenRooms2",
            "Labyrinth",
            "LabyrinthFlipped",
            "Labyrinth2",
            "StandardMaze",
            "StandardMaze2",
            "StandardMaze3",
        ]), 
        lifetime = jnp.full(8, 2500),
        buffer_id = jnp.arange(8)
    )

    def _eval_fn(
        actor_state,
        dummy_sampler,
        level_sampler,
        level_buffer,
        eval_buffer,
        x, 
        y,
    ):
        # --- Collecting return on sampled train set levels ---
        rng, _rng = jax.random.split(rng)
        train_metric_level_idxs = jax.random.choice(rng, args.buffer_size, (16, ), p=x)
        train_metric_levels = jax.tree_util.tree_map(lambda x: x[train_metric_level_idxs], level_buffer.level)

        train_metric_hstate = Actor.initialize_carry((16, args.env_workers, ))
        rng, _rng = jax.random.split(rng)
        _rng = jax.random.split(_rng, 16)


        metrics["return/train_mean"] = jax.vmap(
            lambda r, e, a, ew, hs: eval_agent(r, level_sampler.rollout_manager, e, a, ew, hs),
            in_axes=(0, 0, None, None, 0)
        )(
            _rng, train_metric_levels.env_params, actor_state, args.env_workers, train_metric_hstate
        ).mean()

        # --- Collecting return on sampled eval set levels ---
        rng, _rng = jax.random.split(rng)
        eval_metric_level_idxs = jax.random.choice(rng, args.buffer_size, (16, ), p=y)
        eval_metric_levels = jax.tree_util.tree_map(lambda x: x[eval_metric_level_idxs], eval_buffer.level)

        eval_metric_hstate = Actor.initialize_carry((16, args.env_workers, ))
        rng, _rng = jax.random.split(rng)
        _rng = jax.random.split(_rng, 16)


        metrics["return/eval_mean"] = jax.vmap(
            lambda r, e, a, ew, hs: eval_agent(r, level_sampler.rollout_manager, e, a, ew, hs),
            in_axes=(0, 0, None, None, 0)
        )(
            _rng, eval_metric_levels.env_params, actor_state, args.env_workers, eval_metric_hstate
        ).mean()

        # --- Collecting return on randomly sampled levels ---
        rng, _rng = jax.random.split(rng)
        test_buffer = dummy_sampler.initialize_buffer(_rng)
        test_env_params = jax.tree_util.tree_map(lambda x: x[:16], test_buffer.level.env_params)
        test_hstates = Actor.initialize_carry((16, args.env_workers, ))

        rng, _rng = jax.random.split(rng)
        _rng = jax.random.split(_rng, 16)

        metrics["return/random_mean"] = jax.vmap(
            lambda r, e, a, ew, hs: eval_agent(r, level_sampler.rollout_manager, e, a, ew, hs),
            in_axes=(0, 0, None, None, 0)
        )(
            _rng, test_env_params, actor_state, args.env_workers, test_hstates
        ).mean()

        # --- Collecting return on the holdout set level ---
        eval_hstates = Actor.initialize_carry((8, args.env_workers, ))
        rng, _rng = jax.random.split(rng)
        _rng = jax.random.split(_rng, 8)


        agent_return_on_holdout_set = jax.vmap(
            lambda r, e, a, ew, hs: eval_agent(r, level_sampler.rollout_manager, e, a, ew, hs),
            in_axes=(0, 0, None, None, 0)
        )(
            _rng, holdout_levels.env_params, actor_state, args.env_workers, eval_hstates
        )

        metrics["return/holdout_mean"] = agent_return_on_holdout_set.mean()
        holdout_set_success_rate = jnp.where(agent_return_on_holdout_set > 0, 1, 0)
        metrics["solve_rate/holdout_mean"] = holdout_set_success_rate.sum() / jnp.size(holdout_set_success_rate)

        metrics = {
            **metrics,
            "return/SixteenRooms":agent_return_on_holdout_set[0].mean(),
            "solve_rate/SixteenRooms":holdout_set_success_rate[0].sum() / len(holdout_set_success_rate[0]),

            "return/SixteenRooms2":agent_return_on_holdout_set[1].mean(),
            "solve_rate/SixteenRooms2":holdout_set_success_rate[1].sum() / len(holdout_set_success_rate[0]),

            "return/Labyrinth":agent_return_on_holdout_set[2].mean(),
            "solve_rate/Labyrinth":holdout_set_success_rate[2].sum() / len(holdout_set_success_rate[0]),

            "return/LabyrinthFlipped":agent_return_on_holdout_set[3].mean(),
            "solve_rate/LabyrinthFlipped":holdout_set_success_rate[3].sum() / len(holdout_set_success_rate[0]),

            "return/Labyrinth2":agent_return_on_holdout_set[4].mean(),
            "solve_rate/Labyrinth2":holdout_set_success_rate[4].sum() / len(holdout_set_success_rate[0]),

            "return/StandardMaze":agent_return_on_holdout_set[5].mean(),
            "solve_rate/StandardMaze":holdout_set_success_rate[5].sum() / len(holdout_set_success_rate[0]),

            "return/StandardMaze2":agent_return_on_holdout_set[6].mean(),
            "solve_rate/StandardMaze2":holdout_set_success_rate[6].sum() / len(holdout_set_success_rate[0]),

            "return/StandardMaze3":agent_return_on_holdout_set[7].mean(),
            "solve_rate/StandardMaze3":holdout_set_success_rate[7].sum() / len(holdout_set_success_rate[0]),
        }

        return metrics
        
    return _eval_fn