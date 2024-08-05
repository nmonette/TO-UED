import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jax.experimental.pjit import pjit
from rich.traceback import install
import numpy as np

from ued.gd_sampler import GDSampler
from ued.level_sampler import LevelSampler
from ued.train import train_agent
from ued.rnn import eval_agent_nomean as eval_agent, Actor
from experiments.parse_args import parse_args
from experiments.logging import init_logger, log_results
from util.jax import jax_debug_wrapper
from util.data import Level
from environments.jaxued.maze import Level as MazeLevel, prefabs

from functools import partial
import sys

def make_train(args, eval_args):
    def _train_fn(rng):
        # --- Initialize policy and level buffers and samplers ---
        rng, agent_rng, dummy_rng, buffer_rng, holdout_rng = jax.random.split(rng, 5)

        level_sampler = GDSampler(eval_args)
        dummy_sampler = LevelSampler(args)
        
        level_buffer = dummy_sampler.initialize_buffer(dummy_rng)
        eval_buffer = level_sampler.initialize_buffer(buffer_rng)

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

        init_agent = dummy_sampler._create_agent(
            agent_rng, jax.tree_util.tree_map(lambda x: x[0], level_buffer.level), True
        ) 

        actor_state = init_agent.actor_state

        train_agent_fn = partial(
            train_agent, 
            num_epochs=args.num_epochs,
            num_mini_batches=args.num_mini_batches,
            num_workers=args.env_workers,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_eps=args.clip_eps,
            critic_coeff=args.critic_coeff,
            entropy_coeff=args.ppo_entropy_coeff
        )
        
        # --- TRAIN LOOP ---
        def _ued_train_loop(carry, t):
            rng, actor_state, level_buffer, eval_buffer, x_grad, y_grad, \
                actor_hstate, init_obs, init_state = carry

            # --- Sample new levels ---
            rng, _rng = jax.random.split(rng)
            level_buffer, eval_buffer, eval_levels, x, y_lp = GDSampler.sample_step(
                GDSampler(args), _rng, level_buffer, eval_buffer, x_grad, y_grad
            )

            # --- Add new level dist to level sampler ---
            # next line: replace `x` with `level_buffer.score` and it might do better
            reset_dist = args.p_replay * x + (1 - args.p_replay) * jnp.where(level_buffer.new, 1 / level_buffer.new.sum(), 0.)
            level_sampler = GDSampler(
                args, None, reset_dist, level_buffer.level.env_params
            )
            
            # --- Train agents on sampled levels ---
            rng, _rng = jax.random.split(rng)
            (actor_state, actor_hstate, init_obs, init_state), metrics = train_agent_fn(
                rng=_rng,
                # NOTE: we should be using dummy_sampler here in order to be compatible with differing train and eval distributions 
                rollout_manager=level_sampler.rollout_manager,
                actor_state=actor_state,
                # TODO: figure out if this needs to be a reprsentative set or not,
                #       or if this is only taking the stationary EnvParams
                env_params=init_train_levels.env_params, 
                actor_hstate=actor_hstate,
                init_obs=init_obs, 
                init_state=init_state, 
            )

            # --- Calculate resulting train distribution ---
            count_fn = lambda idx: jnp.zeros(args.buffer_size).at[init_state.prev_idx[idx]].set(jax.lax.select(init_state.newly_done[idx], 1., 0.))
            done_counts = jax.vmap(count_fn)(jnp.arange(len(init_state.newly_done))).sum(axis=0)

            grad_fn = jax.grad(lambda x, idx: jnp.log(x[idx]) * done_counts[idx])
            x_lp = jax.vmap(grad_fn, in_axes=(None, 0))(reset_dist, jnp.arange(len(level_buffer)))
            
            # --- Update level buffers ---
            rng, _rng = jax.random.split(rng)
            level_buffer, eval_buffer, x_grad, y_grad, eval_regret = level_sampler.eval_step(
                _rng, actor_state, eval_levels, level_buffer, eval_buffer, x_lp, y_lp, x_grad, y_grad
            )

            metrics["eval_regret"] = eval_regret

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
            
            carry = (rng, actor_state, level_buffer, eval_buffer, \
                x_grad, y_grad, actor_hstate, init_obs, init_state)
            
            return carry, metrics
        
        # --- Initialize train_levels, eval_levels, hstates ---
        rng, train_rng, eval_rng = jax.random.split(rng, 3)
        train_idxs = jax.random.choice(train_rng, len(level_buffer), (args.num_agents, ))
        eval_idxs = jax.random.choice(eval_rng, len(eval_buffer), (args.num_agents, ))
        train_fn = lambda x: x[train_idxs]
        eval_fn = lambda x: x[eval_idxs]
        init_train_levels = jax.tree_util.tree_map(train_fn, level_buffer.level)
        init_eval_levels = jax.tree_util.tree_map(eval_fn, eval_buffer.level)

        level_buffer = level_buffer.replace(
            new = level_buffer.new.at[init_train_levels.buffer_id].set(False)
        )

        eval_buffer = eval_buffer.replace(
            new = eval_buffer.new.at[init_eval_levels.buffer_id].set(False)
        )

        rng, _rng = jax.random.split(rng)
        zeros = jnp.zeros_like(level_buffer.score)
        level_buffer, eval_buffer, init_train_levels, init_eval_levels, x_grad, y_grad, eval_regret = level_sampler.sample(
            _rng, level_buffer, eval_buffer, zeros, zeros, actor_state
        )

        level_sampler = GDSampler(
            args, None, level_buffer.score, level_buffer.level.env_params
        )

        # NOTE: batch_reset has been modified to accept a batch of env_params
        rng, _rng = jax.random.split(rng)
        init_obs, init_state = level_sampler.rollout_manager.batch_reset(_rng, init_train_levels.env_params)
        hstate = Actor.initialize_carry(init_state.time.shape)

        # --- Stack and return metrics ---
        carry = (rng, actor_state, level_buffer, eval_buffer, x_grad, y_grad, \
                hstate, init_obs, init_state)
        carry, metrics = jax.lax.scan(
            _ued_train_loop, carry, jnp.arange(args.train_steps), args.train_steps
        )
        return metrics, actor_state, level_buffer, eval_buffer

    return _train_fn


def run_training_experiment(args, eval_args):
    if args.log:
        init_logger(args)
    train_fn = make_train(args, eval_args)
    rng = jax.random.PRNGKey(args.seed)
    with Mesh(np.array(jax.devices()), ("devices", )):
        metrics, actor_state, level_buffer, eval_buffer = jax.jit(train_fn, in_shardings=None, out_shardings=None)(rng)
    if args.log:
        log_results(args, metrics, actor_state, level_buffer)
    else:
        print(metrics)


def main(cmd_args=sys.argv[1:]):
    args = parse_args(cmd_args)
    eval_args = parse_args(cmd_args)
    eval_args.env_mode = eval_args.eval_env_mode
    eval_args.env_name = eval_args.eval_env_name
    eval_args.train_env_mode = args.env_mode
    eval_args.train_env_name = args.env_name
    experiment_fn = jax_debug_wrapper(args, run_training_experiment)
    return experiment_fn(args, eval_args)


if __name__ == "__main__":
    # install()
    main()