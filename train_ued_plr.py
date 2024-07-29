import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jax.experimental.pjit import pjit
from rich.traceback import install
import numpy as np

from ued.plr_sampler import PLRSampler
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

        level_sampler = PLRSampler(args)
        
        level_buffer = level_sampler.initialize_buffer(dummy_rng)

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

        init_agent = level_sampler._create_agent(
            agent_rng, jax.tree_util.tree_map(lambda x: x[0], level_buffer.level), True
        ) 

        actor_state = init_agent.actor_state
        critic_state = init_agent.critic_state

        train_agent_fn = partial(
            train_agent, 
            rollout_manager=level_sampler.rollout_manager,
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
            rng, actor_state, critic_state, level_buffer, \
                train_levels, x_grad, y_grad, \
                    actor_hstate, critic_hstate, init_obs, init_state = carry
            
            # --- Train agents on sampled levels ---
            rng, _rng = jax.random.split(rng)
            (actor_state, critic_state, actor_hstate, critic_hstate, init_obs, init_state), metrics = train_agent_fn(
                rng=_rng,
                actor_state=actor_state,
                critic_state=critic_state,
                env_params=train_levels.env_params, 
                actor_hstate=actor_hstate,
                critic_hstate=critic_hstate,
                init_obs=init_obs, 
                init_state=init_state, 
            )
            
            # --- Sample new levels as required ---
            def sample(rng, level_buffer, train_levels, actor_state, critic_state):
                rng, _rng = jax.random.split(rng)
                train_levels, level_buffer = level_sampler.sample( 
                    _rng, 
                    level_buffer,
                    train_levels,
                    actor_state, 
                    critic_state,
                    t
                )

                rng, _rng = jax.random.split(rng)
                init_obs, init_state = level_sampler.rollout_manager.batch_reset(_rng, init_train_levels.env_params)
                hstate = Actor.initialize_carry(init_state.time.shape)
                return level_buffer, train_levels, init_obs, init_state, hstate, hstate
            
            def identity(rng, level_buffer, train_levels, actor_state, critic_state):
                return level_buffer, train_levels, init_obs, init_state, hstate, hstate

            rng, _rng = jax.random.split(rng)
            level_buffer, train_levels, init_obs, init_state, actor_hstate, critic_hstate = jax.lax.cond(
                t % args.regret_frequency == 0, sample, identity, _rng, level_buffer, train_levels, actor_state, critic_state
            )

            # --- Collecting return on sampled train set levels ---
            rng, _rng = jax.random.split(rng)
            train_metric_level_idxs = jax.random.choice(rng, args.num_agents, (16, ))
            train_metric_levels = jax.tree_util.tree_map(lambda x: x[train_metric_level_idxs], train_levels)

            train_metric_hstate = Actor.initialize_carry((16, args.env_workers, ))
            rng, _rng = jax.random.split(rng)
            _rng = jax.random.split(_rng, 16)

            metrics["agent_return_on_train_set"] = jax.vmap(
                lambda r, e, a, ew, hs: eval_agent(r, level_sampler.rollout_manager, e, a, ew, hs),
                in_axes=(0, 0, None, None, 0)
            )(
                _rng, train_metric_levels.env_params, actor_state, args.env_workers, train_metric_hstate
            ).mean()

            # --- Collecting return on randomly sampled levels ---
            rng, _rng = jax.random.split(rng)
            test_buffer = level_sampler.initialize_buffer(_rng)
            test_env_params = jax.tree_util.tree_map(lambda x: x[:16], test_buffer.level.env_params)
            test_hstates = Actor.initialize_carry((16, args.env_workers, ))
            
            rng, _rng = jax.random.split(rng)
            _rng = jax.random.split(_rng, 16)

            metrics["agent_return_on_random_set"] = jax.vmap(
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

            metrics = {
                **metrics,
                "SixteenRooms_return":agent_return_on_holdout_set[0].mean(),
                "SixteenRooms2_return":agent_return_on_holdout_set[1].mean(),
                "Labyrinth_return":agent_return_on_holdout_set[2].mean(),
                "LabyrinthFlipped_return":agent_return_on_holdout_set[3].mean(),
                "Labyrinth2_return":agent_return_on_holdout_set[4].mean(),
                "StandardMaze_return":agent_return_on_holdout_set[5].mean(),
                "StandardMaze2_return":agent_return_on_holdout_set[6].mean(),
                "StandardMaze3_return":agent_return_on_holdout_set[7].mean(),
            }

            metrics["agent_return_on_holdout_set"] = agent_return_on_holdout_set.mean()
            holdout_set_success_rate = jnp.where(agent_return_on_holdout_set > 0, 1, 0)
            holdout_set_success_rate = holdout_set_success_rate.sum() / jnp.size(holdout_set_success_rate)
            metrics["holdout_set_success_rate"] = holdout_set_success_rate
            
            carry = (rng, actor_state, critic_state, level_buffer, \
                train_levels, x_grad, y_grad, actor_hstate, critic_hstate, init_obs, init_state)
            
            return carry, metrics
        
        # Initialize train_levels, eval_levels, hstates

        tile_fn = lambda x: jnp.array([x[0]]).repeat(args.num_agents, axis=0).squeeze()
        init_train_levels = jax.tree_util.tree_map(tile_fn, level_buffer.level)

        # NOTE: batch_reset has been modified to accept a batch of env_params
        rng, _rng = jax.random.split(rng)
        init_obs, init_state = level_sampler.rollout_manager.batch_reset(_rng, init_train_levels.env_params)
        hstate = Actor.initialize_carry(init_state.time.shape)

        # --- Stack and return metrics ---
        zeros = jnp.zeros_like(level_buffer.score)
        carry = (rng, actor_state, critic_state, level_buffer, \
                init_train_levels, zeros, zeros, \
                hstate, hstate, init_obs, init_state)
        carry, metrics = jax.lax.scan(
            _ued_train_loop, carry, jnp.arange(args.train_steps), args.train_steps
        )
        return metrics, actor_state, critic_state, level_buffer

    return _train_fn


def run_training_experiment(args, eval_args):
    if args.log:
        init_logger(args)
    train_fn = make_train(args, eval_args)
    rng = jax.random.PRNGKey(args.seed)
    with Mesh(np.array(jax.devices()), ("devices", )):
        metrics, actor_state, critic_state, level_buffer = jax.jit(train_fn, in_shardings=None, out_shardings=None)(rng)
    if args.log:
        log_results(args, metrics, (actor_state, critic_state), level_buffer)
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