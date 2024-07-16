import jax
import jax.numpy as jnp
from rich.traceback import install

from ued.level_sampler import LevelSampler
from ued.train_single_env import train_agent
from ued.rnn import eval_agent, Actor
from experiments.parse_args import parse_args
from experiments.logging import init_logger, log_results
from util.jax import jax_debug_wrapper
from ued.rnn import eval_agent

from functools import partial
import sys

def make_train(args, eval_args):
    def _train_fn(rng):
        # --- Initialize policy and level buffers and samplers ---
        rng, agent_rng, dummy_rng = jax.random.split(rng, 3)
        dummy_sampler = LevelSampler(args)
        
        level_buffer = dummy_sampler.initialize_buffer(dummy_rng)
        level_buffer = level_buffer.replace(
            new=level_buffer.new.at[0].set(True)
        )

        init_agent = dummy_sampler._create_agent(
            agent_rng, jax.tree_util.tree_map(lambda x: x[0], level_buffer.level), True
        ) 

        actor_state = init_agent.actor_state
        critic_state = init_agent.critic_state

        train_agent_fn = partial(
            train_agent, 
            rollout_manager=dummy_sampler.rollout_manager,
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
            rng, actor_state, critic_state, level, actor_hstate, critic_hstate, init_obs, init_state = carry
            
            # --- Train agents on sampled levels ---
            rng, _rng = jax.random.split(rng)
            (actor_state, critic_state, actor_hstate, critic_hstate, init_obs, init_state), metrics = train_agent_fn(
                rng=_rng,
                actor_state=actor_state,
                critic_state=critic_state,
                env_params=level.env_params, 
                actor_hstate=actor_hstate,
                critic_hstate=critic_hstate,
                init_obs=init_obs, 
                init_state=init_state, 
            )

            rng, _rng = jax.random.split(rng)
            eval_hstates = Actor.initialize_carry((args.env_workers, ))
            metrics["agent_return"] = eval_agent(
                _rng, 
                dummy_sampler.rollout_manager, 
                level.env_params,
                actor_state,
                args.env_workers, 
                eval_hstates
            )
            
            carry = (rng, actor_state, critic_state, level, actor_hstate, critic_hstate, init_obs, init_state)
            
            return carry, metrics

        level = jax.tree_util.tree_map(lambda x: x[0], level_buffer.level)

        # NOTE: batch_reset has been modified to accept a batch of env_params
        rng, _rng = jax.random.split(rng)
        init_obs, init_state = dummy_sampler.rollout_manager.batch_reset_single_env(_rng, level.env_params, args.env_workers)
        hstate = Actor.initialize_carry(init_state.time.shape)

        # --- Stack and return metrics ---
        carry = (rng, actor_state, critic_state, level, hstate, hstate, init_obs, init_state)
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
    metrics, actor_state, critic_state, level_buffer = jax.jit(train_fn)(rng)
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