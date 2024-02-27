import jax
import sys

from jax import random
from rich.traceback import install

from util import *
from environments.level_sampler import LevelSampler
from experiments.parse_args import parse_args
from experiments.logging import init_logger, log_results
from meta.meta import create_lpg_train_state, make_lpg_train_step


def make_train(args):
    def _train_fn(rng):
        # --- Initialize LPG and level sampler ---
        rng, lpg_rng, train_buffer_rng, eval_buffer_rng = jax.random.split(rng, 4)
        train_state = create_lpg_train_state(lpg_rng, args)

        train_sampler = LevelSampler(args)
        train_buffer = train_sampler.initialize_buffer(train_buffer_rng)

        eval_sampler = LevelSampler(args)
        eval_buffer = eval_sampler.initialize_buffer(eval_buffer_rng)

        # --- Initialze agents and value critics ---
        require_value_critic = not args.use_es
        rng, train_rng, eval_rng = jax.random.split(rng, 3)
        
        train_buffer, train_agent_states, train_value_critic_states = train_sampler.initial_sample(
            train_rng, train_buffer, args.num_agents, require_value_critic
        )
        eval_buffer, eval_agent_states, eval_value_critic_states = eval_sampler.initial_sample(
            eval_rng, eval_buffer, args.num_agents, require_value_critic
        )

        # --- TRAIN LOOP ---
        lpg_train_step_fn = make_lpg_train_step(args, train_sampler.rollout_manager)

        def _meta_train_loop(carry, _):
            rng, train_state, \
                train_agent_states, eval_agent_states, \
                train_value_critic_states, eval_value_critic_states, \
                train_buffer, eval_buffer = carry

            # --- Update LPG ---
            rng, train_rng, eval_rng = jax.random.split(rng, 3)
            train_state, train_agent_states, train_value_critic_states, metrics = lpg_train_step_fn(
                rng=train_rng,
                lpg_train_state=train_state,
                agent_states=train_agent_states,
                value_critic_states=train_value_critic_states,
            )

            train_state, eval_agent_states, eval_value_critic_states, eval_metrics = lpg_train_step_fn(
                rng=eval_rng,
                lpg_train_state=train_state,
                agent_states=eval_agent_states,
                value_critic_states=eval_value_critic_states,
            ) # we can deal with eval metrics later...

            # --- Sample new levels and agents as required ---
            rng, train_rng, eval_rng = jax.random.split(rng, 3)
            train_buffer, train_agent_states, train_value_critic_states = train_sampler.sample(
                train_rng, train_buffer, train_agent_states, train_value_critic_states
            )
            eval_buffer, eval_agent_states, eval_value_critic_states = eval_sampler.sample(
                eval_rng, eval_buffer, eval_agent_states, eval_value_critic_states
            )
            
            carry = (rng, train_state,
                                    train_agent_states, eval_agent_states, 
                                    train_value_critic_states, eval_value_critic_states, 
                                    train_buffer, eval_buffer)
            return carry, metrics

        # --- Stack and return metrics ---
        carry = (rng, train_state,
                                train_agent_states, eval_agent_states, 
                                train_value_critic_states, eval_value_critic_states, 
                                train_buffer, eval_buffer)
        carry, metrics = jax.lax.scan(
            _meta_train_loop, carry, None, length=args.train_steps
        )
        return metrics, train_state, train_buffer

    return _train_fn


def run_training_experiment(args):
    if args.log:
        init_logger(args)
    train_fn = make_train(args)
    rng = random.PRNGKey(args.seed)
    metrics, train_state, level_buffer = jax.jit(train_fn)(rng)
    if args.log:
        log_results(args, metrics, train_state, level_buffer)
    else:
        print(metrics)


def main(cmd_args=sys.argv[1:]):
    args = parse_args(cmd_args)
    experiment_fn = jax_debug_wrapper(args, run_training_experiment)
    return experiment_fn(args)


if __name__ == "__main__":
    install()
    main()
