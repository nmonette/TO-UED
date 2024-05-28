import jax
import sys

from jax import random
from rich.traceback import install

from util import *

from environments.nash_sampler import NashSampler
from experiments.parse_args import parse_args
from experiments.logging import init_logger, log_results
from meta.meta import make_lpg_train_step, create_lpg_train_state

def make_train(args):
    def _train_fn(rng):
        # --- Initialize nash distributions ---
        train_nash = jnp.zeros((args.buffer_size, )).at[0].set(1) 
        eval_nash = jnp.zeros((args.buffer_size, )).at[0].set(1) 

        # --- Initialize sampler and buffers ---
        level_sampler = NashSampler(args)
        rng, buffer_rng, train_rng = jax.random.split(rng, 3)
        train_buffer, eval_buffer = level_sampler.initialize_buffers(buffer_rng)
        train_state = create_lpg_train_state(train_rng, args)        
    
        # --- TRAIN LOOP ---
        lpg_train_step_fn = make_lpg_train_step(args, level_sampler)

        metrics_list = []
        
        # --- Double Oracle (environment generating) Loop ---
        for t in range(args.buffer_size):
            
            train_tree = jax.tree.map(lambda *xs: jnp.stack(xs), *train_buffer)
            eval_tree = jax.tree.map(lambda *xs: jnp.stack(xs), *train_buffer)
            
            if len(train_buffer) == 1:
                train_nash = jnp.ones((1, ))
                eval_nash = jnp.ones((1, ))
            else:
                train_nash, eval_nash, game = level_sampler.compute_nash(nash_rng, train_state, train_tree, eval_tree)
                print(game)

            # --- Initialilze training agents ---
            rng, _rng = jax.random.split(rng)
            agent_states, value_critic_states = level_sampler.get_training_levels(rng, train_tree, train_nash, create_value_critic=not args.use_es)
            
            # --- Update LPG ---
            rng, _rng = jax.random.split(rng)
            train_state, agent_states, value_critic_states, metrics = lpg_train_step_fn(
                rng=_rng,
                lpg_train_state=train_state,
                agent_states=agent_states,
                value_critic_states=value_critic_states,
            )

            """
            TODO: Figure out if we want to do this or not

            # --- Sample new levels and agents as required ---
            rng, _rng = jax.random.split(rng)
            agent_states, value_critic_states = level_sampler.sample(
                _rng, train_buffer, train_nash, agent_states, value_critic_states
            )
            """
            # --- Get best response levels ---
            rng, train_rng, eval_rng, nash_rng = jax.random.split(rng, 4)
            new_train = level_sampler.get_train_br(train_rng, train_state, eval_nash, eval_buffer)
            new_eval, eval_regret = level_sampler.get_eval_br(eval_rng, train_state)

            train_buffer.append(new_train.replace(active=True))
            new_eval.append(new_eval.replace(active=True))

            metrics["GT"] = {
                "eval_regret": eval_regret
            }

            metrics_list.append(metrics)
        
        metrics = jax.tree.map(lambda *xs: list(xs), *metrics_list)

        return metrics, train_state, train_buffer
    
    return _train_fn
    
def run_training_experiment(args):
    if args.log:
        init_logger(args)
    train_fn = make_train(args)
    rng = random.PRNGKey(args.seed)
    metrics, train_state, level_buffer = train_fn(rng) # jax.jit(train_fn)(rng)
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
    main() # "--num_agents 2 -br 10 --num_mini_batches 2 --buffer_size 3 --use_es --lifetime_conditioning".split(" ")
