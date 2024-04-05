import jax
import sys
from functools import partial

from jax import random
from rich.traceback import install

from util import *
from environments.level_sampler import LevelSampler, LevelBuffer
from environments.nash_sampler import NashSampler
from environments.environments import get_env, reset_env_params, get_env_spec
from environments.rollout import RolloutWrapper
from agents.agents import AgentHyperparams, AgentState, create_value_critic, create_agent
from agents.a2c import A2CHyperparams
from experiments.parse_args import parse_args
from experiments.logging import init_logger, log_results
from meta.meta import create_lpg_train_state, make_lpg_train_step
from meta.train import train_lpg_agent

def make_train(args): 
    def _train_fn(rng):
        """
        Sample a random environment for the evaluator and the meta-policy, 
        then add those to each buffer, then initialize the nash strategy
        of each as [1]. Then, scan over _meta_train_loop.  
        """      
        # --- Initialize nash distributions ---
        train_nash = jnp.ones((1, ))# jnp.zeros((args.doi, )).at[0].set(1)
        eval_nash = jnp.ones((1, )) # jnp.zeros((args.doi, )).at[0].set(1)

        # Initialize sampler and buffers, 
        level_sampler = NashSampler(args)
        train_buffer, eval_buffer = level_sampler.initialize_buffers(rng)

        # --- Flag initial environments as active ---
        train_buffer = train_buffer.levels.at[0].replace(active=True)
        eval_buffer = eval_buffer.levels.at[0].replace(active=True)

        # --- Double Oracle (environment generating) Loop ---
        def _meta_do_loop(carry, t):
            rng, train_buffer, eval_buffer, train_nash, eval_nash = carry

            new_train = level_sampler.get_train_br(rng, eval_nash, eval_buffer).replace(active=True)
            new_eval = level_sampler.get_eval_br(rng, train_nash, train_buffer).replace(active=True)

            train_buffer = train_buffer.level.at[t].set(new_train)
            eval_buffer = eval_buffer.level.at[t].set(new_train)

            train_nash, eval_nash = level_sampler.compute_nash(rng, train_buffer, eval_buffer)

            return (rng, train_buffer, eval_buffer, train_nash), None
        
        carry = (rng, train_buffer, eval_buffer, train_nash, eval_nash)
        carry , _ = jax.lax.scan(_meta_do_loop, carry, jnp.arange(0, args.doi), args.doi)

        rng, train_buffer, eval_buffer, train_nash, eval_nash = carry

        train_buffer = train_buffer.level.replace(
            active = jnp.full_like(train_buffer.levels, False)
        )

        # --- Initialize agent states ---
        rng, _rng = jax.random.split(rng)
        agent_states = level_sampler.get_training_levels(_rng, train_buffer, train_nash)
        ## we won't create a value_critic here because we will use ES.

        lpg_train_step_fn = make_lpg_train_step(args, level_sampler.rollout_manager)

        def _meta_train_loop(carry, _):
            rng, train_state, agent_states, value_critic_states, level_buffer = carry

            # --- Update LPG ---
            rng, _rng = jax.random.split(rng)
            train_state, agent_states, value_critic_states, metrics = lpg_train_step_fn(
                rng=_rng,
                lpg_train_state=train_state,
                agent_states=agent_states,
                value_critic_states=value_critic_states,
            )
            carry = (rng, train_state, agent_states, value_critic_states, level_buffer)
            return carry, metrics

        carry = (rng, train_state, agent_states, None, train_buffer)
        carry, metrics = jax.lax.scan(
            _meta_train_loop, carry, None, length=args.train_steps
        )
        rng, train_state, agent_states, _, level_buffer = carry
            
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
