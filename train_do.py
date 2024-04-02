import jax
import sys

from jax import random
from rich.traceback import install

from util import *
from environments.level_sampler import LevelSampler
from environments.environments import get_env, reset_env_params, get_env_spec
from environments.rollout import RolloutWrapper
from agents.agents import AgentHyperparams, AgentState, create_value_critic, create_agent
from agents.a2c import A2CHyperparams
from experiments.parse_args import parse_args
from experiments.logging import init_logger, log_results
from meta.meta import create_lpg_train_state, make_lpg_train_step


def make_train(args):
    def _train_fn(rng):
        # --- Initialize LPG and level sampler ---
        rng, lpg_rng, buffer_rng = jax.random.split(rng, 3)
        train_state = create_lpg_train_state(lpg_rng, args)
        level_sampler = LevelSampler(args)
        level_buffer = level_sampler.initialize_buffer(buffer_rng)

        # --- Initialze agents and value critics ---
        require_value_critic = not args.use_es
        rng, _rng = jax.random.split(rng)
        level_buffer, agent_states, value_critic_states = level_sampler.initial_sample(
            _rng, level_buffer, args.num_agents, require_value_critic
        )

        # --- TRAIN LOOP ---
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

            # --- Sample new levels and agents as required ---
            rng, _rng = jax.random.split(rng)
            level_buffer, agent_states, value_critic_states = level_sampler.sample(
                _rng, level_buffer, agent_states, value_critic_states
            )
            carry = (rng, train_state, agent_states, value_critic_states, level_buffer)
            return carry, metrics

        # --- Stack and return metrics ---
        carry = (rng, train_state, agent_states, value_critic_states, level_buffer)
        carry, metrics = jax.lax.scan(
            _meta_train_loop, carry, None, length=args.train_steps
        )
        return metrics, train_state, level_buffer

    return _train_fn

def make_train_new(args): 
    # need to add num_iters to arg parser to make payoff matrix compatible with JIT
    def _train_fn(rng):
        """
        Sample a random environment for the evaluator and the meta-policy, 
        then add those to each buffer, then initialize the nash strategy
        of each as [1]. Then, scan over _meta_train_loop.  
        """
        # --- Initialize environment ---
        env_kwargs, max_rollout_len, max_lifetime = get_env_spec(
            args.env_name, args.env_mode
        )
        env = get_env(args.env_name, env_kwargs)

        # --- Get agent hyperparameters ---
        agent_hypers = AgentHyperparams.from_args(args)
        a2c_hypers = A2CHyperparams(
            args.gamma, args.gae_lambda, args.entropy_coeff
        )
        require_value_critic = not args.use_es
        if require_value_critic:
            agent_hypers = agent_hypers.replace(critic_dims=1)            

        def _meta_train_loop(rng):
            """
            Call `_get_meta_policy_br` and `_get_adv_br` to generate new
            environments for each buffer. Collect LPG regrets (note here that 
            we do not have to use ES because there is only one environment), 
            and then fill out the table by vmapping the get regret function.
            Solve for the new nash of the meta-game, and the iteration is done. 
            """
            def _get_meta_policy_br(rng):
                """
                Scan across the _meta_loop for n iterations, aiming to
                minimize regret. Return the best environment. 
                """
                def _meta_loop(carry, _):
                    """
                    Generate or sample an environment, train an LPG on 
                    it, then collect the convex combination of 
                    algorithmic regrets over the nash of evaluative 
                    environments. If the regret
                    is lower than the current lowest, replace the
                    'current min' with the newly generated/sampled
                    environment. If choosing to use a generator, 
                    use the -regret as a reward for the generator.
                    """
                    rng, min_env, min_regret = carry 
                    # --- Sample a level ---
                    rng, env_rng = jax.random.split(rng)
                    env_params, lifetime = reset_env_params(rng, args.env_name, args.env_mode)
                    level = Level(env_params, lifetime, 0)

                    # --- Initialize rollout manager ---
                    rollout_manager = RolloutWrapper(
                        args.env_name, args.train_rollout_len, max_rollout_len, env_kwargs
                    )

                    # --- Initialize LPG/LPG Loop ---
                    rng, lpg_rng = jax.random.split(rng)
                    train_state = create_lpg_train_state(lpg_rng, args)
                    lpg_train_step_fn = make_lpg_train_step(args, rollout_manager, True)

                    def init_lpg_train(rng):
                        # Code here mocks LevelSampler.initial_sample:
                        # --- Initialize agent ---
                        rng, worker_rng, agent_rng = jax.random.split(rng, 3)
                        env_obs, env_state = rollout_manager.batch_reset(
                            worker_rng, level.env_params, args.env_workers
                        )
                        actor_state, critic_state = create_agent(
                            agent_rng, agent_hypers, env.num_actions, env.observation_space
                        )

                        # --- Initialize value critic (if called for) ---
                        value_critic_states = None
                        if require_value_critic:
                            rng, critic_rng = jax.random.split(rng)
                            value_critic_states = create_value_critic(
                                critic_rng, agent_hypers, env.observation_space
                            )

                        return AgentState(
                            actor_state=actor_state,
                            critic_state=critic_state,
                            level=level,
                            env_obs=env_obs,
                            env_state=env_state
                        ), value_critic_states

                    
                    def _lpg_loop(carry, _):
                        rng, train_state, agent_states, value_critic_states = carry

                        # --- Update LPG ---
                        rng, _rng = jax.random.split(rng)
                        train_state, agent_states, value_critic_states, metrics = lpg_train_step_fn(
                            rng=_rng,
                            lpg_train_state=train_state,
                            agent_states=agent_states,
                            value_critic_states=value_critic_states,
                        )

                        return carry, metrics

                    # --- Run LPG loop and return metrics ---
                    agent_state, value_critic_states = init_lpg_train(rng)
                    carry = (rng, train_state, agent_state, value_critic_states)
                    carry, metrics = jax.lax.scan(
                        _lpg_loop, carry, None, length=args.train_steps
                    )

                    # --- Collect Algorithmic Regret --- 

                    rng, a2c_rng = jax.random.split(rng)
                    a2c_agent_state = create_agent()

                    
                    return rng, min_env, min_regret

            def _get_adv_br(rng):
                """
                Scan across the _meta_loop for n iterations, aiming
                to maximize regret. Return the best environment. 
                """
                def _meta_loop(carry, _):
                    """
                    Generate or sample an environment, train an LPG
                    over the nash of the training environments, and 
                    then collect the evaluative regret on the 
                    generated/sampled environment. If the regret
                    is higher than the current highest, replace the
                    'current max' with the newly generated/sampled
                    environment. If choosing to use a generator, 
                    use the regret as a reward for the generator. 
                    """
                    pass
                pass
            pass

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
