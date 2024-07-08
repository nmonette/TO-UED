import jax
import jax.numpy as jnp

from environments.gd_sampler import GDSampler
from environments.level_sampler import LevelSampler
from experiments.parse_args import parse_args
from experiments.logging import init_logger, log_results

from functools import partial


def make_train(args, eval_args):
    def _train_fn(rng):
        # --- Initialize LPG and level sampler ---
        rng, lpg_rng, dummy_rng, buffer_rng = jax.random.split(rng, 4)
        train_state = create_lpg_train_state(lpg_rng, args)

        level_sampler = GDSampler(eval_args)
        dummy_sampler = LevelSampler(args)
        level_buffer = dummy_sampler.initialize_buffer(dummy_rng)
        eval_buffer = level_sampler.initialize_buffer(buffer_rng)

        # --- Initialze agents and value critics ---
        require_value_critic = not args.use_es
        rng, train_rng = jax.random.split(rng)
        level_buffer, agent_states, value_critic_states = dummy_sampler.initial_sample(
            train_rng, level_buffer, args.num_agents, require_value_critic
        )

        # --- TRAIN LOOP ---
        lpg_train_step_fn = make_lpg_train_step(args, dummy_sampler)

        def _meta_train_loop(carry, t):
            rng, train_state, agent_states, value_critic_states, level_buffer, eval_buffer, x_grad, y_grad = carry

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

            def sample(*_):
                return partial(level_sampler.sample, dummy_sampler.agent_state_from_levels)(
                    rng, train_state, level_buffer, eval_buffer, x_grad, y_grad, agent_states, value_critic_states
                )
            def identity(*_):
                return level_buffer, eval_buffer, x_grad, y_grad, agent_states, value_critic_states
            
            level_buffer, eval_buffer, x_grad, y_grad, agent_states, value_critic_states = jax.lax.cond(
                t % args.regret_frequency == 0, sample, identity
            )

            carry = (rng, train_state, agent_states, value_critic_states, level_buffer, eval_buffer, x_grad, y_grad)
            return carry, metrics

        # --- Stack and return metrics ---
        zeros = jnp.zeros_like(level_buffer.score)
        carry = (rng, train_state, agent_states, value_critic_states, level_buffer, eval_buffer, zeros, zeros)
        carry, metrics = jax.lax.scan(
            _meta_train_loop, carry, jnp.arange(args.train_steps), args.train_steps
        )
        return metrics, train_state, level_buffer

    return _train_fn


def run_training_experiment(args, eval_args):
    if args.log:
        init_logger(args)
    train_fn = make_train(args, eval_args)
    rng = random.PRNGKey(args.seed)
    metrics, train_state, level_buffer = jax.jit(train_fn)(rng)
    if args.log:
        log_results(args, metrics, train_state, level_buffer)
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
    install()
    main()