import jax
import jax.numpy as jnp

from flax.struct import dataclass

from util import *
from .gd_sampler import GDSampler

@dataclass
class MetaTrainState: # comments are array sizes
    x_vtable: jnp.ndarray # (args.regret_frequency + 1, )
    y_vtable: jnp.ndarray # (args.regret_frequency + 1, )
    regrets: jnp.ndarray  # (args.regret_frequency, )

    prev_x_grad: jnp.ndarray # (args.buffer_size, )
    prev_y_grad: jnp.ndarray # (args.buffer_size, )

    x_hat: jnp.ndarray # (args.buffer_size, )
    y_hat: jnp.ndarray # (args.buffer_size, )

    x: jnp.ndarray
    y: jnp.ndarray

    x_lp: jnp.ndarray # (args.regret_frequency, args.buffer_size, )
    y_lp: jnp.ndarray # (args.regret_frequency, args.buffer_size, )

    @staticmethod
    def from_args(args, x_hat, y_hat):
        
        return MetaTrainState(
            x_vtable=jnp.zeros((args.regret_frequency + 1, )),
            y_vtable=jnp.zeros((args.regret_frequency + 1, )),
            regrets=jnp.zeros((args.regret_frequency, )),
            prev_x_grad=jnp.zeros((args.buffer_size, )),
            prev_y_grad=jnp.zeros((args.buffer_size, )),
            x_hat=x_hat, 
            y_hat=y_hat,
            x=x_hat,
            y=y_hat,
            x_lp=jnp.zeros((args.regret_frequency, args.buffer_size)),
            y_lp=jnp.zeros((args.regret_frequency, args.buffer_size))
        )

def make_meta_step(args):

    def meta_step(rng, meta_state, train_buffer, eval_buffer):
        
        # --- Updating V-tables ---
        lr = args.meta_value_lr
        gamma = args.meta_gamma

        def td_step(value, next_value, reward):
            return (1 - lr) * value + lr * gamma * reward + next_value
        
        x_vtable = meta_state.x_vtable.at[:-1].set(jax.vmap(td_step)(
            meta_state.x_vtable[:-1],
            meta_state.x_vtable[1:],
            -meta_state.regrets
        ))
        y_vtable = meta_state.y_vtable.at[:-1].set(jax.vmap(td_step)(
            meta_state.y_vtable[:-1],
            meta_state.y_vtable[1:],
            meta_state.regrets
        ))

        # --- Calculate gradients ---
        # NOTE: if we are doing a 1-state table, take the mean here over the 0-axis
        x_gae, _ = gae(
            x_vtable, 
            -meta_state.regrets, 
            jnp.full_like(meta_state.regrets, False).at[-1].set(True),
            gamma,
            args.gae_lambda
        )
        y_gae, _ = gae(
            y_vtable, 
            meta_state.regrets, 
            jnp.full_like(meta_state.regrets, False).at[-1].set(True),
            gamma,
            args.gae_lambda
        )
        
        x_grad = (x_gae * meta_state.x_lp.T).mean(axis=1)
        y_grad = (y_gae * meta_state.y_lp.T).mean(axis=1)

        # --- Update meta-policies ---
        lr = args.ogd_learning_rate
        trunc = args.ogd_trunc_size

        x_hat = projection_simplex_truncated(meta_state.x_hat + lr * x_grad, trunc)
        y_hat = projection_simplex_truncated(meta_state.y_hat + lr * y_grad, trunc)

        # --- Replacing lowest scoring levels ---
        # NOTE: we would do this in `level_sampler._sample_step` if multi-state policy
        rng, train_rng, eval_rng = jax.random.split(rng, 3)
        level_sampler = GDSampler(args)
        new_train = level_sampler._reset_lowest_scoring(train_rng, train_buffer.replace(score=meta_state.x), args.num_agents)
        new_eval = level_sampler._reset_lowest_scoring(eval_rng, eval_buffer.replace(score=meta_state.y), args.num_agents)

        # --- Make sure to mark levels as not new ---
        new_train = new_train.replace(
            new = jnp.where(x_grad != 0, False, new_train.new)
        )

        meta_state = meta_state.replace(
            x_vtable=x_vtable,
            y_vtable=y_vtable,
            regrets=jnp.zeros_like(meta_state.regrets),

            prev_x_grad=x_grad,
            prev_y_grad=y_grad,

            x_hat=x_hat,
            y_hat=y_hat,

            x_lp = jnp.zeros_like(meta_state.x_lp),
            y_lp = jnp.zeros_like(meta_state.y_lp)
        )

        return meta_state, new_train, new_eval

    return meta_step

