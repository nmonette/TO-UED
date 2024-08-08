import jax
import jax.numpy as jnp

from flax.struct import dataclass

from util import *
from .gd_sampler import GDSampler

ETA = 0.001
BETA = 0.01
EPS = 0.1
KAPPA = 10

@dataclass
class UncoupledMetaTrainState:
    tau: jnp.ndarray # (args.regret_frequency, )

    regrets: jnp.ndarray  # (args.regret_frequency, )
    x_vtable: jnp.ndarray # (args.regret_frequency + 1, )
    y_vtable: jnp.ndarray # (args.regret_frequency + 1, )
    x_vsim: jnp.ndarray # (args.regret_frequency + 1, )
    y_vsim: jnp.ndarray # (args.regret_frequency + 1, )

    x: jnp.ndarray # (args.regret_frequency, args.buffer_size)
    y: jnp.ndarray # (args.regret_frequency, args.buffer_size)

    @staticmethod
    def from_args(args, x, y):

        return UncoupledMetaTrainState(
            tau = jnp.zeros(args.regret_frequency),
            regrets=jnp.zeros(args.regret_frequency),
            x_vtable=jnp.zeros(args.regret_frequency + 1),
            y_vtable=jnp.zeros(args.regret_frequency + 1),
            x_vsim=jnp.zeros(args.regret_frequency + 1),   
            y_vsim=jnp.zeros(args.regret_frequency + 1),
            x=x,
            y=y
        )


def make_meta_step(args):

    def meta_step(meta_state, done_counts, eval_counts, t):

        tau = meta_state.tau.at[t].set(meta_state.tau.at[t] + 1)
        H = jnp.log(args.regret_frequency) / (1 - args.meta_gamma)
        alpha = (H + 1)/(H + tau[t])
        bns = KAPPA * args.buffer_size * jnp.log(jnp.log(args.regret_frequency * args.buffer_size * args.regret_frequency)) * (BETA + (1/ETA) * alpha) / jnp.square(1 - args.meta_gamma)

        prob_x = jax.scipy.stats.multinomial.pmf(done_counts, args.num_agents, meta_state.x[t])
        prob_y = jax.scipy.stats.multinomial.pmf(eval_counts, args.num_agents, meta_state.y[t])
        g_x = jnp.where(done_counts > 0, 1., 0.) * (meta_state.regrets[t] + args.gamma * meta_state.x_vtable[t + 1]) / (prob_x + beta)
        g_y = jnp.where(eval_counts > 0, 1., 0.) * (-meta_state.regrets[t] + args.gamma * meta_state.y_vtable[t + 1]) / (prob_y + beta)

        grad_fn = jax.grad(lambda x, g, xst: x.T @ g + (1/ETA) * jax.scipy.special.kl_div(x, xst).sum())
        # carry = (x, g, xst)
        argmin = lambda carry, _: (projection_simplex_truncated(carry[0] - grad_fn(carry[0], carry[1], carry[2]) * lr, 1 / (args.buffer_size * args.regret_frequency)), None)

        x, _ = jax.lax.scan(argmin, (meta_state.x[t], g_x, meta_state.x[t]), length=1000)
        y, _ = jax.lax.scan(argmin, (meta_state.y[t], g_y, meta_state.y[t]), length=1000)

        lr = args.meta_value_lr
        gamma = args.meta_gamma
        x_v = (1 - lr) * meta_state.x_vsim[t] + lr * (meta_state.regrets[t] + gamma * meta_state.x_vtable[t + 1] - bns)
        y_v = (1 - lr) * meta_state.y_vsim[t] + lr * (-meta_state.regrets[t] + gamma * meta_state.y_vtable[t + 1] - bns)
        
        return UncoupledMetaTrainState(
            tau=tau,
            regrets=meta_state.regrets,
            x_vtable=meta_state.x_vtable.at[t].set(jax.lax.select(x_v > 0., x_v, 0.)),
            #NOTE: need to figure out how this max part works with the y, because it has negative utilities
            y_vtable=meta_state.y_vtable.at[t].set(jax.lax.select(y_v > 0., y_v, 0.)),
            x_vsim=meta_state.x_vsim.at[t].set(x_v),
            y_vsim=meta_state.y_vsim.at[t].set(y_v),
            x=x,
            y=y
        )
    
    return meta_step


