import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState


def batch_rollout_entropy(train_state: TrainState, x: jnp.ndarray):
    """Computes the entropy of the policy/target over a batch of rollouts."""
    probs = train_state.apply_fn({"params": train_state.params}, x)
    probs += 1e-8
    return -jnp.mean(jnp.multiply(probs, jnp.log(probs)).sum(axis=-1)), probs


def kl_divergence(p: jnp.array, q: jnp.array, eps: float = 1e-8):
    """Calculates the Kullback-Leibler divergence between arrays p and q."""
    return p.dot(jnp.log(p + eps) - jnp.log(q + eps))


def gae(
    value: jnp.array,
    reward: jnp.array,
    done: jnp.array,
    discount: float,
    gae_lambda: float,
):
    """
    Modified from Gymnax-blines
    Value has length T+1, reward and done have length T
    Returns advantages and value targets
    """

    def loop(gae, t):
        value_diff = discount * value[t + 1] * (1 - done[t]) - value[t]
        delta = reward[t] + value_diff
        gae = delta + discount * gae_lambda * (1 - done[t]) * gae
        return gae, gae
    
    _, advantages = jax.lax.scan(loop, 0.0, jnp.arange(len(done)), reverse=True)

    return advantages, advantages + value[:-1]
