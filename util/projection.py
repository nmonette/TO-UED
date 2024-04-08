import jax
import jax.numpy as jnp
from functools import partial

"""
Code modified from jaxopt:
https://jaxopt.github.io/stable/_modules/jaxopt/_src/projection.html#projection_sparse_simplex
"""

def projection_simplex(
    x: jnp.ndarray, max_nz: int) -> jnp.ndarray:
    """
    Projection onto the unit simplex where assuming that the first max_nz elements are nonzero and the rest are zero,
    it will maintain that property.
    """
    # max_nz = 2

    max_nz_indices = jnp.nonzero(x, size=x.shape[0], fill_value=0)
    max_nz_values = jnp.where(jnp.arange(0, x.shape[0]) <= max_nz, x, 0)

    # Projection the sorted top k values onto the unit simplex
    cumsum_max_nz_values = jnp.cumsum(max_nz_values)
    cumsum_max_nz_values = jnp.where(jnp.arange(0, x.shape[0]) <= max_nz, cumsum_max_nz_values, 0)
    ind = jnp.arange(x.shape[0])
    cond = (1 / ind + (max_nz_values - cumsum_max_nz_values / ind))
    cond = jnp.where(jnp.arange(0, x.shape[0]) <= max_nz, cond, 0)
    idx = jnp.count_nonzero(cond)
    to_relu = 1 / idx + (max_nz_values - cumsum_max_nz_values[idx - 1] / idx)
    to_relu = jnp.where(jnp.arange(0, x.shape[0]) <= max_nz, to_relu, 0)
    max_nz_simplex_projection = jax.nn.relu(
        to_relu)
    
    # Put the projection of max_nz_values to their original indices;
    # set all other indices zero.
    sparse_simplex_projection = jnp.sum(
        (max_nz_simplex_projection[ :, jnp.newaxis] * jax.nn.one_hot(
            max_nz_indices, len(x), dtype=x.dtype)).reshape(x.shape[0], x.shape[0]), axis=0)
    
    return sparse_simplex_projection