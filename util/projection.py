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
    idx = jnp.argsort(jnp.where(jnp.arange(0, x.shape[0]) < max_nz, x, -jnp.inf), descending=True)

    max_nz_indices = jnp.where(idx > -jnp.inf, idx, -1)
    max_nz_values = jnp.where(idx > -1, x[idx], 0)
    # Projection the sorted top k values onto the unit simplex
    cumsum_max_nz_values = jnp.cumsum(max_nz_values)
    cumsum_max_nz_values = jnp.where(jnp.arange(0, x.shape[0]) < max_nz, cumsum_max_nz_values, 0)
    ind = jnp.arange(x.shape[0]) + 1
    cond = jnp.nan_to_num(1 / ind + (max_nz_values - cumsum_max_nz_values / ind)) > 0
    cond = jnp.where(jnp.arange(0, x.shape[0]) < max_nz, cond, False) 
    idx = jnp.count_nonzero(cond)
    to_relu = 1 / idx + (max_nz_values - cumsum_max_nz_values[idx - 1] / idx)
    to_relu = jnp.where(jnp.arange(0, x.shape[0]) < max_nz, to_relu, 0)
    max_nz_simplex_projection = jax.nn.relu(to_relu)

    # Put the projection of max_nz_values to their original indices;
    # set all other indices zero.
    
    sparse_simplex_projection = jnp.sum(
      max_nz_simplex_projection[ :, jnp.newaxis] * jax.nn.one_hot(
          max_nz_indices, len(x), dtype=x.dtype), axis=0)
    
    return sparse_simplex_projection

def projection_simplex_truncated(x: jnp.ndarray, eps: float) -> jnp.ndarray: 
    """
    Code adapted from 
    https://www.ryanhmckenna.com/2019/10/projecting-onto-probability-simplex.html
    To represent truncated simplex projection. Assumes 1D vector. 
    """
    ones = jnp.ones_like(x)
    lambdas = jnp.concatenate((ones * eps - x, ones - x), axis=-1)
    idx = jnp.argsort(lambdas)
    lambdas = jnp.take_along_axis(lambdas, idx, -1)
    active = jnp.cumsum((jnp.float32(idx < x.shape[-1])) * 2 - 1, axis=-1)[..., :-1]
    diffs = jnp.diff(lambdas, n=1, axis=-1)
    left = (ones * eps).sum(axis=-1)
    left = left.reshape(*left.shape, 1)
    totals = left + jnp.cumsum(active*diffs, axis=-1)

    def generate_vmap(counter, func):
        if counter == 0:
            return func
        else:
            return generate_vmap(counter - 1, jax.vmap(func))
                
    i = jnp.expand_dims(generate_vmap(len(totals.shape) - 1, partial(jnp.searchsorted, v=1))(totals), -1)
    lam = (1 - jnp.take_along_axis(totals, i, -1)) / jnp.take_along_axis(active, i, -1) + jnp.take_along_axis(lambdas, i+1, -1)
    return jnp.clip(x + lam, eps, 1)
