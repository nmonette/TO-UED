import jax
import jax as jnp
# from jax.config import config as jax_config
jax_config = jax.config

def jax_debug_wrapper(args, f):
    def wrapped_fn(*x):
        debug = args.debug
        debug_nans = args.debug_nans
        if debug_nans:
            jax_config.update("jax_debug_nans", True)
        if debug:
            with jax.disable_jit():
                return f(*x)
        else:
            return f(*x)

    return wrapped_fn


@jax.vmap
def gather(action_probabilities, action_index):
    return action_probabilities[action_index]


def _scan_with_static(f, _, xs, in_axes):

    def fn(_, nonstatic):
        return f(_, jax.tree_util.tree_map(lambda ax,ful,ns: ful if ax is None else ns, in_axes, xs, nonstatic, is_leaf= lambda x: x is None)) 
        
    return jax.lax.scan(fn, None, jax.tree_util.tree_map(lambda ax, y: None if ax is None else y, in_axes, xs))

def mini_batch_vmap(f, num_mini_batches, in_axes = 0, size=None):
    """
    Execute a function in sequential, vmapped mini-batches.
    Enables execution of batches too large to fit in memory.
    """
    def reshape(kp, x, batch_size=num_mini_batches):
        if type(in_axes) is not int and in_axes[kp[0].idx] is None:
            return x
        else:
            return x.reshape((batch_size, -1, *x.shape[1:]))
        
    def mapped_fn(*args):
        def batched_fn(_, x):
            return None, jax.vmap(f, in_axes=in_axes)(*x)

        
        if size is not None and size <= num_mini_batches:
            return jax.vmap(f, in_axes=in_axes)(*args)

        elif size is not None and size % num_mini_batches != 0 and size // num_mini_batches > 1:
            mini_batched_args = jax.tree_util.tree_map(
               lambda x: jnp.array_split(x, num_mini_batches), args
            )

            first = jax.tree_util.tree_map(
                lambda x: x[0], mini_batched_args
            )   

            rest = jax.tree_util.tree_map_with_path(
                lambda kp, x: reshape(kp, jnp.stack(x[1:]), size // num_mini_batches), mini_batched_args
            ) 

            ret1 = jax.vmap(f, in_axes = in_axes)(*first)
            _, ret2 = _scan_with_static(batched_fn, None, rest, in_axes)
            ret2 = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), ret2)

            return jnp.stack((ret1, ret2))

        elif size is not None and size % num_mini_batches != 0 and size // num_mini_batches == 1:
            div = size // num_mini_batches
            first = jax.tree_util.tree_map(
                lambda x: x[:div], args
            )   

            rest = jax.tree_util.tree_map(
                lambda x: x[div:], args
            ) 

            ret1 = jax.vmap(f, in_axes = in_axes)(*first)
            ret2 = jax.vmap(f, in_axes = in_axes)(*first)

            return jnp.stack((ret1, ret2))

        else:
            mini_batched_args = jax.tree_util.tree_map_with_path(
                reshape, args
            )
            _, ret = _scan_with_static(batched_fn, None, mini_batched_args, in_axes)
            return jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), ret)
        
    return mapped_fn
