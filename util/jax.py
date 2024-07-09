import jax


def jax_debug_wrapper(args, f):
    def wrapped_fn(*x):
        debug = args.debug
        debug_nans = args.debug_nans
        if debug_nans:
            jax.config.update("jax_debug_nans", True)
        if debug:
            with jax.disable_jit():
                return f(*x)
        else:
            return f(*x)

    return wrapped_fn


@jax.vmap
def gather(action_probabilities, action_index):
    return action_probabilities[action_index]


def mini_batch_vmap(f, num_mini_batches):
    """
    Execute a function in sequential, vmapped mini-batches.
    Enables execution of batches too large to fit in memory.
    """

    def mapped_fn(*args, **kwargs):
        def batched_fn(_, args):
            args, kwargs = args
            return None, jax.vmap(f)(*args, **kwargs)

        reshape_fn = lambda x: x.reshape((num_mini_batches, -1, *x.shape[1:]))
        mini_batched_args = jax.tree_util.tree_map(reshape_fn, args)
        mini_batched_kwargs = jax.tree_util.tree_map(reshape_fn, kwargs)
        _, ret = jax.lax.scan(batched_fn, None, (mini_batched_args, mini_batched_kwargs))
        return jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), ret)

    return mapped_fn