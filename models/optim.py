import optax
from evosax import OpenES


def create_optimizer(optimizer, learning_rate, max_grad_norm):
    if optimizer == "SGD":
        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.scale(learning_rate),
        )
    elif optimizer == "Adam":
        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate, eps=1e-5),
        )
    raise ValueError(f"Unknown optimizer: {optimizer}")


def create_es_strategy(args, params):
    return OpenES(
        popsize=args.num_agents * 2,
        maximize=True,  # Using return for fitness, so maximise with ES
        pholder_params=params,
        opt_name=args.lpg_opt.lower(),
        lrate_init=args.lpg_learning_rate,
        lrate_decay=args.es_lrate_decay,
        lrate_limit=args.es_lrate_limit,
        sigma_init=args.es_sigma_init,
        sigma_decay=args.es_sigma_decay,
        sigma_limit=args.es_sigma_limit,
        mean_decay=args.es_mean_decay,
    )
