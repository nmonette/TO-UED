import sys
import argparse


def parse_args(cmd_args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Debug mode (disable JIT)")
    parser.add_argument(
        "--debug_nans",
        action="store_true",
        help="Exit and stack trace when NaNs are encountered",
    )

    # --- ENVIRONMENT ---
    parser.add_argument(
        "--env_name", help="Environment name", type=str, default="GridWorld-v0"
    )
    parser.add_argument(
        "--env_mode", help="Environment mode", type=str, default="all_shortlife"
    )
    parser.add_argument(
        "--eval_env_mode", help="Environment mode for eval levels", default="all_shortlife"
    )
    parser.add_argument(
        "--env_workers",
        help="Number of environment workers per agent",
        type=int,
        default=64,
    )

    # --- EXPERIMENT ---
    # Settings
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--train_steps", help="Number of train steps", type=int, default=int(3e4)
    )
    parser.add_argument(
        "--num_agents",
        help="Meta-train batch size, doubled for antithetic task sampling when using ES",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--num_mini_batches",
        help="Number of meta-training mini-batches",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--regret-frequency",
        help="Number of meta-training iterations before another regret update",
        type=int,
        default=1,
    )

    # Double Oracle
    parser.add_argument(
        "-br",
        "--best-response-length",
        help="Number of best response iterates, must be multiple of 20",
        dest="br", 
        default=10, 
        type=int
    )
    # Logging
    parser.add_argument("--log", action="store_true", help="Log with WandB")
    parser.add_argument("--wandb_project", type=str, help="Wandb project")
    parser.add_argument("--wandb_entity", type=str, help="Wandb entity")
    parser.add_argument("--wandb_group", type=str, default="debug", help="WandB group")

    # --- AGENT ---
    parser.add_argument(
        "--train_rollout_len",
        help="Number of environment steps per agent update",
        type=int,
        default=20,
    )  # Reference: 20
    parser.add_argument("--gamma", help="Discount factor", type=float, default=0.99)
    parser.add_argument(
        "--gae_lambda",
        help="Lambda parameter for Generalized Advantage Estimation",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--entropy_coeff",
        help="Actor entropy coefficient for A2C agents",
        type=float,
        default=0.01,
    )

    # --- LPG ---
    parser.add_argument(
        "--lpg_embedding_net_width",
        help="Width of LPG embedding network",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lpg_gru_width", help="Number of LPG LSTM units", type=int, default=256
    )
    parser.add_argument(
        "--lpg_target_width",
        help="Size of categorical prediction vector (target) generated by LPG",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lpg_agent_target_coeff",
        help="(alpha_y) Agent target KL divergence.",
        type=float,
        default=5e-1,
    )

    # Meta-optimization
    parser.add_argument("--lpg_opt", help="LPG optimizer", type=str, default="Adam")
    parser.add_argument(
        "--lpg_learning_rate", help="LPG learning rate", type=float, default=1e-4
    )

    # Meta-gradients
    parser.add_argument(
        "--num_agent_updates",
        help="(K) Number of agent updates per LPG train step",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--lpg_max_grad_norm",
        help="Max gradient norm for LPG optimisation",
        type=float,
        default=0.5,
    )

    # Meta-gradient regularization coefficients
    parser.add_argument(
        "--lpg_policy_entropy_coeff",
        help="(beta_0) Trained agent policy entropy.",
        type=float,
        default=5e-2,
    )
    parser.add_argument(
        "--lpg_target_entropy_coeff",
        help="(beta_1) Trained agent target entropy.",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--lpg_policy_l2_coeff",
        help="(beta_2) Policy update (pi_hat) L2 regularization.",
        type=float,
        default=5e-3,
    )
    parser.add_argument(
        "--lpg_target_l2_coeff",
        help="(beta_3) Target update (y_hat) L2 regularization.",
        type=float,
        default=1e-3,
    )

    # ES
    parser.add_argument("--use_es", action="store_true", help="Optimize LPG with ES")
    parser.add_argument("--es_lrate_decay", type=float, default=0.999)
    parser.add_argument("--es_lrate_limit", type=float, default=1e-5)
    parser.add_argument("--es_sigma_init", type=float, default=0.1)
    parser.add_argument("--es_sigma_decay", type=float, default=1.0)
    parser.add_argument("--es_sigma_limit", type=float, default=0.1)
    parser.add_argument("--es_mean_decay", type=float, default=0.0)

    # TA-LPG
    parser.add_argument(
        "--lifetime_conditioning",
        help="Condition LPG on agent lifetime",
        action="store_true",
    )

    # --- UED ---
    parser.add_argument(
        "--buffer_size", help="Size of level buffer", type=int, default=4000
    )
    parser.add_argument(
        "--score_function",
        help="UED level scoring function",
        type=str,
        default="random",
    )
    parser.add_argument(
        "--p_replay",
        help="Probability of replaying a level from the buffer (vs. random sampling)",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--score_transform",
        help="Transform to apply to level score",
        type=str,
        default="rank",
    )
    parser.add_argument(
        "--score_temperature",
        help="Temperature of score transformation function",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--regret_method", help="Method for computing regret", type=str, default="loop", choices=["loop", "mini_batch_vmap", "heuristic", "test"]
    )
    parser.add_argument(
        "--num_regret_updates", help="Number of regret updates to VMAP when using the regret vmap heuristic", type=int, default=32
    )

    args, rest_args = parser.parse_known_args(cmd_args)
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")
    if args.num_agents % args.num_mini_batches != 0:
        raise ValueError(
            f"Number of agents ({args.num_agents}) must be divisible by number of mini-batches ({args.num_mini_batches})"
        )
    return args
