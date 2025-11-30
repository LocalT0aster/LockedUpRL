import argparse
import sys

from rl import escape_game


def _build_parser():
    parser = argparse.ArgumentParser(description="LockedUpRL entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=False)

    train_p = subparsers.add_parser("train", help="Train DQN catchers")
    train_p.add_argument("--episodes", type=int, default=escape_game.EPISODES,
                         help="Number of training episodes")

    eval_p = subparsers.add_parser("eval", help="Evaluate the trained model")
    eval_p.add_argument("-n", "--num-episodes", type=int, default=300,
                        help="Number of evaluation episodes")
    eval_p.add_argument("--runner-type", choices=["astar", "human"], default="astar",
                        help="Runner controller during evaluation")

    play_p = subparsers.add_parser("play", help="Start the interactive pygame client")
    play_p.add_argument("--runner", choices=["astar", "human"], default=None,
                        help="Runner controller (overrides prompt if provided)")

    pipe_p = subparsers.add_parser("pipe", help="Pipe-based evaluation for Godot")
    pipe_p.add_argument("--role", choices=["runner", "catcher"], default="catcher",
                        help="Which agent role this process controls")
    pipe_p.add_argument("--catcher-index", type=int, default=0,
                        help="Index of the catcher to control (if multiple are present)")
    pipe_p.add_argument("--allow-unknown", action="store_true",
                        help="Allow stepping into unknown tiles in the provided vision map")

    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    cmd = args.command or "play"

    if cmd == "train":
        escape_game.train_dqn(episodes=args.episodes)
    elif cmd in ("eval", "evaluate"):
        escape_game.evaluate(num_episodes=args.num_episodes, runner_type=args.runner_type)
    elif cmd == "pipe":
        escape_game.run_pipe_agent(
            role=args.role,
            catcher_index=args.catcher_index,
            allow_unknown=args.allow_unknown,
        )
    else:
        # play
        if getattr(args, "runner", None) is not None:
            # Force runner type without prompting
            original_input = escape_game.input
            def _fixed_input(_prompt):
                return "2" if args.runner == "human" else "1"
            escape_game.input = _fixed_input
            try:
                escape_game.play_loop()
            finally:
                escape_game.input = original_input
        else:
            escape_game.play_loop()


if __name__ == "__main__":
    main(sys.argv[1:])
