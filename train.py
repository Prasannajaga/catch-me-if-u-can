import argparse
import json
import shutil
from pathlib import Path

from stable_baselines3 import PPO
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from envs.catch_env import CatchMeEnv


ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = ROOT / "runs" / "ppo_catchme"
DEFAULT_MODEL_PATH = ROOT / "models" / "catchme_ppo.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Catch Me If You Can RL agent")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Legacy run name support. If set and --log-dir is not set, logs are saved under runs/<run-name>.",
    )
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--check-env", action="store_true", help="Run Gymnasium environment checker before training")
    parser.add_argument("--render-test", action="store_true", help="Render one random-policy episode before training")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to save the final model zip")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for training logs/monitor files/eval/checkpoints (default: runs/ppo_catchme)",
    )
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Evaluation frequency in env timesteps")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Episodes per evaluation run")
    parser.add_argument("--checkpoint-freq", type=int, default=25_000, help="Checkpoint frequency in env timesteps")
    parser.add_argument("--no-eval", action="store_true", help="Disable periodic evaluation callback")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable periodic checkpoint callback")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.log_dir is not None:
        run_dir = args.log_dir
    elif args.run_name is not None:
        run_dir = ROOT / "runs" / args.run_name
    else:
        run_dir = DEFAULT_LOG_DIR

    eval_dir = run_dir / "eval"
    checkpoint_dir = run_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4, default=str)

    if args.check_env:
        print("Checking environment...")
        check_env(CatchMeEnv(), warn=True)

    if args.render_test:
        render_random_episode()

    env = make_vec_env(
        lambda: CatchMeEnv(max_steps=600),
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=str(run_dir)
    )

    logger = configure(
        folder=str(run_dir),
        format_strings=["stdout", "csv"],
    )

    eval_env = Monitor(CatchMeEnv(max_steps=600))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
    )

    model.set_logger(logger)

    callbacks = []
    if not args.no_eval:
        callbacks.append(
            EvalCallback(
                eval_env=eval_env,
                best_model_save_path=str(eval_dir),
                log_path=str(eval_dir),
                eval_freq=max(args.eval_freq // args.n_envs, 1),
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                render=False,
            )
        )
    if not args.no_checkpoint:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                save_path=str(checkpoint_dir),
                name_prefix="catchme_ppo",
            )
        )

    callback = CallbackList(callbacks) if callbacks else None

    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)
    model.save(args.model_path)
    final_steps = int(model.num_timesteps)

    # Keep the existing final model path, and also write a step-tagged copy.
    final_step_model_path = args.model_path.with_name(f"{args.model_path.stem}_{final_steps}_steps.zip")
    shutil.copy2(args.model_path, final_step_model_path)

    # Keep EvalCallback's best_model.zip, and also create a step-tagged snapshot copy.
    best_model_path = eval_dir / "best_model.zip"
    best_step_model_path = eval_dir / f"best_model_{final_steps}_steps.zip"
    if not args.no_eval and best_model_path.exists():
        shutil.copy2(best_model_path, best_step_model_path)

    env.close()
    eval_env.close()

    print(f"Saved final model to: {args.model_path}")
    print(f"Saved step-tagged final model to: {final_step_model_path}")
    if not args.no_eval:
        print(f"Best eval model path: {best_model_path}")
        if best_model_path.exists():
            print(f"Saved step-tagged best model to: {best_step_model_path}")
    if not args.no_checkpoint:
        print(f"Checkpoint directory: {checkpoint_dir}")

    plot_training_curves(run_dir)


def read_monitor_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, skiprows=1)


def plot_training_curves(run_dir: Path) -> None: 
    monitor_files = list(run_dir.glob("*.monitor.csv"))
    progress_path = run_dir / "progress.csv"

    if not monitor_files:
        print(f"No monitor files found in: {run_dir}")
    else:
        plot_episode_rewards(monitor_files[0], run_dir)
        plot_episode_lengths(monitor_files[0], run_dir)

    if not progress_path.exists():
        print(f"Progress file not found: {progress_path}")
    else:
        plot_losses(progress_path, run_dir)

def plot_episode_rewards(monitor_path: Path, run_dir: Path) -> None:
    df = read_monitor_csv(monitor_path)

    if "r" not in df.columns:
        print("Reward column 'r' not found in monitor.csv")
        return

    df["episode"] = range(1, len(df) + 1)
    df["reward_ma_20"] = df["r"].rolling(window=20, min_periods=1).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df["r"], alpha=0.35, label="Episode reward")
    plt.plot(df["episode"], df["reward_ma_20"], label="Moving avg reward - 20 episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = run_dir / "reward_curve.png"
    plt.savefig(output_path)
    plt.show()

    print(f"Saved reward curve to: {output_path}")


def plot_episode_lengths(monitor_path: Path, run_dir: Path) -> None:
    df = read_monitor_csv(monitor_path)

    if "l" not in df.columns:
        print("Episode length column 'l' not found in monitor.csv")
        return

    df["episode"] = range(1, len(df) + 1)
    df["length_ma_20"] = df["l"].rolling(window=20, min_periods=1).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df["l"], alpha=0.35, label="Episode length")
    plt.plot(df["episode"], df["length_ma_20"], label="Moving avg length - 20 episodes")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Episode Length Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = run_dir / "episode_length_curve.png"
    plt.savefig(output_path)
    plt.show()

    print(f"Saved episode length curve to: {output_path}")

def plot_losses(progress_path: Path, run_dir: Path) -> None:
    df = pd.read_csv(progress_path)

    possible_loss_columns = [
        "train/loss",
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
    ]

    available_columns = [col for col in possible_loss_columns if col in df.columns]

    if not available_columns:
        print("No PPO loss columns found in progress.csv")
        print("Available columns:", list(df.columns))
        return

    x_column = "time/total_timesteps" if "time/total_timesteps" in df.columns else None
    if x_column:
        df[x_column] = pd.to_numeric(df[x_column], errors="coerce")
    else:
        df["_log_step"] = pd.RangeIndex(start=0, stop=len(df), step=1)
        x_column = "_log_step"

    # Keep only numeric values and drop columns that are entirely NaN.
    plottable_columns: list[str] = []
    for col in available_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].notna().any():
            plottable_columns.append(col)

    if not plottable_columns:
        print("Loss columns exist but contain no numeric data yet.")
        return

    plt.figure(figsize=(10, 5))
    plotted_any = False

    for col in plottable_columns:
        mask = df[x_column].notna() & df[col].notna()
        if mask.any():
            plt.plot(df.loc[mask, x_column], df.loc[mask, col], label=col)
            plotted_any = True

    if not plotted_any:
        print("No valid points to plot for loss curves.")
        return

    plt.xlabel("Timesteps" if x_column == "time/total_timesteps" else "Log step")
    plt.ylabel("Loss / metric value")
    plt.title("PPO Training Loss Curves")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    output_path = run_dir / "loss_curves.png"
    plt.savefig(output_path)
    plt.show()

    print(f"Saved loss curves to: {output_path}")

def render_random_episode() -> None:
    env = CatchMeEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()


if __name__ == "__main__":
    main()
