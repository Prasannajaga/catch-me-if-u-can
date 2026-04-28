import argparse
from pathlib import Path
import json

from stable_baselines3 import PPO
import pandas as pd
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from envs.catch_env import CatchMeEnv


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Catch Me If You Can RL agent")
    parser.add_argument("--run-name", type=str, default="catchme_ppo_run", help="Name for this training run")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    model_path = run_dir / "catchme_ppo.zip"

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

    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(model_path)
    env.close()

    print(f"Saved model to: {model_path}")

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


import matplotlib.ticker as ticker

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

    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    for col in available_columns:
        if x_column:
            plt.plot(df[x_column], df[col], label=col)
        else:
            plt.plot(df.index, df[col], label=col)

    # Make the y-axis ticks follow a gap of 5 (0, 5, 10, 15...)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

    plt.xlabel("Timesteps" if x_column else "Log step")
    plt.ylabel("Loss / metric value")
    plt.title("PPO Training Loss Curves")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
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