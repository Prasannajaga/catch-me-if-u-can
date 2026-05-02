from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from envs.catch_env import CatchMeEnv
from envs.continuous_action_wrapper import ContinuousActionWrapper


ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = ROOT / "runs" / "ppo_catchme_continuous"
DEFAULT_MODEL_PATH = ROOT / "models" / "catchme_continuous_ppo.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Catch Me If You Can with continuous action [dx, dy]")
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
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--render-test", action="store_true", help="Render one random-policy episode before training")
    return parser.parse_args()


def make_continuous_env(*, max_steps: int, render_mode: str | None = None) -> ContinuousActionWrapper:
    base = CatchMeEnv(max_steps=max_steps, render_mode=render_mode)
    return ContinuousActionWrapper(base)


def render_random_episode(max_steps: int) -> None:
    env = make_continuous_env(max_steps=max_steps, render_mode="human")
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()


def main() -> None:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    if args.render_test:
        render_random_episode(max_steps=args.max_steps)

    env = make_vec_env(
        lambda: make_continuous_env(max_steps=args.max_steps, render_mode=None),
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=str(args.log_dir),
    )
    eval_env = Monitor(make_continuous_env(max_steps=args.max_steps, render_mode=None))

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

    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(args.log_dir / "eval"),
        log_path=str(args.log_dir / "eval"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_cb, progress_bar=True)
    model.save(args.model_path)
    env.close()
    eval_env.close()

    print(f"Saved model: {args.model_path}")
    print(f"Log dir: {args.log_dir}")


if __name__ == "__main__":
    main()
