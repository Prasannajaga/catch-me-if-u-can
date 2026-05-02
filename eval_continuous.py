from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from stable_baselines3 import PPO

from envs.catch_env import CatchMeEnv
from envs.continuous_action_wrapper import ContinuousActionWrapper


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "models" / "catchme_continuous_ppo.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate continuous-action Catch Me policy")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory under runs/ to append manual eval history")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def make_continuous_env(*, max_steps: int, render_mode: str | None = None) -> ContinuousActionWrapper:
    base = CatchMeEnv(max_steps=max_steps, render_mode=render_mode)
    return ContinuousActionWrapper(base)


def _infer_run_dir(model_path: Path) -> Path | None:
    parts = model_path.resolve().parts
    if "runs" not in parts:
        return None

    runs_idx = parts.index("runs")
    if runs_idx + 1 >= len(parts):
        return None

    run_dir = Path(*parts[: runs_idx + 2])
    return run_dir if run_dir.is_dir() else None


def _append_manual_eval_history(
    *,
    run_dir: Path,
    model_path: Path,
    rewards: list[float],
    lengths: list[int],
    caught_flags: list[bool],
) -> None:
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    summary_path = eval_dir / "manual_eval_history.csv"
    episode_path = eval_dir / "manual_eval_episodes.csv"

    now = datetime.now().isoformat(timespec="seconds")
    eval_id = uuid4().hex[:12]
    episodes = len(rewards)
    caught_count = int(sum(caught_flags))
    mean_reward = float(np.mean(rewards))
    mean_steps = float(np.mean(lengths))
    catch_rate = float(caught_count / episodes)
    survival_rate = float(1.0 - catch_rate)

    summary_fields = [
        "timestamp",
        "eval_id",
        "model_path",
        "episodes",
        "mean_reward",
        "mean_steps",
        "catch_rate",
        "survival_rate",
    ]
    summary_row = {
        "timestamp": now,
        "eval_id": eval_id,
        "model_path": str(model_path.resolve()),
        "episodes": episodes,
        "mean_reward": round(mean_reward, 6),
        "mean_steps": round(mean_steps, 6),
        "catch_rate": round(catch_rate, 6),
        "survival_rate": round(survival_rate, 6),
    }

    episode_fields = [
        "timestamp",
        "eval_id",
        "episode_index",
        "reward",
        "steps",
        "caught",
    ]

    episode_rows = []
    for idx, (reward, steps, caught) in enumerate(zip(rewards, lengths, caught_flags), start=1):
        episode_rows.append(
            {
                "timestamp": now,
                "eval_id": eval_id,
                "episode_index": idx,
                "reward": round(float(reward), 6),
                "steps": int(steps),
                "caught": bool(caught),
            }
        )

    write_summary_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        if write_summary_header:
            writer.writeheader()
        writer.writerow(summary_row)

    write_episode_header = not episode_path.exists()
    with open(episode_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=episode_fields)
        if write_episode_header:
            writer.writeheader()
        writer.writerows(episode_rows)


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}. Train first with python -m train_continuous")

    model = PPO.load(args.model_path)
    env = make_continuous_env(max_steps=args.max_steps, render_mode="human" if args.render else None)

    rewards: list[float] = []
    lengths: list[int] = []
    caught_count = 0
    caught_flags: list[bool] = []

    for episode in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        caught = False

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
            caught = bool(info["caught"])

        rewards.append(total_reward)
        lengths.append(steps)
        caught_count += int(caught)
        caught_flags.append(caught)
        print(f"episode={episode + 1:03d} reward={total_reward:8.2f} steps={steps:4d} caught={caught}")

    env.close()

    print("\nSummary")
    print(f"episodes:       {args.episodes}")
    print(f"mean_reward:    {np.mean(rewards):.2f}")
    print(f"mean_steps:     {np.mean(lengths):.2f}")
    print(f"catch_rate:     {caught_count / args.episodes:.2%}")
    print(f"survival_rate:  {(args.episodes - caught_count) / args.episodes:.2%}")

    run_dir = args.run_dir if args.run_dir is not None else _infer_run_dir(args.model_path)
    if run_dir is not None:
        _append_manual_eval_history(
            run_dir=run_dir,
            model_path=args.model_path,
            rewards=rewards,
            lengths=lengths,
            caught_flags=caught_flags,
        )
        print(f"manual_eval_saved: {run_dir / 'eval' / 'manual_eval_history.csv'}")
        print(f"manual_episode_saved: {run_dir / 'eval' / 'manual_eval_episodes.csv'}")
    else:
        print("manual_eval_saved: skipped (pass --run-dir to store history in a specific run)")


if __name__ == "__main__":
    main()
