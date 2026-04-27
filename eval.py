from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO 
from envs.catch_env import CatchMeEnv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "catchme_ppo.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained Catch Me policy")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}. Train first with python -m catchme.train")

    model = PPO.load(args.model_path)
    env = CatchMeEnv(render_mode="human" if args.render else None)

    rewards: list[float] = []
    lengths: list[int] = []
    caught_count = 0

    for episode in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        caught = False

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
            caught = bool(info["caught"])

        rewards.append(total_reward)
        lengths.append(steps)
        caught_count += int(caught)
        print(f"episode={episode + 1:03d} reward={total_reward:8.2f} steps={steps:4d} caught={caught}")

    env.close()

    print("\nSummary")
    print(f"episodes:       {args.episodes}")
    print(f"mean_reward:    {np.mean(rewards):.2f}")
    print(f"mean_steps:     {np.mean(lengths):.2f}")
    print(f"catch_rate:     {caught_count / args.episodes:.2%}")
    print(f"survival_rate:  {(args.episodes - caught_count) / args.episodes:.2%}")


if __name__ == "__main__":
    main()