import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from envs.catch_env import CatchMeEnv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "catchme_ppo.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Catch Me If You Can RL agent")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--check-env", action="store_true", help="Run Gymnasium environment checker before training")
    parser.add_argument("--render-test", action="store_true", help="Render one random-policy episode before training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    if args.check_env:
        print("Checking environment...")
        check_env(CatchMeEnv(), warn=True)

    if args.render_test:
        render_random_episode()

    env = make_vec_env(
        lambda: CatchMeEnv(max_steps=600),
        n_envs=args.n_envs,
        seed=args.seed,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=args.seed,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(args.model_path)
    env.close()

    print(f"Saved model to: {args.model_path}")


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