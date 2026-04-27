from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from stable_baselines3 import PPO

from game.character import Character
from game.renderer import draw_live_overlay
from vision.camera import Camera, CameraConfig
from vision.hand_tracker import HandTracker


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "catchme_ppo.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Catch Me If You Can with webcam hand tracking")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--catch-radius", type=float, default=0.075)
    parser.add_argument("--character-speed", type=float, default=0.035)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}. Train first with python -m catchme.train")

    model = PPO.load(args.model_path )
    rng = np.random.default_rng(42)
    character = Character(
        position=np.array([0.5, 0.5], dtype=np.float32),
        max_speed=args.character_speed,
        radius=args.catch_radius * 0.55,
    )

    previous_player_position: np.ndarray | None = None

    camera_config = CameraConfig(
        device_index=args.camera,
        width=args.width,
        height=args.height,
        flip_horizontal=True,
    )

    with Camera(camera_config) as camera, HandTracker() as tracker:
        while True:
            frame = camera.read()
            hand_point = tracker.detect(frame)

            caught = False
            player_position: np.ndarray | None = None
            status_text = "show your hand to start"

            if hand_point is not None:
                player_position = hand_point.as_array()
                if previous_player_position is None:
                    player_velocity = np.zeros(2, dtype=np.float32)
                else:
                    player_velocity = (player_position - previous_player_position).astype(np.float32)
                previous_player_position = player_position.copy()

                obs = build_live_observation(
                    character_position=character.position,
                    player_position=player_position,
                    character_velocity=character.velocity,
                    player_velocity=player_velocity,
                )

                action, _state = model.predict(obs, deterministic=True)
                character.apply_action(int(action))

                distance = float(np.linalg.norm(character.position - player_position))
                caught = distance <= args.catch_radius
                status_text = f"distance={distance:.3f}"

                if caught:
                    character.reset(sample_safe_position(rng, player_position))
                    status_text = "caught - respawned"
            else:
                previous_player_position = None

            output = draw_live_overlay(
                frame,
                character_position=character.position,
                player_position=player_position,
                catch_radius=args.catch_radius,
                caught=caught,
                status_text=status_text,
            )

            cv2.imshow("Catch Me If You Can - Live", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


def build_live_observation(
    *,
    character_position: np.ndarray,
    player_position: np.ndarray,
    character_velocity: np.ndarray,
    player_velocity: np.ndarray,
) -> np.ndarray:
    """Build the same observation layout used by CatchMeEnv."""
    delta = character_position - player_position
    distance = np.array([np.linalg.norm(delta)], dtype=np.float32)
    return np.concatenate(
        [
            character_position.astype(np.float32),
            player_position.astype(np.float32),
            delta.astype(np.float32),
            distance,
            character_velocity.astype(np.float32),
            player_velocity.astype(np.float32),
        ]
    ).astype(np.float32)


def sample_safe_position(rng: np.random.Generator, player_position: np.ndarray) -> np.ndarray:
    """Respawn the character away from the hand."""
    position = rng.uniform(low=0.08, high=0.92, size=(2,)).astype(np.float32)
    for _ in range(100):
        if float(np.linalg.norm(position - player_position)) > 0.45:
            return position
        position = rng.uniform(low=0.08, high=0.92, size=(2,)).astype(np.float32)
    return position


if __name__ == "__main__":
    main()