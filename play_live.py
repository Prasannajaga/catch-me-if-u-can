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

INVULNERABILITY_FRAMES = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Catch Me If You Can with webcam hand tracking")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fullscreen", action="store_true", help="Run the live window in fullscreen mode")
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=0.06,
        help="Expand hand coordinate range near borders (0.0 disables; typical 0.03-0.08).",
    )
    parser.add_argument("--catch-radius", type=float, default=0.075)
    parser.add_argument("--character-speed", type=float, default=0.035)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}. Train first with python -m catchme.train")

    model = PPO.load(args.model_path)
    rng = np.random.default_rng(42)
    character = Character(
        position=np.array([0.5, 0.5], dtype=np.float32),
        max_speed=args.character_speed,
        radius=args.catch_radius * 0.55,
    )

    previous_player_position: np.ndarray | None = None
    invulnerable_remaining = 0

    camera_config = CameraConfig(
        device_index=args.camera,
        width=args.width,
        height=args.height,
        flip_horizontal=True,
    )

    edge_margin = float(np.clip(args.edge_margin, 0.0, 0.45))
    window_name = "Catch Me If You Can - Live"
    fullscreen_helper = FullscreenWindow(window_name, enabled=args.fullscreen)
    fullscreen_helper.create()
    if not args.fullscreen:
        cv2.resizeWindow(window_name, args.width, args.height)

    with Camera(camera_config) as camera, HandTracker() as tracker:
        while True:
            frame = camera.read()
            hand_point = tracker.detect(frame)
            if hand_point is not None and edge_margin > 0.0:
                hand_point.x = stretch_edge_coordinate(hand_point.x, margin=edge_margin)
                hand_point.y = stretch_edge_coordinate(hand_point.y, margin=edge_margin)

            caught = False
            player_position: np.ndarray | None = None
            status_text = "show your hand to start"

            if invulnerable_remaining > 0:
                invulnerable_remaining -= 1

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
                status_text = f"distance={distance:.3f}"

                # Only check catch when invulnerability has expired.
                if invulnerable_remaining <= 0 and distance <= args.catch_radius:
                    caught = True
                    character.reset(sample_safe_position(rng, player_position))
                    invulnerable_remaining = INVULNERABILITY_FRAMES
                    status_text = "caught - respawned"
                elif invulnerable_remaining > 0:
                    status_text += f"  [shield {invulnerable_remaining}]"
            else:
                previous_player_position = None

            output = draw_live_overlay(
                frame,
                character_position=character.position,
                character_velocity=character.velocity,
                player_position=player_position,
                catch_radius=args.catch_radius,
                caught=caught,
                status_text=status_text,
            )

            cv2.imshow(window_name, output)
            fullscreen_helper.apply_if_needed()
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


def sample_safe_position(
    rng: np.random.Generator,
    player_position: np.ndarray,
    margin: float = 0.10,
    min_distance: float = 0.45,
) -> np.ndarray:
    """Respawn the character away from the hand and inset from edges.

    ``margin`` keeps the character's sprite fully on-screen.
    ``min_distance`` ensures the respawn isn't immediately on top of the player.
    """
    for _ in range(100):
        position = rng.uniform(low=margin, high=1.0 - margin, size=(2,)).astype(np.float32)
        if float(np.linalg.norm(position - player_position)) > min_distance:
            return position
    # Fallback — centre of the screen.
    return np.array([0.5, 0.5], dtype=np.float32)


def stretch_edge_coordinate(value: float, margin: float) -> float:
    """Expand interior [margin, 1-margin] to full [0, 1] to improve edge reach."""
    if margin <= 0.0:
        return float(np.clip(value, 0.0, 1.0))
    lo = margin
    hi = 1.0 - margin
    if hi <= lo:
        return float(np.clip(value, 0.0, 1.0))
    stretched = (float(value) - lo) / (hi - lo)
    return float(np.clip(stretched, 0.0, 1.0))


class FullscreenWindow:
    """Best-effort cross-platform fullscreen helper for OpenCV HighGUI."""

    def __init__(self, window_name: str, *, enabled: bool) -> None:
        self.window_name = window_name
        self.enabled = enabled
        self._applied = False
        self._tries = 0
        self._max_tries = 20
        self._wnd_prop_fullscreen = getattr(cv2, "WND_PROP_FULLSCREEN", 0)
        self._window_fullscreen = getattr(cv2, "WINDOW_FULLSCREEN", 1)

    def create(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Some Linux backends only honor fullscreen after first imshow.
        if self.enabled:
            self._try_apply_fullscreen()

    def apply_if_needed(self) -> None:
        if not self.enabled or self._applied:
            return
        if self._tries >= self._max_tries:
            return
        self._try_apply_fullscreen()

    def _try_apply_fullscreen(self) -> None:
        self._tries += 1
        try:
            cv2.setWindowProperty(
                self.window_name,
                self._wnd_prop_fullscreen,
                float(self._window_fullscreen),
            )
            current = cv2.getWindowProperty(self.window_name, self._wnd_prop_fullscreen)
            if current >= 0:
                self._applied = True
        except cv2.error:
            # Keep trying over the next few frames for Linux/Wayland/X11 quirks.
            return


if __name__ == "__main__":
    main()
