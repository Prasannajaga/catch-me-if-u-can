"""Fake player/chaser used during training.

The real player comes from the webcam later. During training we need a fast,
repeatable opponent so the RL character can learn without waiting for a human.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

BotMode = Literal["direct", "predictive", "zigzag", "random_mixed"]


@dataclass
class ChaserBot:
    """A simple bot that chases the character."""

    speed: float = 0.025
    mode: BotMode = "random_mixed"
    prediction_scale: float = 8.0

    def reset(self, rng: np.random.Generator) -> None:
        """Randomize mode if requested."""
        if self.mode == "random_mixed":
            self._episode_mode: BotMode = rng.choice(["direct", "predictive", "zigzag"]).item()
        else:
            self._episode_mode = self.mode

    def next_position(
        self,
        *,
        player_position: np.ndarray,
        character_position: np.ndarray,
        character_velocity: np.ndarray,
        step_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Move the fake player toward the character.

        Returns:
            (new_player_position, player_velocity)
        """
        target = np.asarray(character_position, dtype=np.float32)

        if self._episode_mode == "predictive":
            target = target + character_velocity * self.prediction_scale
        elif self._episode_mode == "zigzag":
            # Chases while adding a small perpendicular oscillation.
            chase = character_position - player_position
            perp = np.array([-chase[1], chase[0]], dtype=np.float32)
            perp_norm = float(np.linalg.norm(perp))
            if perp_norm > 0:
                perp = perp / perp_norm
            target = target + 0.12 * np.sin(step_count * 0.25) * perp

        direction = target - player_position
        norm = float(np.linalg.norm(direction))
        if norm > 0:
            direction = direction / norm

        velocity = direction.astype(np.float32) * self.speed
        new_position = np.clip(player_position + velocity, 0.0, 1.0).astype(np.float32)
        return new_position, velocity.astype(np.float32)