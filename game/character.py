"""Character movement logic.

The RL agent controls this character. Positions are normalized:
- x = 0.0 is the left side of the screen
- x = 1.0 is the right side
- y = 0.0 is the top
- y = 1.0 is the bottom
"""


from dataclasses import dataclass, field 
import numpy as np


# 9 discrete moves: stay + 8 directions.
# This is intentionally simple for the first RL version.
ACTION_TO_DIRECTION: dict[int, tuple[float, float]] = {
    0: (0.0, 0.0),
    1: (0.0, -1.0),
    2: (0.0, 1.0),
    3: (-1.0, 0.0),
    4: (1.0, 0.0),
    5: (-1.0, -1.0),
    6: (1.0, -1.0),
    7: (-1.0, 1.0),
    8: (1.0, 1.0),
}


@dataclass
class Character:
    """Simple 2D character controlled by the RL policy."""

    position: np.ndarray
    max_speed: float = 0.035
    radius: float = 0.035
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))

    def apply_action(self, action: int) -> bool:
        """Move the character using a discrete action.

        Returns:
            True if movement tried to leave the [0, 1] screen bounds.
        """
        direction = np.array(ACTION_TO_DIRECTION[int(action)], dtype=np.float32)

        # Normalize diagonal movement so diagonals are not faster than straight moves.
        norm = float(np.linalg.norm(direction))
        if norm > 0:
            direction = direction / norm

        self.velocity = direction * self.max_speed
        next_position = self.position + self.velocity

        hit_wall = bool(np.any(next_position < 0.0) or np.any(next_position > 1.0))
        self.position = np.clip(next_position, 0.0, 1.0).astype(np.float32)
        return hit_wall

    def reset(self, position: np.ndarray) -> None:
        """Reset position and velocity."""
        self.position = np.asarray(position, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)