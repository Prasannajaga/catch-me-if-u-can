from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ContinuousActionWrapper(gym.Wrapper):
    """Wrap CatchMeEnv with a continuous 2D movement action.

    Action format:
        action = [dx, dy], where each element is in [-1, 1].

    Movement behavior:
    - Direction comes from the action vector.
    - Magnitude controls speed (0 = no move, 1 = max speed).
    - Vector norm is clamped to 1 so diagonal movement is not faster.
    """

    action_space: spaces.Box

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_arr = np.asarray(action, dtype=np.float32).reshape(2)
        action_arr = np.clip(action_arr, -1.0, 1.0)

        self.env.step_count += 1
        previous_distance = self.env._distance(self.env.character.position, self.env.player_position)
        hit_wall = self._apply_continuous_action(action_arr)

        self.env.player_position, self.env.player_velocity = self.env.chaser.next_position(
            player_position=self.env.player_position,
            character_position=self.env.character.position,
            character_velocity=self.env.character.velocity,
            step_count=self.env.step_count,
        )

        distance = self.env._distance(self.env.character.position, self.env.player_position)
        caught = distance <= self.env.catch_radius
        terminated = bool(caught)
        truncated = bool(self.env.step_count >= self.env.max_steps)

        reward = self.env._reward(
            distance=distance,
            previous_distance=previous_distance,
            caught=caught,
            hit_wall=hit_wall,
        )
        if truncated and not caught:
            reward += 25.0

        obs = self.env._get_obs()
        info = self.env._get_info()

        if self.env.render_mode == "human":
            self.env.render()

        return obs, float(reward), terminated, truncated, info

    def _apply_continuous_action(self, action: np.ndarray) -> bool:
        norm = float(np.linalg.norm(action))
        direction = action.copy()
        if norm > 1.0:
            direction /= norm

        velocity = direction * float(self.env.character.max_speed)
        self.env.character.velocity = velocity.astype(np.float32)
        next_position = self.env.character.position + self.env.character.velocity

        margin = float(self.env.character.radius)
        hit_wall = bool(np.any(next_position < margin) or np.any(next_position > 1.0 - margin))
        self.env.character.position = np.clip(next_position, margin, 1.0 - margin).astype(np.float32)
        return hit_wall
