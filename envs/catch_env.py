from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.player_bot import ChaserBot
from game.character import Character

RenderMode = Literal["human", "rgb_array", None]


class CatchMeEnv(gym.Env):
    """Custom 2D escape environment.

    Observation vector:
        [
            character_x,
            character_y,
            player_x,
            player_y,
            delta_x,              # character_x - player_x
            delta_y,              # character_y - player_y
            distance_to_player,
            character_velocity_x,
            character_velocity_y,
            player_velocity_x,
            player_velocity_y,
        ]

    Action space:
        0 = stay
        1 = up
        2 = down
        3 = left
        4 = right
        5 = up-left
        6 = up-right
        7 = down-left
        8 = down-right
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        *,
        max_steps: int = 600,
        catch_radius: float = 0.075,
        character_speed: float = 0.035,
        player_speed: float = 0.026,
        render_mode: RenderMode = None,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.catch_radius = catch_radius
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(9)

        low = np.array([0, 0, 0, 0, -1, -1, 0, -1, -1, -1, -1], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1, np.sqrt(2), 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.character = Character(
            position=np.array([0.5, 0.5], dtype=np.float32),
            max_speed=character_speed,
            radius=catch_radius * 0.55,
        )
        self.chaser = ChaserBot(speed=player_speed, mode="random_mixed")
        self.player_position = np.array([0.2, 0.2], dtype=np.float32)
        self.player_velocity = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self._renderer = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.step_count = 0
        self.chaser.reset(self.np_random)

        # Keep initial positions separated so every episode starts playable.
        self.character.reset(self._sample_position())
        self.player_position = self._sample_position()
        while self._distance(self.character.position, self.player_position) < 0.35:
            self.player_position = self._sample_position()

        self.player_velocity = np.zeros(2, dtype=np.float32)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1

        previous_distance = self._distance(self.character.position, self.player_position)
        hit_wall = self.character.apply_action(int(action))

        self.player_position, self.player_velocity = self.chaser.next_position(
            player_position=self.player_position,
            character_position=self.character.position,
            character_velocity=self.character.velocity,
            step_count=self.step_count,
        )

        distance = self._distance(self.character.position, self.player_position)
        caught = distance <= self.catch_radius
        terminated = bool(caught)
        truncated = bool(self.step_count >= self.max_steps)

        reward = self._reward(
            distance=distance,
            previous_distance=previous_distance,
            caught=caught,
            hit_wall=hit_wall,
        )

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode not in {"human", "rgb_array"}:
            return None

        from game.renderer import PygameRenderer

        if self._renderer is None:
            self._renderer = PygameRenderer(width=900, height=700, fps=self.metadata["render_fps"])

        return self._renderer.draw(
            character_position=self.character.position,
            player_position=self.player_position,
            catch_radius=self.catch_radius,
            info=self._get_info(),
            return_rgb_array=self.render_mode == "rgb_array",
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _sample_position(self) -> np.ndarray:
        return self.np_random.uniform(low=0.08, high=0.92, size=(2,)).astype(np.float32)

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _get_obs(self) -> np.ndarray:
        delta = self.character.position - self.player_position
        distance = np.array([np.linalg.norm(delta)], dtype=np.float32)
        obs = np.concatenate(
            [
                self.character.position,
                self.player_position,
                delta.astype(np.float32),
                distance,
                self.character.velocity.astype(np.float32),
                self.player_velocity.astype(np.float32),
            ]
        ).astype(np.float32)
        return obs

    def _get_info(self) -> dict[str, Any]:
        distance = self._distance(self.character.position, self.player_position)
        return {
            "step": self.step_count,
            "distance": distance,
            "caught": distance <= self.catch_radius,
            "character_position": self.character.position.copy(),
            "player_position": self.player_position.copy(),
        }

    def _reward(
        self,
        *,
        distance: float,
        previous_distance: float,
        caught: bool,
        hit_wall: bool,
    ) -> float:
        if caught:
            return -10.0

        # Main goal: stay away from the player.
        distance_reward = 1.5 * distance

        # Encourage increasing the distance, not only sitting far away.
        escape_reward = 4.0 * (distance - previous_distance)

        # Small survival bonus.
        survival_reward = 0.03

        # Avoid degenerate policy: hiding at borders/corners.
        x, y = self.character.position
        edge_distance = min(x, 1.0 - x, y, 1.0 - y)
        edge_margin = 0.08
        edge_penalty = max(0.0, edge_margin - edge_distance) / edge_margin

        wall_penalty = 0.2 if hit_wall else 0.0

        return distance_reward + escape_reward + survival_reward - edge_penalty - wall_penalty