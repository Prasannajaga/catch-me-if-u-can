"""Rendering helpers.

Training/eval uses Pygame. Live webcam mode uses OpenCV overlay helpers.
"""

from __future__ import annotations

import numpy as np


class PygameRenderer:
    """Simple Pygame renderer for the simulated environment."""

    def __init__(self, *, width: int = 900, height: int = 700, fps: int = 30) -> None:
        import pygame

        self.pygame = pygame
        pygame.init()
        self.width = width
        self.height = height
        self.fps = fps
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Catch Me If You Can - RL Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)

    def draw(
        self,
        *,
        character_position: np.ndarray,
        player_position: np.ndarray,
        catch_radius: float,
        info: dict,
        return_rgb_array: bool = False,
    ) -> np.ndarray | None:
        pygame = self.pygame
        self._handle_events()

        self.screen.fill((18, 18, 24))
        self._draw_grid()

        player_xy = self._world_to_screen(player_position)
        char_xy = self._world_to_screen(character_position)

        catch_radius_px = int(catch_radius * min(self.width, self.height))
        pygame.draw.circle(self.screen, (90, 90, 120), player_xy, catch_radius_px, width=2)
        pygame.draw.circle(self.screen, (245, 90, 90), player_xy, 16)
        pygame.draw.circle(self.screen, (80, 220, 140), char_xy, 14)

        pygame.draw.line(self.screen, (90, 90, 100), player_xy, char_xy, width=1)

        text = f"step={info['step']} distance={info['distance']:.3f} caught={info['caught']}"
        text_surface = self.font.render(text, True, (230, 230, 235))
        self.screen.blit(text_surface, (16, 16))

        pygame.display.flip()
        self.clock.tick(self.fps)

        if return_rgb_array:
            return np.transpose(pygame.surfarray.array3d(self.screen), axes=(1, 0, 2))
        return None

    def close(self) -> None:
        self.pygame.quit()

    def _world_to_screen(self, position: np.ndarray) -> tuple[int, int]:
        x = int(float(position[0]) * self.width)
        y = int(float(position[1]) * self.height)
        return x, y

    def _draw_grid(self) -> None:
        pygame = self.pygame
        for x in range(0, self.width, 50):
            pygame.draw.line(self.screen, (28, 28, 36), (x, 0), (x, self.height))
        for y in range(0, self.height, 50):
            pygame.draw.line(self.screen, (28, 28, 36), (0, y), (self.width, y))

    def _handle_events(self) -> None:
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit


def draw_live_overlay(
    frame_bgr: np.ndarray,
    *,
    character_position: np.ndarray,
    player_position: np.ndarray | None,
    catch_radius: float,
    caught: bool,
    status_text: str,
) -> np.ndarray:
    """Draw the trained character on top of a webcam frame."""
    import cv2

    output = frame_bgr.copy()
    height, width = output.shape[:2]

    def to_px(position: np.ndarray) -> tuple[int, int]:
        return int(float(position[0]) * width), int(float(position[1]) * height)

    char_xy = to_px(character_position)
    char_radius = max(8, int(catch_radius * min(width, height) * 0.55))

    cv2.circle(output, char_xy, char_radius + 4, (20, 20, 20), thickness=-1)
    cv2.circle(output, char_xy, char_radius, (80, 220, 140), thickness=-1)

    if player_position is not None:
        player_xy = to_px(player_position)
        catch_radius_px = int(catch_radius * min(width, height))
        cv2.circle(output, player_xy, 8, (70, 70, 255), thickness=-1)
        cv2.circle(output, player_xy, catch_radius_px, (100, 100, 255), thickness=2)
        cv2.line(output, player_xy, char_xy, (90, 90, 90), thickness=1)

    if caught:
        cv2.putText(output, "CAUGHT", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (50, 50, 255), 3)

    cv2.putText(output, status_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2)
    cv2.putText(output, "press q to quit", (20, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2)
    return output