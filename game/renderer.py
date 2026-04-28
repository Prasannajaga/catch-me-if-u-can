"""Rendering helpers.

Training/eval uses Pygame. Live webcam mode uses OpenCV overlay helpers.
"""

from __future__ import annotations

import numpy as np
import cv2
import time


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


# ---------------------------------------------------------------------------
# Live-overlay sprite cache
# ---------------------------------------------------------------------------
_STICKMAN_RAW: np.ndarray | None = None
_SPRITE_CACHE: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}


def _get_sprite_frames(
    sprite_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return cached list of (colored_frame, mask_3ch) tuples for each animation frame."""
    global _STICKMAN_RAW

    if sprite_size in _SPRITE_CACHE:
        return _SPRITE_CACHE[sprite_size]

    if _STICKMAN_RAW is None:
        sprite_path = "assets/stickman.png"
        _STICKMAN_RAW = cv2.imread(sprite_path)

    if _STICKMAN_RAW is None:
        _SPRITE_CACHE[sprite_size] = []
        return []

    s_h, s_w = _STICKMAN_RAW.shape[:2]
    f_w, f_h = s_w // 2, s_h // 2

    frames: list[tuple[np.ndarray, np.ndarray]] = []
    for idx in range(4):
        fx, fy = (idx % 2) * f_w, (idx // 2) * f_h
        raw_frame = _STICKMAN_RAW[fy : fy + f_h, fx : fx + f_w]
        resized = cv2.resize(raw_frame, (sprite_size, sprite_size))

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, mask_1ch = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        mask_3ch = cv2.cvtColor(mask_1ch, cv2.COLOR_GRAY2BGR)

        colored = np.full_like(resized, (80, 220, 140))
        frames.append((colored, mask_3ch))

    _SPRITE_CACHE[sprite_size] = frames
    return frames


def draw_live_overlay(
    frame_bgr: np.ndarray,
    *,
    character_position: np.ndarray,
    player_position: np.ndarray | None,
    catch_radius: float,
    caught: bool,
    status_text: str,
) -> np.ndarray:
    """Draw the trained character on top of a webcam frame.

    Draws directly on *frame_bgr* to avoid a full-frame copy.
    The caller should not reuse the input array after this call.
    """
    output = frame_bgr
    height, width = output.shape[:2]

    char_x = int(float(character_position[0]) * width)
    char_y = int(float(character_position[1]) * height)
    char_radius = max(8, int(catch_radius * min(width, height) * 0.55))

    # --- ANIMATE STICKMAN SPRITE (cached, single-pass compositing) ---
    sprite_size = char_radius * 4
    sprite_frames = _get_sprite_frames(sprite_size)

    if sprite_frames:
        frame_idx = int(time.time() * 8) % 4
        colored, mask_3ch = sprite_frames[frame_idx]

        half = sprite_size // 2
        y1_raw, y2_raw = char_y - half, char_y + half
        x1_raw, x2_raw = char_x - half, char_x + half

        y1, y2 = max(0, y1_raw), min(height, y2_raw)
        x1, x2 = max(0, x1_raw), min(width, x2_raw)

        if y2 > y1 and x2 > x1:
            sy1 = y1 - y1_raw
            sy2 = sy1 + (y2 - y1)
            sx1 = x1 - x1_raw
            sx2 = sx1 + (x2 - x1)

            mask_clip = mask_3ch[sy1:sy2, sx1:sx2]
            color_clip = colored[sy1:sy2, sx1:sx2]
            roi = output[y1:y2, x1:x2]

            # Single-pass compositing via np.where (mask is 0 or 255).
            output[y1:y2, x1:x2] = np.where(mask_clip, color_clip, roi)
    else:
        cv2.circle(output, (char_x, char_y), char_radius, (80, 220, 140), thickness=-1)

    if player_position is not None:
        player_x = int(float(player_position[0]) * width)
        player_y = int(float(player_position[1]) * height)
        catch_radius_px = int(catch_radius * min(width, height))
        cv2.circle(output, (player_x, player_y), catch_radius_px, (255, 180, 80), thickness=2)

    if caught:
        cv2.putText(output, "CAUGHT", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 0), 3)

    cv2.putText(output, status_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2)
    cv2.putText(output, "press q to quit", (20, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2)
    return output