"""Pygame rendering."""

import pygame
import numpy as np
from scipy.ndimage import gaussian_filter
from . import config


class Renderer:
    """Handles all rendering."""
    
    def __init__(self):
        self.screen = pygame.display.set_mode(
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        )
        pygame.display.set_caption("Loss Landscape Visualizer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

    def world_to_screen(self, world_pos):
        """Map world coordinates to screen coordinates across top+bottom panels."""
        wx, wy = float(world_pos[0]), float(world_pos[1])
        sx = int(wx / (config.LANDSCAPE_WIDTH - 1) * (config.WINDOW_WIDTH - 1))

        if wy < config.LANDSCAPE_HEIGHT:
            sy_local = int(wy / (config.LANDSCAPE_HEIGHT - 1) * (config.TOP_PANEL_HEIGHT - 1))
        else:
            by = wy - config.LANDSCAPE_HEIGHT
            sy_local = int(
                config.TOP_PANEL_HEIGHT
                + by / (config.LANDSCAPE_HEIGHT - 1) * (config.BOTTOM_PANEL_HEIGHT - 1)
            )
        sy = config.MENU_BAR_HEIGHT + sy_local
        return sx, sy

    def _apply_colormap(self, values):
        """Apply a vivid multi-color palette to normalized [0, 1] values."""
        stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        colors = np.array([
            [25, 25, 110],
            [40, 170, 230],
            [70, 210, 120],
            [250, 210, 80],
            [215, 45, 45],
        ], dtype=np.float32)

        flat = values.reshape(-1)
        idx = np.searchsorted(stops, flat, side="right") - 1
        idx = np.clip(idx, 0, len(stops) - 2)
        left = stops[idx]
        right = stops[idx + 1]
        t = (flat - left) / (right - left + 1e-8)
        rgb = (1.0 - t)[:, None] * colors[idx] + t[:, None] * colors[idx + 1]
        return rgb.reshape(values.shape[0], values.shape[1], 3).astype(np.uint8)

    def _build_heatmap(self, landscape, distances=None, strict_match=False):
        """Build normalized heatmap used by both rendering and color sampling."""
        if distances is not None:
            valid = np.isfinite(distances)
            if not np.any(valid):
                return np.zeros_like(landscape)

            heatmap = np.zeros_like(distances, dtype=np.float64)
            valid_values = distances[valid]
            heatmap[valid] = valid_values / (valid_values.max() + 1e-6)

            if strict_match:
                # Strict mode keeps raw score structure for drop-vs-map fidelity.
                lo = float(heatmap[valid].min())
                hi = float(heatmap[valid].max())
                if hi <= lo:
                    heatmap = np.zeros_like(heatmap)
                else:
                    heatmap[valid] = (heatmap[valid] - lo) / (hi - lo)
                return np.clip(heatmap, 0.0, 1.0)

            # Visual smoothing mode for prettier maps.
            heatmap = gaussian_filter(heatmap, sigma=1.0)
        else:
            return np.zeros_like(landscape)

        # Robust normalization reduces visual overreaction to extreme values.
        lo = float(np.percentile(heatmap[valid], 2.0))
        hi = float(np.percentile(heatmap[valid], 98.0))
        if hi <= lo:
            heatmap = np.zeros_like(heatmap)
        else:
            heatmap[valid] = np.clip((heatmap[valid] - lo) / (hi - lo), 0.0, 1.0)
        return heatmap

    def sample_landscape_color(self, world_pos, landscape, distances=None, strict_match=False):
        """Sample rendered landscape color at a world position in top panel space."""
        heatmap = self._build_heatmap(
            landscape,
            distances,
            strict_match=strict_match,
        )

        x = int(np.clip(round(float(world_pos[0])), 0, heatmap.shape[1] - 1))
        y = int(np.clip(round(float(world_pos[1])), 0, heatmap.shape[0] - 1))
        rgb = self._apply_colormap(heatmap)
        px = rgb[y, x]
        return int(px[0]), int(px[1]), int(px[2])
    
    def draw_landscape(self, landscape, distances=None, strict_match=False):
        """
        Draw landscape heatmap in top panel.
        
        Args:
            landscape: (H, W) height map
            distances: (H, W) distance scores (or None)
            strict_match: if True, use raw distance normalization (no smoothing)
        """
        h, w = landscape.shape
        
        heatmap = self._build_heatmap(
            landscape,
            distances,
            strict_match=strict_match,
        )
        
        # Create RGB image with vivid multi-color mapping
        img = self._apply_colormap(heatmap)
        
        # Scale to fill full top panel area
        scaled = pygame.image.fromstring(img.tobytes(), (w, h), "RGB")
        scaled = pygame.transform.smoothscale(
            scaled,
            (config.WINDOW_WIDTH, config.TOP_PANEL_HEIGHT),
        )
        
        # Draw on top panel
        self.screen.blit(scaled, (0, config.MENU_BAR_HEIGHT))
    
    def draw_obstacles(self, obstacles):
        """Draw obstacles in bottom panel."""
        y_offset = config.MENU_BAR_HEIGHT + config.TOP_PANEL_HEIGHT
        
        for obs in obstacles:
            if obs['type'] == 'circle':
                pygame.draw.circle(
                    self.screen,
                    config.COLOR_OBSTACLE,
                    (int(obs['x']), int(obs['y']) + y_offset),
                    obs['radius']
                )
            elif obs['type'] == 'box':
                # TODO: Draw rotated box
                pygame.draw.rect(
                    self.screen,
                    config.COLOR_OBSTACLE,
                    (int(obs['x'] - obs['width']/2), 
                     int(obs['y'] + y_offset - obs['height']/2),
                     obs['width'], obs['height'])
                )
    
    def draw_target(self, target_pos):
        """Draw target marker."""
        if target_pos is not None:
            pygame.draw.circle(
                self.screen, config.COLOR_TARGET,
                self.world_to_screen(target_pos),
                8,
            )

    def draw_live_balls(self, balls):
        """Draw all balls with semi-transparent trails using each ball's color."""
        trail_surface = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.SRCALPHA)

        for ball in balls:
            ball_pos = ball["pos"]
            trail = ball["trail"]
            color = ball.get("color", config.COLOR_BALL)

            if trail:
                points = [self.world_to_screen(p) for p in trail]
                if len(points) >= 2:
                    trail_color = (color[0], color[1], color[2], config.TRAIL_ALPHA)
                    pygame.draw.lines(trail_surface, trail_color, False, points, 2)

        self.screen.blit(trail_surface, (0, 0))

        for ball in balls:
            ball_pos = ball["pos"]
            color = ball.get("color", config.COLOR_BALL)
            pygame.draw.circle(
                self.screen,
                color,
                self.world_to_screen(ball_pos),
                config.BALL_RADIUS_SCREEN,
            )
    
    def draw_hud(
        self,
        bounce_factor,
        paused,
        target_edit_mode,
        menu_open,
        full_pixel_scoring,
        fps,
    ):
        """Draw HUD text."""
        text = (
            f"Bounce: {bounce_factor:.2f} | "
            f"FPS: {fps:.1f} | "
            f"Score: DISTANCE | "
            f"Score mode: {'FULL PIXEL' if full_pixel_scoring else 'SAMPLED'} | "
            f"Target mode: {'ON' if target_edit_mode else 'OFF'} | "
            f"Menu: {'ON' if menu_open else 'OFF'} | "
            f"{'PAUSED' if paused else 'RUNNING'}"
        )
        surf = pygame.font.Font(None, 24).render(text, True, (255, 255, 255))
        self.screen.blit(surf, (10, config.MENU_BAR_HEIGHT + 6))

        help_text = (
            "Top click: drop ball(s) | Bottom click: obstacle | T: target mode | "
            "M: menu | P: full-pixel score | , .: bounce"
        )
        help_surf = pygame.font.Font(None, 24).render(help_text, True, (220, 220, 220))
        self.screen.blit(help_surf, (10, config.MENU_BAR_HEIGHT + 30))

    def draw_top_menu(self, menu_open, sections, active_section_index, rows, compute_running, compute_progress):
        """Draw top menu bar, section tabs, and section controls."""
        bar_rect = pygame.Rect(0, 0, config.WINDOW_WIDTH, config.MENU_BAR_HEIGHT)
        pygame.draw.rect(self.screen, (22, 24, 28), bar_rect)
        pygame.draw.line(
            self.screen,
            (58, 63, 74),
            (0, config.MENU_BAR_HEIGHT - 1),
            (config.WINDOW_WIDTH, config.MENU_BAR_HEIGHT - 1),
            1,
        )

        tab_font = pygame.font.Font(None, 24)
        for idx, name in enumerate(sections):
            x = config.MENU_LEFT_PAD + idx * (config.MENU_TAB_WIDTH + config.MENU_TAB_GAP)
            rect = pygame.Rect(x, 6, config.MENU_TAB_WIDTH, config.MENU_BAR_HEIGHT - 12)
            active = idx == active_section_index
            fill = (63, 120, 205) if active else (45, 49, 58)
            pygame.draw.rect(self.screen, fill, rect, border_radius=6)
            txt = tab_font.render(name, True, (245, 245, 245))
            self.screen.blit(txt, txt.get_rect(center=rect.center))

        # Progress indicator on the right.
        if compute_running:
            bar_w = 220
            bar_h = 14
            bar_x = config.WINDOW_WIDTH - bar_w - 14
            bar_y = (config.MENU_BAR_HEIGHT - bar_h) // 2
            outer = pygame.Rect(bar_x, bar_y, bar_w, bar_h)
            pygame.draw.rect(self.screen, (60, 60, 60), outer, border_radius=7)
            fill_w = int((bar_w - 2) * max(0.0, min(1.0, compute_progress)))
            inner = pygame.Rect(bar_x + 1, bar_y + 1, fill_w, bar_h - 2)
            pygame.draw.rect(self.screen, (80, 190, 110), inner, border_radius=6)
            pct = int(compute_progress * 100)
            pct_txt = tab_font.render(f"Computing {pct}%", True, (245, 245, 245))
            self.screen.blit(pct_txt, (bar_x - 118, bar_y - 4))

        if not menu_open:
            return

        panel_x = config.MENU_PANEL_PAD
        panel_y = config.MENU_BAR_HEIGHT + config.MENU_PANEL_PAD
        panel_h = config.MENU_PANEL_PAD * 2 + len(rows) * config.MENU_ROW_HEIGHT
        panel = pygame.Surface((config.MENU_PANEL_WIDTH, panel_h), pygame.SRCALPHA)
        panel.fill((18, 18, 20, 230))
        self.screen.blit(panel, (panel_x, panel_y))

        row_font = pygame.font.Font(None, 24)
        for idx, row in enumerate(rows):
            y = panel_y + config.MENU_PANEL_PAD + idx * config.MENU_ROW_HEIGHT
            label = row_font.render(row["label"], True, (235, 235, 235))
            self.screen.blit(label, (panel_x + 10, y + 6))

            if row["kind"] == "number":
                value = row_font.render(f"{row['value']:.3f}", True, (255, 223, 140))
                self.screen.blit(value, (panel_x + 190, y + 6))
                minus_rect = pygame.Rect(panel_x + 275, y + 3, 28, 24)
                plus_rect = pygame.Rect(panel_x + 312, y + 3, 28, 24)
                pygame.draw.rect(self.screen, (66, 71, 82), minus_rect, border_radius=4)
                pygame.draw.rect(self.screen, (66, 71, 82), plus_rect, border_radius=4)
                self.screen.blit(row_font.render("-", True, (245, 245, 245)), (minus_rect.x + 9, minus_rect.y + 3))
                self.screen.blit(row_font.render("+", True, (245, 245, 245)), (plus_rect.x + 8, plus_rect.y + 3))
            elif row["kind"] == "toggle":
                state_txt = "ON" if row["value"] else "OFF"
                color = (90, 195, 115) if row["value"] else (190, 92, 92)
                badge = pygame.Rect(panel_x + 270, y + 3, 70, 24)
                pygame.draw.rect(self.screen, color, badge, border_radius=12)
                txt = row_font.render(state_txt, True, (20, 20, 20))
                self.screen.blit(txt, txt.get_rect(center=badge.center))
            elif row["kind"] == "action":
                btn = pygame.Rect(panel_x + 230, y + 3, 110, 24)
                pygame.draw.rect(self.screen, (83, 111, 160), btn, border_radius=6)
                txt = row_font.render(row.get("button", "Run"), True, (245, 245, 245))
                self.screen.blit(txt, txt.get_rect(center=btn.center))

    def draw_menu(self, menu_open, selected_index, items):
        """Draw lightweight keyboard-driven config menu overlay."""
        if not menu_open:
            return

        panel_w = 520
        panel_h = 260
        panel_x = config.WINDOW_WIDTH - panel_w - 20
        panel_y = 20
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((15, 15, 15, 220))
        self.screen.blit(panel, (panel_x, panel_y))

        title = self.font.render("Config Menu (Up/Down select, Left/Right adjust)", True, (255, 255, 255))
        self.screen.blit(title, (panel_x + 12, panel_y + 10))

        row_font = pygame.font.Font(None, 28)
        for idx, item in enumerate(items):
            color = (255, 220, 120) if idx == selected_index else (220, 220, 220)
            row = row_font.render(f"{item['label']}: {item['value']:.3f}", True, color)
            self.screen.blit(row, (panel_x + 16, panel_y + 50 + idx * 30))
    
    def clear(self):
        """Clear screen."""
        self.screen.fill((0, 0, 0))
    
    def flip(self):
        """Update display."""
        pygame.display.flip()
    
    def tick(self, fps):
        """Wait for next frame."""
        self.clock.tick(fps)
