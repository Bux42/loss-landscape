"""Main entry point."""

import argparse
import pygame
import numpy as np
from scipy.ndimage import zoom
from src import config
from src.landscape import generate_landscape
from src.compute_worker import ComputeWorker
from src.renderer import Renderer
from src.physics import simulate_ball_step


def _to_world_from_top_panel(x, y):
    """Convert top-panel screen coordinates to landscape coordinates."""
    y_local = y - config.MENU_BAR_HEIGHT
    wx = x / (config.WINDOW_WIDTH - 1) * (config.LANDSCAPE_WIDTH - 1)
    wy = y_local / (config.TOP_PANEL_HEIGHT - 1) * (config.LANDSCAPE_HEIGHT - 1)
    return np.array([
        wx,
        wy,
    ], dtype=np.float64)


def _to_world_from_bottom_panel(x, y):
    """Convert bottom-panel screen coordinates to landscape coordinates."""
    y_local = y - (config.MENU_BAR_HEIGHT + config.TOP_PANEL_HEIGHT)
    wx = x / (config.WINDOW_WIDTH - 1) * (config.LANDSCAPE_WIDTH - 1)
    wy = y_local / (config.BOTTOM_PANEL_HEIGHT - 1) * (config.LANDSCAPE_HEIGHT - 1)
    return np.array([
        wx,
        config.LANDSCAPE_HEIGHT + wy,
    ], dtype=np.float64)


def _obstacles_to_world_circles(obstacles):
    """Convert screen-space bottom obstacles to world-space circles."""
    circles = []
    for obs in obstacles:
        if obs.get("type") == "circle":
            world_x = obs["x"] / (config.WINDOW_WIDTH - 1) * (config.LANDSCAPE_WIDTH - 1)
            world_bottom_y = obs["y"] / (config.BOTTOM_PANEL_HEIGHT - 1) * (config.LANDSCAPE_HEIGHT - 1)
            world_y = config.LANDSCAPE_HEIGHT + world_bottom_y
            radius_world = obs["radius"] * (config.LANDSCAPE_WIDTH - 1) / (config.WINDOW_WIDTH - 1)
            circles.append([world_x, world_y, radius_world])
    if circles:
        return np.array(circles, dtype=np.float64)
    return np.empty((0, 3), dtype=np.float64)


def _upscale_score_grid(values, grid_size, target_h, target_w):
    """Map coarse batch scores to full landscape resolution."""
    grid_h, grid_w = grid_size

    if grid_h == target_h and grid_w == target_w:
        return values.reshape(target_h, target_w)

    # Bilinear interpolation produces smoother transitions than nearest-neighbor.
    grid = values.reshape(grid_h, grid_w)
    zoom_y = target_h / grid_h
    zoom_x = target_w / grid_w
    upscaled = zoom(grid, (zoom_y, zoom_x), order=1)

    # Guard shape edge-cases from floating-point rounding in zoom.
    return upscaled[:target_h, :target_w]


def _build_start_grid(grid_w, grid_h):
    """Build world-space start positions with configurable sample resolution."""
    xs = np.linspace(0, config.LANDSCAPE_WIDTH - 1, grid_w)
    ys = np.linspace(0, config.LANDSCAPE_HEIGHT - 1, grid_h)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _snap_to_score_grid(world_pos, score_grid):
    """Snap a world position to the nearest active score-grid start point."""
    grid_w, grid_h = score_grid
    wx = float(world_pos[0])
    wy = float(world_pos[1])

    if grid_w > 1:
        ix = int(round(wx * (grid_w - 1) / (config.LANDSCAPE_WIDTH - 1)))
        ix = max(0, min(grid_w - 1, ix))
        wx = ix * (config.LANDSCAPE_WIDTH - 1) / (grid_w - 1)

    if grid_h > 1:
        iy = int(round(wy * (grid_h - 1) / (config.LANDSCAPE_HEIGHT - 1)))
        iy = max(0, min(grid_h - 1, iy))
        wy = iy * (config.LANDSCAPE_HEIGHT - 1) / (grid_h - 1)

    return np.array([wx, wy], dtype=np.float64)


def _submit_recompute(worker, score_grid, target_pos, current_task_id, compute_enabled):
    """Submit a full recompute for the current target and score grid when enabled."""
    if (not compute_enabled) or (target_pos is None):
        return current_task_id, -1, False
    starts = _build_start_grid(score_grid[0], score_grid[1])
    worker.submit_task(starts, target_pos, current_task_id)
    return current_task_id + 1, current_task_id, True


def _parse_args():
    """Parse CLI arguments for fixed window dimensions."""
    parser = argparse.ArgumentParser(description="Loss landscape visualizer")
    parser.add_argument("--width", type=int, default=200, help="Window width in pixels")
    parser.add_argument("--height", type=int, default=200, help="Window height in pixels")
    return parser.parse_args()


def _build_menu_rows(
    active_section,
    friction,
    bounce_factor,
    gravity,
    velocity_threshold,
    max_steps,
    full_compute_enabled,
    full_pixel_scoring,
    strict_visual_match,
    target_edit_mode,
):
    """Build visible menu rows for the active section."""
    if active_section == "Physics":
        return [
            {"id": "gravity", "label": "Gravity", "kind": "number", "value": gravity, "step": 0.5, "min": 1.0, "max": 100.0},
            {"id": "friction", "label": "Friction", "kind": "number", "value": friction, "step": 0.005, "min": 0.80, "max": 0.999},
            {"id": "bounce", "label": "Bounce", "kind": "number", "value": bounce_factor, "step": 0.05, "min": 0.0, "max": 1.0},
            {"id": "threshold", "label": "Stop Threshold", "kind": "number", "value": velocity_threshold, "step": 0.002, "min": 0.001, "max": 0.1},
        ]
    if active_section == "Scoring":
        return [
            {"id": "full_compute", "label": "Full Landscape Compute", "kind": "toggle", "value": full_compute_enabled},
            {"id": "full_pixel", "label": "Full Pixel Scoring", "kind": "toggle", "value": full_pixel_scoring},
            {"id": "strict_match", "label": "Strict Drop/Map Match", "kind": "toggle", "value": strict_visual_match},
        ]
    if active_section == "Compute":
        return [
            {"id": "max_steps", "label": "Max Steps", "kind": "number", "value": float(max_steps), "step": 100.0, "min": 100.0, "max": 5000.0},
            {"id": "recompute", "label": "Recompute Landscape", "kind": "action", "value": 0.0},
            {"id": "cancel_compute", "label": "Cancel Compute", "kind": "action", "value": 0.0, "button": "Cancel"},
        ]
    return [
        {"id": "target_mode", "label": "Target Edit Mode", "kind": "toggle", "value": target_edit_mode},
        {"id": "clear_obstacles", "label": "Clear Obstacles", "kind": "action", "value": 0.0},
        {"id": "clear_balls", "label": "Clear Balls", "kind": "action", "value": 0.0},
    ]


def _apply_dimensions(width, height):
    """Apply runtime fixed window dimensions.

    Landscape resolution remains independent (from config defaults), while
    top/bottom panel sizes are derived from window height.
    """
    width = max(16, int(width))
    height = max(16, int(height))

    config.WINDOW_WIDTH = width
    config.WINDOW_HEIGHT = height + config.MENU_BAR_HEIGHT
    config.TOP_PANEL_HEIGHT = max(1, height // 2)
    config.BOTTOM_PANEL_HEIGHT = max(1, height - config.TOP_PANEL_HEIGHT)

    # World height is tied to landscape resolution, not window size.
    config.WORLD_HEIGHT_CELLS = config.LANDSCAPE_HEIGHT * 2


def main():
    """Main application loop."""
    args = _parse_args()
    _apply_dimensions(args.width, args.height)

    pygame.init()
    
    # Generate landscape
    landscape = generate_landscape(config.LANDSCAPE_WIDTH, config.LANDSCAPE_HEIGHT)
    
    # Initialize obstacles
    obstacles = []
    
    # Initialize renderer
    renderer = Renderer()
    
    # Initialize compute worker
    friction = config.FRICTION_DEFAULT
    bounce_factor = config.BOUNCE_FACTOR_DEFAULT
    gravity = config.GRAVITY
    velocity_threshold = config.VELOCITY_THRESHOLD
    dt = config.DT
    max_steps = config.MAX_STEPS

    worker = ComputeWorker(
        landscape,
        config.WORLD_HEIGHT_CELLS,
        obstacles,
        friction=friction,
        bounce_factor=bounce_factor,
        gravity=gravity,
        velocity_threshold=velocity_threshold,
        dt=dt,
        max_steps=max_steps,
    )
    worker.start()
    
    # Application state
    target_pos = None
    paused = False
    scores_distance = None
    current_task_id = 0
    sample_grid = (min(100, config.LANDSCAPE_WIDTH), min(100, config.LANDSCAPE_HEIGHT))
    full_compute_enabled = True
    full_pixel_scoring = False
    strict_visual_match = True
    score_grid = sample_grid
    manual_scores_distance = np.full(
        (config.LANDSCAPE_HEIGHT, config.LANDSCAPE_WIDTH),
        np.nan,
        dtype=np.float64,
    )
    target_edit_mode = False
    target_dragging = False
    live_balls = []
    compute_running = False
    compute_progress = 0.0
    latest_task_id = -1

    menu_open = False
    menu_sections = ["Physics", "Scoring", "Compute", "Editor"]
    active_section_index = 0
    
    # Main loop
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()

                # Top bar: section tabs.
                if y < config.MENU_BAR_HEIGHT:
                    tab_clicked = False
                    for idx, name in enumerate(menu_sections):
                        tab_x = config.MENU_LEFT_PAD + idx * (config.MENU_TAB_WIDTH + config.MENU_TAB_GAP)
                        tab_rect = pygame.Rect(tab_x, 6, config.MENU_TAB_WIDTH, config.MENU_BAR_HEIGHT - 12)
                        if tab_rect.collidepoint(x, y):
                            tab_clicked = True
                            if menu_open and idx == active_section_index:
                                menu_open = False
                            else:
                                active_section_index = idx
                                menu_open = True
                            break
                    if not tab_clicked:
                        menu_open = False
                    continue

                # Menu panel controls (mouse-driven).
                rows_for_menu = _build_menu_rows(
                    menu_sections[active_section_index],
                    friction,
                    bounce_factor,
                    gravity,
                    velocity_threshold,
                    max_steps,
                    full_compute_enabled,
                    full_pixel_scoring,
                    strict_visual_match,
                    target_edit_mode,
                )
                panel_x = config.MENU_PANEL_PAD
                panel_y = config.MENU_BAR_HEIGHT + config.MENU_PANEL_PAD
                panel_h = config.MENU_PANEL_PAD * 2 + len(rows_for_menu) * config.MENU_ROW_HEIGHT
                panel_rect = pygame.Rect(panel_x, panel_y, config.MENU_PANEL_WIDTH, panel_h)

                # Click-away collapse behavior for menu panel.
                if menu_open and not panel_rect.collidepoint(x, y):
                    menu_open = False

                if menu_open and panel_rect.collidepoint(x, y):
                    row_idx = (y - (panel_y + config.MENU_PANEL_PAD)) // config.MENU_ROW_HEIGHT
                    if 0 <= row_idx < len(rows_for_menu):
                        row = rows_for_menu[row_idx]
                        row_y = panel_y + config.MENU_PANEL_PAD + row_idx * config.MENU_ROW_HEIGHT
                        if row["kind"] == "number":
                            minus_rect = pygame.Rect(panel_x + 275, row_y + 3, 28, 24)
                            plus_rect = pygame.Rect(panel_x + 312, row_y + 3, 28, 24)
                            delta = 0.0
                            if minus_rect.collidepoint(x, y):
                                delta = -row["step"]
                            elif plus_rect.collidepoint(x, y):
                                delta = row["step"]

                            if delta != 0.0:
                                if row["id"] == "gravity":
                                    gravity = float(np.clip(gravity + delta, row["min"], row["max"]))
                                elif row["id"] == "friction":
                                    friction = float(np.clip(friction + delta, row["min"], row["max"]))
                                elif row["id"] == "bounce":
                                    bounce_factor = float(np.clip(bounce_factor + delta, row["min"], row["max"]))
                                elif row["id"] == "threshold":
                                    velocity_threshold = float(np.clip(velocity_threshold + delta, row["min"], row["max"]))
                                elif row["id"] == "max_steps":
                                    max_steps = int(np.clip(max_steps + delta, row["min"], row["max"]))

                                worker.set_simulation_params(
                                    friction=friction,
                                    bounce_factor=bounce_factor,
                                    gravity=gravity,
                                    velocity_threshold=velocity_threshold,
                                    max_steps=max_steps,
                                )
                                current_task_id, latest_task_id, started = _submit_recompute(
                                        worker,
                                        score_grid,
                                        target_pos,
                                        current_task_id,
                                        full_compute_enabled,
                                    )
                                if started:
                                    compute_running = True
                                    compute_progress = 0.0

                        elif row["kind"] == "toggle":
                            if row["id"] == "full_compute":
                                full_compute_enabled = not full_compute_enabled
                                if full_compute_enabled:
                                    current_task_id, latest_task_id, started = _submit_recompute(
                                        worker,
                                        score_grid,
                                        target_pos,
                                        current_task_id,
                                        full_compute_enabled,
                                    )
                                    if started:
                                        compute_running = True
                                        compute_progress = 0.0
                                else:
                                    worker.cancel_current_task(clear_pending=True)
                                    compute_running = False
                                    compute_progress = 0.0
                                    latest_task_id = -1
                                    scores_distance = manual_scores_distance
                            elif row["id"] == "full_pixel":
                                full_pixel_scoring = not full_pixel_scoring
                                score_grid = (
                                    (config.LANDSCAPE_WIDTH, config.LANDSCAPE_HEIGHT)
                                    if full_pixel_scoring
                                    else sample_grid
                                )
                                current_task_id, latest_task_id, started = _submit_recompute(
                                        worker,
                                        score_grid,
                                        target_pos,
                                        current_task_id,
                                        full_compute_enabled,
                                    )
                                if started:
                                    compute_running = True
                                    compute_progress = 0.0
                            elif row["id"] == "target_mode":
                                target_edit_mode = not target_edit_mode
                            elif row["id"] == "strict_match":
                                strict_visual_match = not strict_visual_match

                        elif row["kind"] == "action":
                            if row["id"] == "recompute" and target_pos is not None:
                                current_task_id, latest_task_id, started = _submit_recompute(
                                    worker,
                                    score_grid,
                                    target_pos,
                                    current_task_id,
                                    full_compute_enabled,
                                )
                                if started:
                                    compute_running = True
                                    compute_progress = 0.0
                            elif row["id"] == "cancel_compute":
                                worker.cancel_current_task(clear_pending=True)
                                compute_running = False
                                compute_progress = 0.0
                                latest_task_id = -1
                            elif row["id"] == "clear_obstacles":
                                obstacles.clear()
                            elif row["id"] == "clear_balls":
                                live_balls.clear()
                    continue
                
                if y < config.MENU_BAR_HEIGHT + config.TOP_PANEL_HEIGHT:
                    # Click on landscape - drop one more ball in real time
                    new_pos = _to_world_from_top_panel(x, y)
                    if strict_visual_match:
                        new_pos = _snap_to_score_grid(new_pos, score_grid)
                    launch_color = renderer.sample_landscape_color(
                        new_pos,
                        landscape,
                        scores_distance,
                        strict_match=strict_visual_match,
                    )
                    live_balls.append(
                        {
                            "pos": new_pos,
                            "vel": np.array([0.0, 0.0], dtype=np.float64),
                            "trail": [new_pos.copy()],
                            "settled": False,
                            "settled_frames": 0,
                            "steps": 0,
                            "launch_ix": int(np.clip(round(float(new_pos[0])), 0, config.LANDSCAPE_WIDTH - 1)),
                            "launch_iy": int(np.clip(round(float(new_pos[1])), 0, config.LANDSCAPE_HEIGHT - 1)),
                            "color": launch_color,
                        }
                    )
                    print(f"Ball dropped at {new_pos} | active balls: {len(live_balls)}")
                
                else:
                    if target_edit_mode:
                        # In target mode, bottom click places target.
                        target_pos = _to_world_from_bottom_panel(x, y)
                        target_pos[0] = np.clip(target_pos[0], 0, config.LANDSCAPE_WIDTH - 1)
                        target_pos[1] = np.clip(target_pos[1], config.LANDSCAPE_HEIGHT, config.WORLD_HEIGHT_CELLS - 1)
                        target_dragging = True
                        print(f"Target set/moved to {target_pos}")

                        # Recompute map for new target
                        current_task_id, latest_task_id, started = _submit_recompute(
                            worker,
                            score_grid,
                            target_pos,
                            current_task_id,
                            full_compute_enabled,
                        )
                        if started:
                            compute_running = True
                            compute_progress = 0.0
                    else:
                        # Click on obstacles panel - add obstacle
                        x_obs = x
                        y_obs = y - (config.MENU_BAR_HEIGHT + config.TOP_PANEL_HEIGHT)
                        obstacles.append({
                            'type': 'circle',
                            'x': x_obs,
                            'y': y_obs,
                            'radius': 20
                        })
                        print(f"Obstacle added at ({x_obs}, {y_obs})")
                        if target_pos is not None:
                            current_task_id, latest_task_id, started = _submit_recompute(
                                worker,
                                score_grid,
                                target_pos,
                                current_task_id,
                                full_compute_enabled,
                            )
                            if started:
                                compute_running = True
                                compute_progress = 0.0

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    if target_edit_mode and target_dragging and target_pos is not None:
                        current_task_id, latest_task_id, started = _submit_recompute(
                            worker,
                            score_grid,
                            target_pos,
                            current_task_id,
                            full_compute_enabled,
                        )
                        if started:
                            compute_running = True
                            compute_progress = 0.0
                    target_dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if target_edit_mode and target_dragging:
                    x, y = pygame.mouse.get_pos()
                    if y >= config.MENU_BAR_HEIGHT + config.TOP_PANEL_HEIGHT:
                        target_pos = _to_world_from_bottom_panel(x, y)
                        target_pos[0] = np.clip(target_pos[0], 0, config.LANDSCAPE_WIDTH - 1)
                        target_pos[1] = np.clip(target_pos[1], config.LANDSCAPE_HEIGHT, config.WORLD_HEIGHT_CELLS - 1)
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Toggle pause
                    paused = not paused
                    if paused:
                        worker.pause()
                    else:
                        worker.resume()
                    print(f"Paused: {paused}")
                
                elif event.key == pygame.K_c:
                    # Clear obstacles
                    obstacles.clear()
                    print("Obstacles cleared")
                    if target_pos is not None:
                        current_task_id, latest_task_id, started = _submit_recompute(
                            worker,
                            score_grid,
                            target_pos,
                            current_task_id,
                            full_compute_enabled,
                        )
                        if started:
                            compute_running = True
                            compute_progress = 0.0

                elif event.key == pygame.K_COMMA:
                    # Decrease bounce factor
                    bounce_factor = max(
                        config.BOUNCE_FACTOR_MIN,
                        bounce_factor - config.BOUNCE_FACTOR_STEP,
                    )
                    worker.set_simulation_params(bounce_factor=bounce_factor)
                    print(f"Bounce factor: {bounce_factor:.2f}")
                    if target_pos is not None:
                        current_task_id, latest_task_id, started = _submit_recompute(
                            worker,
                            score_grid,
                            target_pos,
                            current_task_id,
                            full_compute_enabled,
                        )
                        if started:
                            compute_running = True
                            compute_progress = 0.0

                elif event.key == pygame.K_PERIOD:
                    # Increase bounce factor
                    bounce_factor = min(
                        config.BOUNCE_FACTOR_MAX,
                        bounce_factor + config.BOUNCE_FACTOR_STEP,
                    )
                    worker.set_simulation_params(bounce_factor=bounce_factor)
                    print(f"Bounce factor: {bounce_factor:.2f}")
                    if target_pos is not None:
                        current_task_id, latest_task_id, started = _submit_recompute(
                            worker,
                            score_grid,
                            target_pos,
                            current_task_id,
                            full_compute_enabled,
                        )
                        if started:
                            compute_running = True
                            compute_progress = 0.0

                elif event.key == pygame.K_t:
                    target_edit_mode = not target_edit_mode
                    print(f"Target edit mode: {target_edit_mode}")

                elif event.key == pygame.K_m:
                    menu_open = not menu_open

                elif event.key == pygame.K_p:
                    full_pixel_scoring = not full_pixel_scoring
                    score_grid = (
                        (config.LANDSCAPE_WIDTH, config.LANDSCAPE_HEIGHT)
                        if full_pixel_scoring
                        else sample_grid
                    )
                    print(
                        "Score mode:",
                        "FULL PIXEL" if full_pixel_scoring else "SAMPLED",
                        f"({score_grid[0]}x{score_grid[1]})",
                    )
                    if target_pos is not None:
                        current_task_id, latest_task_id, started = _submit_recompute(
                            worker,
                            score_grid,
                            target_pos,
                            current_task_id,
                            full_compute_enabled,
                        )
                        if started:
                            compute_running = True
                            compute_progress = 0.0

                elif event.key == pygame.K_k:
                    worker.cancel_current_task(clear_pending=True)
                    compute_running = False
                    compute_progress = 0.0
                    latest_task_id = -1
                    print("Compute cancelled")

                # Sectioned menu is mouse-driven.
        
        # Check for results
        while True:
            progress = worker.get_progress(block=False)
            if progress is None:
                break
            task_id, done_count, total_count = progress
            if task_id == latest_task_id and total_count > 0:
                compute_progress = done_count / total_count
                compute_running = compute_progress < 1.0

        result = worker.get_result(block=False)
        if result is not None:
            if len(result) == 3:
                task_id, distances, cancelled = result
            elif len(result) == 4:
                task_id, distances, _legacy_times, cancelled = result
            else:
                task_id, distances = result
                cancelled = False

            if cancelled:
                if task_id == latest_task_id:
                    compute_running = False
                    compute_progress = 0.0
                print(f"Task {task_id} cancelled")
            elif task_id == latest_task_id:
                scores_distance = _upscale_score_grid(
                    distances,
                    score_grid,
                    config.LANDSCAPE_HEIGHT,
                    config.LANDSCAPE_WIDTH,
                )
                compute_progress = 1.0
                compute_running = False
                print(f"Task {task_id} complete: {len(distances)} simulations")

        # Advance one real-time simulation step for active ball
        if live_balls:
            circle_obstacles = _obstacles_to_world_circles(obstacles)
            ball_radius = config.BALL_RADIUS_SCREEN * (config.LANDSCAPE_WIDTH - 1) / (config.WINDOW_WIDTH - 1)

            for ball in live_balls:
                if ball["settled"]:
                    ball["settled_frames"] = ball.get("settled_frames", 0) + 1
                    if ball["settled_frames"] >= config.SETTLED_BALL_LINGER_FRAMES and len(ball["trail"]) > 1:
                        ball["trail"] = [ball["pos"].copy()]
                    continue

                for _ in range(config.LIVE_SIM_STEPS_PER_FRAME):
                    new_pos, new_vel, is_stopped = simulate_ball_step(
                        ball["pos"],
                        ball["vel"],
                        landscape,
                        circle_obstacles,
                        ball_radius,
                        config.WORLD_HEIGHT_CELLS,
                        friction,
                        bounce_factor,
                        dt,
                        gravity,
                        velocity_threshold,
                    )
                    ball["pos"] = new_pos
                    ball["vel"] = new_vel
                    ball["steps"] = ball.get("steps", 0) + 1
                    ball["trail"].append(new_pos.copy())
                    if len(ball["trail"]) > 240:
                        ball["trail"].pop(0)
                    if is_stopped or ball["steps"] >= max_steps:
                        ball["settled"] = True
                        ball["settled_frames"] = 0
                        if target_pos is not None:
                            dx = ball["pos"][0] - target_pos[0]
                            dy = ball["pos"][1] - target_pos[1]
                            dist = float(np.sqrt(dx * dx + dy * dy))
                            ix = ball.get("launch_ix", int(np.clip(round(float(ball["pos"][0])), 0, config.LANDSCAPE_WIDTH - 1)))
                            iy = ball.get("launch_iy", int(np.clip(round(float(ball["pos"][1])), 0, config.LANDSCAPE_HEIGHT - 1)))
                            manual_scores_distance[iy, ix] = dist
                            if not full_compute_enabled:
                                scores_distance = manual_scores_distance
                        break
        
        # Render
        renderer.clear()
        renderer.draw_landscape(
            landscape,
            scores_distance,
            strict_match=strict_visual_match,
        )
        renderer.draw_obstacles(obstacles)
        if target_pos is not None:
            renderer.draw_target(target_pos)
        renderer.draw_live_balls(live_balls)
        menu_rows = _build_menu_rows(
            menu_sections[active_section_index],
            friction,
            bounce_factor,
            gravity,
            velocity_threshold,
            max_steps,
            full_compute_enabled,
            full_pixel_scoring,
            strict_visual_match,
            target_edit_mode,
        )

        renderer.draw_top_menu(
            menu_open,
            menu_sections,
            active_section_index,
            menu_rows,
            compute_running,
            compute_progress,
        )

        renderer.draw_hud(
            bounce_factor,
            paused,
            target_edit_mode,
            menu_open,
            full_pixel_scoring,
            renderer.clock.get_fps(),
        )
        renderer.flip()
        
        renderer.tick(config.FPS)
    
    worker.stop()
    pygame.quit()


if __name__ == '__main__':
    main()
