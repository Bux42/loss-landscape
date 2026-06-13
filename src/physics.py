"""Physics engine using NumPy + Numba for batch simulations."""

import numpy as np
from numba import njit, prange


@njit
def _resolve_circle_collisions(pos, vel, circle_obstacles, ball_radius, bounce_factor, iterations):
    """Resolve overlap against circle obstacles with a few solver iterations."""
    for _ in range(iterations):
        had_collision = False
        for i in range(circle_obstacles.shape[0]):
            cx = circle_obstacles[i, 0]
            cy = circle_obstacles[i, 1]
            cr = circle_obstacles[i, 2]

            dx = pos[0] - cx
            dy = pos[1] - cy
            min_dist = cr + ball_radius
            dist_sq = dx * dx + dy * dy

            if dist_sq < (min_dist * min_dist):
                had_collision = True
                dist = np.sqrt(dist_sq)
                if dist > 1e-8:
                    nx = dx / dist
                    ny = dy / dist
                else:
                    nx = 0.0
                    ny = -1.0

                penetration = min_dist - dist
                pos[0] = pos[0] + nx * (penetration + 1e-6)
                pos[1] = pos[1] + ny * (penetration + 1e-6)

                # Reflect only when velocity is heading into the surface.
                vn = vel[0] * nx + vel[1] * ny
                if vn < 0.0:
                    vel[0] = vel[0] - (1.0 + bounce_factor) * vn * nx
                    vel[1] = vel[1] - (1.0 + bounce_factor) * vn * ny

        if not had_collision:
            break

    return pos, vel


@njit
def simulate_ball_step(
    pos,
    vel,
    landscape,
    circle_obstacles,
    ball_radius,
    world_height_cells,
    friction,
    bounce_factor,
    dt,
    gravity,
    velocity_threshold,
):
    """
    Single physics step for a ball.
    
    Args:
        pos: [x, y] position
        vel: [vx, vy] velocity
        landscape: 2D height map
        friction: velocity decay factor
        bounce_factor: boundary restitution [0, 1]
        dt: time step
        gravity: gravity magnitude
        velocity_threshold: speed threshold for stopping
    
    Returns:
        (new_pos, new_vel, is_stopped)
    """
    # Work on copies so we can substep safely.
    new_pos = np.array([pos[0], pos[1]])
    new_vel = np.array([vel[0], vel[1]])

    # Adaptive sub-stepping reduces tunneling through tightly packed circles.
    speed = np.sqrt(new_vel[0] * new_vel[0] + new_vel[1] * new_vel[1])
    max_step = max(ball_radius * 0.5, 1e-4)
    substeps = int((speed * dt) / max_step) + 1
    if substeps < 1:
        substeps = 1
    elif substeps > 10:
        substeps = 10

    sub_dt = dt / substeps
    sub_friction = friction ** (1.0 / substeps)

    min_x = ball_radius
    max_x = (len(landscape[0]) - 1) - ball_radius
    min_y = ball_radius
    max_y = (world_height_cells - 1) - ball_radius

    for _ in range(substeps):
        # Apply gravity based on local landscape slope.
        x_int = int(new_pos[0])
        y_int = int(new_pos[1])
        x_int = max(0, min(len(landscape[0]) - 2, x_int))

        if y_int < len(landscape) - 1:
            y_int = max(0, min(len(landscape) - 2, y_int))
            h_right = landscape[y_int, min(x_int + 1, len(landscape[0]) - 1)]
            h_left = landscape[y_int, x_int]
            h_down = landscape[min(y_int + 1, len(landscape) - 1), x_int]
            h_up = landscape[y_int, x_int]
            slope_x = (h_right - h_left)
            slope_y = (h_down - h_up)
        else:
            slope_x = 0.0
            slope_y = 0.0

        acc_x = slope_x * gravity * sub_dt
        acc_y = slope_y * gravity * sub_dt
        acc_y += gravity * sub_dt

        new_vel[0] = (new_vel[0] + acc_x) * sub_friction
        new_vel[1] = (new_vel[1] + acc_y) * sub_friction

        new_pos[0] = new_pos[0] + new_vel[0] * sub_dt
        new_pos[1] = new_pos[1] + new_vel[1] * sub_dt

        new_pos, new_vel = _resolve_circle_collisions(
            new_pos,
            new_vel,
            circle_obstacles,
            ball_radius,
            bounce_factor,
            iterations=3,
        )

        # Radius-aware boundary check with restitution.
        if new_pos[0] < min_x:
            new_pos[0] = min_x
            if new_vel[0] < 0.0:
                new_vel[0] = -new_vel[0] * bounce_factor
        elif new_pos[0] > max_x:
            new_pos[0] = max_x
            if new_vel[0] > 0.0:
                new_vel[0] = -new_vel[0] * bounce_factor

        if new_pos[1] < min_y:
            new_pos[1] = min_y
            if new_vel[1] < 0.0:
                new_vel[1] = -new_vel[1] * bounce_factor
        elif new_pos[1] > max_y:
            new_pos[1] = max_y
            if new_vel[1] > 0.0:
                new_vel[1] = -new_vel[1] * bounce_factor
    
    # Zero out tiny jitter velocities so near-rest contacts can settle.
    if np.abs(new_vel[0]) < (velocity_threshold * 0.5):
        new_vel[0] = 0.0
    if np.abs(new_vel[1]) < (velocity_threshold * 0.5):
        new_vel[1] = 0.0

    # Check stopping condition.
    speed = np.sqrt(new_vel[0]**2 + new_vel[1]**2)
    is_stopped = speed < velocity_threshold

    # Also allow settling when almost motionless while touching boundaries.
    if not is_stopped:
        near_left = np.abs(new_pos[0] - min_x) < 1e-3
        near_right = np.abs(new_pos[0] - max_x) < 1e-3
        near_top = np.abs(new_pos[1] - min_y) < 1e-3
        near_bottom = np.abs(new_pos[1] - max_y) < 1e-3
        near_boundary = near_left or near_right or near_top or near_bottom

        near_rest = (
            np.abs(new_vel[0]) < (velocity_threshold * 3.0)
            and np.abs(new_vel[1]) < (velocity_threshold * 3.0)
        )
        if near_boundary and near_rest:
            is_stopped = True
    
    return new_pos, new_vel, is_stopped


def simulate_ball(
    start_pos,
    target_pos,
    landscape,
    circle_obstacles,
    ball_radius,
    world_height_cells,
    friction,
    bounce_factor,
    gravity,
    velocity_threshold,
    dt,
    max_steps=1000,
):
    """
    Simulate ball until it stops.
    
    Returns:
        (path, time_steps, distance_to_target, time_score)
    """
    pos = np.array(start_pos, dtype=np.float64)
    vel = np.array([0.0, 0.0], dtype=np.float64)
    path = [pos.copy()]
    
    for step in range(max_steps):
        pos, vel, is_stopped = simulate_ball_step(
            pos,
            vel,
            landscape,
            circle_obstacles,
            ball_radius,
            world_height_cells,
            friction,
            bounce_factor,
            dt=dt,
            gravity=gravity,
            velocity_threshold=velocity_threshold,
        )
        path.append(pos.copy())
        
        if is_stopped:
            break
    
    # Compute scores
    final_dist = np.linalg.norm(pos - target_pos)
    time_score = len(path)  # Number of steps
    
    return np.array(path), time_score, final_dist


@njit
def _simulate_ball_scores(
    start_pos,
    target_pos,
    landscape,
    circle_obstacles,
    ball_radius,
    world_height_cells,
    friction,
    bounce_factor,
    gravity,
    velocity_threshold,
    dt,
    max_steps,
):
    """Numba-compiled distance-only simulation for one start position."""
    pos = np.array([start_pos[0], start_pos[1]])
    vel = np.array([0.0, 0.0])

    for step in range(max_steps):
        pos, vel, is_stopped = simulate_ball_step(
            pos,
            vel,
            landscape,
            circle_obstacles,
            ball_radius,
            world_height_cells,
            friction,
            bounce_factor,
            dt=dt,
            gravity=gravity,
            velocity_threshold=velocity_threshold,
        )
        if is_stopped:
            break

    dx = pos[0] - target_pos[0]
    dy = pos[1] - target_pos[1]
    final_dist = np.sqrt(dx * dx + dy * dy)
    return final_dist


@njit(parallel=True)
def _batch_simulate_scores(
    start_positions,
    target_pos,
    landscape,
    circle_obstacles,
    ball_radius,
    world_height_cells,
    friction,
    bounce_factor,
    gravity,
    velocity_threshold,
    dt,
    max_steps,
):
    """Parallel Numba kernel computing distance scores for all start positions."""
    N = start_positions.shape[0]
    distances = np.empty(N, dtype=np.float64)

    for i in prange(N):
        dist = _simulate_ball_scores(
            start_positions[i],
            target_pos,
            landscape,
            circle_obstacles,
            ball_radius,
            world_height_cells,
            friction,
            bounce_factor,
            gravity,
            velocity_threshold,
            dt,
            max_steps,
        )
        distances[i] = dist

    return distances


def batch_simulate(
    start_positions,
    target_pos,
    landscape,
    circle_obstacles,
    ball_radius,
    world_height_cells,
    friction,
    bounce_factor,
    gravity,
    velocity_threshold,
    dt,
    max_steps=1000,
):
    """
    Simulate multiple balls in parallel.
    
    Args:
        start_positions: (N, 2) array of starting positions
        target_pos: target position
        landscape: height map
        friction: friction coefficient
        bounce_factor: boundary restitution [0, 1]
        gravity: gravity magnitude
        velocity_threshold: speed threshold for stopping
        dt: simulation time step
        max_steps: max simulation steps
    
    Returns:
        distances - array of final distance scores
    """
    starts = np.ascontiguousarray(start_positions, dtype=np.float64)
    target = np.ascontiguousarray(target_pos, dtype=np.float64)
    land = np.ascontiguousarray(landscape, dtype=np.float64)
    circles = np.ascontiguousarray(circle_obstacles, dtype=np.float64)

    return _batch_simulate_scores(
        starts,
        target,
        land,
        circles,
        ball_radius,
        world_height_cells,
        friction,
        bounce_factor,
        gravity,
        velocity_threshold,
        dt,
        max_steps,
    )
