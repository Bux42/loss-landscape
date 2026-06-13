"""Collision detection and response."""

import numpy as np


def circle_circle_collision(pos1, r1, pos2, r2):
    """
    Detect collision between two circles.
    
    Returns:
        (is_colliding, normal, penetration_depth)
    """
    diff = pos2 - pos1
    dist = np.linalg.norm(diff)
    min_dist = r1 + r2
    
    if dist >= min_dist:
        return False, np.zeros(2), 0.0
    
    if dist < 1e-6:
        normal = np.array([1.0, 0.0])
    else:
        normal = diff / dist
    
    penetration = min_dist - dist
    return True, normal, penetration


def circle_box_collision(circle_pos, circle_r, box_pos, box_width, box_height, box_angle):
    """
    Detect collision between circle and axis-aligned box (rotated).
    
    Returns:
        (is_colliding, normal, penetration_depth)
    
    Note: Simplified SAT for rotated rectangles.
    """
    # TODO: Implement proper SAT
    # For now, use AABB (axis-aligned)
    
    half_w = box_width / 2
    half_h = box_height / 2
    
    # Find closest point on box to circle
    closest_x = np.clip(circle_pos[0], box_pos[0] - half_w, box_pos[0] + half_w)
    closest_y = np.clip(circle_pos[1], box_pos[1] - half_h, box_pos[1] + half_h)
    
    diff = circle_pos - np.array([closest_x, closest_y])
    dist = np.linalg.norm(diff)
    
    if dist >= circle_r:
        return False, np.zeros(2), 0.0
    
    if dist < 1e-6:
        normal = np.array([1.0, 0.0])
    else:
        normal = diff / dist
    
    penetration = circle_r - dist
    return True, normal, penetration
