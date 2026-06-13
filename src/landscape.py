"""Procedural loss landscape generation."""

import numpy as np


def generate_landscape(width, height, seed=42):
    """
    Generate procedural landscape using Perlin-like noise.
    
    Args:
        width: Landscape width in cells
        height: Landscape height in cells
        seed: Random seed for reproducibility
    
    Returns:
        (width, height) float array with height values
    """
    np.random.seed(seed)
    
    # Simple approach: smooth random noise
    landscape = np.random.randn(height, width) * 0.3
    
    # Apply Gaussian blur-like smoothing
    from scipy.ndimage import gaussian_filter
    landscape = gaussian_filter(landscape, sigma=5.0)
    
    # Normalize to [0, 1]
    landscape = (landscape - landscape.min()) / (landscape.max() - landscape.min())
    
    return landscape
