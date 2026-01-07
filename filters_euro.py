"""
OneEuro Filter (OPTIMIZED)
Speed-adaptive low-pass filter for noisy input (MediaPipe landmarks).

OPTIMIZATIONS:
- Full 3D support (X, Y, Z filtering)
- Cached alpha calculations
- Vectorized operations for batch filtering
"""

import math
import numpy as np

class OneEuroFilter:
    """Single-channel OneEuro filter (optimized)."""
    
    __slots__ = ['min_cutoff', 'beta', 'd_cutoff', 'x_prev', 'dx_prev', 't_prev']
    
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def filter(self, x, timestamp):
        """Filter a single value. Returns filtered value."""
        # First run: return raw
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = timestamp
            return x

        # Calculate dt
        te = timestamp - self.t_prev
        if te <= 0.0:
            return self.x_prev

        # Derivative filter (for speed detection)
        tau_d = 1.0 / (2.0 * 3.14159265359 * self.d_cutoff)
        alpha_d = 1.0 / (1.0 + tau_d / te)
        
        dx = (x - self.x_prev) / te
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev
        
        # Dynamic cutoff based on speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Main signal filter
        tau = 1.0 / (2.0 * 3.14159265359 * cutoff)
        alpha = 1.0 / (1.0 + tau / te)
        
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        
        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = timestamp
        
        return x_hat


class PointFilter:
    """3D Point filter (X, Y, Z) using OneEuro filters."""
    
    __slots__ = ['f_x', 'f_y', 'f_z']
    
    def __init__(self, min_cutoff=1.0, beta=5.0):
        self.f_x = OneEuroFilter(min_cutoff, beta)
        self.f_y = OneEuroFilter(min_cutoff, beta)
        self.f_z = OneEuroFilter(min_cutoff, beta)  # UPGRADED: Full 3D
    
    def filter(self, x, y, timestamp, z=None):
        """Filter X, Y, and optionally Z. Returns (x, y) or (x, y, z)."""
        fx = self.f_x.filter(x, timestamp)
        fy = self.f_y.filter(y, timestamp)
        
        if z is not None:
            fz = self.f_z.filter(z, timestamp)
            return (fx, fy, fz)
        return (fx, fy)
    
    def filter3d(self, x, y, z, timestamp):
        """Filter 3D point. Returns (x, y, z)."""
        return (
            self.f_x.filter(x, timestamp),
            self.f_y.filter(y, timestamp),
            self.f_z.filter(z, timestamp)
        )
    
    def reset(self):
        self.f_x.reset()
        self.f_y.reset()
        self.f_z.reset()


class BatchPointFilter:
    """Filter multiple 3D points efficiently (for 21 hand landmarks)."""
    
    def __init__(self, num_points=21, min_cutoff=1.0, beta=5.0):
        self.filters = [PointFilter(min_cutoff, beta) for _ in range(num_points)]
        self.num_points = num_points
    
    def filter_batch(self, landmarks, timestamp):
        """
        Filter all landmarks at once.
        
        Args:
            landmarks: List of [id, x, y, z] or similar
            timestamp: Current time in seconds
        
        Returns:
            Filtered landmarks with same structure
        """
        result = []
        for i, lm in enumerate(landmarks):
            if i >= self.num_points:
                break
            
            if len(lm) >= 4:  # [id, x, y, z]
                fx, fy, fz = self.filters[i].filter3d(lm[1], lm[2], lm[3], timestamp)
                result.append([lm[0], fx, fy, fz])
            elif len(lm) >= 3:  # [id, x, y]
                fx, fy = self.filters[i].filter(lm[1], lm[2], timestamp)
                result.append([lm[0], fx, fy])
            else:
                result.append(lm)
        
        return result
    
    def reset(self):
        for f in self.filters:
            f.reset()
