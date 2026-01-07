"""
GestureForge Configuration
Production-ready settings for the application.
"""

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================
WINDOW_WIDTH = 0  # 0 = use camera resolution
WINDOW_HEIGHT = 0  # 0 = use camera resolution
FPS_CAP = 60  # Target frame rate

# ============================================================================
# HAND TRACKING SETTINGS
# ============================================================================
NUM_HANDS = 2  # Maximum hands to track
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
FILTER_MIN_CUTOFF = 1.0  # OneEuro filter smoothness
FILTER_BETA = 10.0  # OneEuro filter responsiveness

# ============================================================================
# VOXEL DRAWING SETTINGS
# ============================================================================
GRID_SIZE = 0.1  # Voxel grid resolution
DRAW_THRESHOLD = 50  # Pinch distance threshold (pixels)
STROKE_INTERPOLATION = True  # Fill gaps between points

# ============================================================================
# AI 3D GENESIS SETTINGS
# ============================================================================
GENESIS_PARTICLES = 2500  # Number of particles (eminent scale)
GENESIS_MORPH_SPEED = 0.05  # Particle lerp speed
GENESIS_FRICTION = 0.98  # Rotation inertia decay
GENESIS_SENSITIVITY = 3.0  # Rotation sensitivity
GENESIS_AUTO_SPIN_SPEED = 0.3  # Degrees per frame when idle
GENESIS_MAX_SCALE = 2.0  # Maximum two-hand scale factor

# ============================================================================
# COLORS (R, G, B) normalized to 0-1
# ============================================================================
COLORS = [
    (0, 1, 1),      # Cyan
    (1, 0, 1),      # Magenta
    (0, 1, 0),      # Green
    (1, 0.5, 0),    # Orange
    (1, 1, 1),      # White
]

# ============================================================================
# EXPORT SETTINGS
# ============================================================================
EXPORTS_DIR = "exports"
RECORDINGS_DIR = "recordings"
WORLD_FILE = "world.json"

# ============================================================================
# DEBUG SETTINGS (set to False for production)
# ============================================================================
DEBUG_MODE = False
SHOW_FPS = True
SHOW_SKELETON = True
