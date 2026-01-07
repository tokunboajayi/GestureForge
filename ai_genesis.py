"""
AI 3D Genesis Module (OPTIMIZED)
Converts 2D images to 3D meshes using local AI models.

Supports:
- TripoSR (Hugging Face): Fast single-image to 3D
- Fallback: Fast procedural sphere for offline/demo mode

OPTIMIZATIONS:
- Zero-delay fallback sphere generation
- Cached Fibonacci sphere for uniform distribution
- NumPy-vectorized operations throughout
"""

import numpy as np
import os
import threading

# Pre-computed golden ratio for Fibonacci sphere
PHI = (1 + np.sqrt(5)) / 2


class AI3DGenerator:
    """Generates 3D meshes from 2D images using AI (OPTIMIZED)."""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.use_fallback = True
        self._cached_sphere = None  # Cache for sphere vertices
        self._load_model_async()
    
    def _load_model_async(self):
        """Attempt to load TripoSR model in background."""
        def load():
            try:
                from tsr.system import TSR
                print("Loading TripoSR model...")
                self.model = TSR.from_pretrained(
                    "stabilityai/TripoSR",
                    config_name="config.yaml",
                    weight_name="model.ckpt"
                )
                self.model_loaded = True
                self.use_fallback = False
                print("TripoSR model loaded!")
            except ImportError:
                print("TripoSR not installed. Using fast fallback.")
            except Exception as e:
                print(f"TripoSR failed: {e}")
        
        threading.Thread(target=load, daemon=True).start()
    
    def generate_mesh(self, image_path, callback=None):
        """Generate 3D mesh from image (async, non-blocking)."""
        def process():
            if self.use_fallback:
                vertices = self._generate_fibonacci_sphere()
            else:
                vertices = self._generate_with_triposr(image_path)
            
            if callback:
                callback(vertices)
        
        threading.Thread(target=process, daemon=True).start()
    
    def _generate_fibonacci_sphere(self, num_points=500, radius=2.0):
        """
        OPTIMIZED: Generate uniform Fibonacci sphere (no random clustering).
        
        Fibonacci sphere provides better visual distribution than random sampling.
        Uses vectorized NumPy for speed.
        """
        # Use cached sphere if available
        if self._cached_sphere is not None and len(self._cached_sphere) == num_points:
            return self._cached_sphere.copy()
        
        # Vectorized Fibonacci sphere generation
        indices = np.arange(num_points)
        
        # Golden angle in radians
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        
        # Y goes from 1 to -1 (uniform distribution along axis)
        y = 1 - (indices / (num_points - 1)) * 2
        
        # Radius at y
        r = np.sqrt(1 - y * y)
        
        # Golden angle increment
        theta = golden_angle * indices
        
        # Convert to Cartesian
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        
        vertices = np.column_stack((x, y, z)).astype(np.float32) * radius
        
        # Cache for reuse
        self._cached_sphere = vertices.copy()
        
        return vertices
    
    def _generate_with_triposr(self, image_path):
        """Use TripoSR to generate 3D from image."""
        try:
            import torch
            from PIL import Image
            from tsr.utils import remove_background
            
            image = Image.open(image_path).convert("RGB")
            image = remove_background(image)
            
            print(f"Generating 3D from: {image_path}")
            with torch.no_grad():
                scene_codes = self.model([image], device="cuda")
                meshes = self.model.extract_mesh(scene_codes)
            
            mesh = meshes[0]
            vertices = np.array(mesh.vertices, dtype=np.float32)
            
            # Normalize to radius 2
            center = vertices.mean(axis=0)
            vertices -= center
            max_dist = np.max(np.linalg.norm(vertices, axis=1))
            vertices = (vertices / max_dist) * 2.0
            
            return vertices
            
        except Exception as e:
            print(f"TripoSR failed: {e}")
            return self._generate_fibonacci_sphere()


class MeshLoader:
    """Load and parse 3D mesh files (.obj, .glb)."""
    
    @staticmethod
    def load_obj(filepath):
        """Load vertices from .obj file (optimized)."""
        vertices = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(vertices, dtype=np.float32)
    
    @staticmethod
    def sample_vertices(vertices, num_samples=500):
        """Sample vertices uniformly (deterministic)."""
        if len(vertices) <= num_samples:
            return vertices
        step = len(vertices) // num_samples
        return vertices[::step][:num_samples]


# Singleton pattern for generator
_generator = None

def get_generator():
    """Get or create the AI 3D generator instance."""
    global _generator
    if _generator is None:
        _generator = AI3DGenerator()
    return _generator

def generate_3d_from_image(image_path, callback):
    """Main API: Generate 3D mesh from image (non-blocking)."""
    generator = get_generator()
    generator.generate_mesh(image_path, callback)
