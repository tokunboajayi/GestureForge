import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import ctypes
import cv2
import time
import math
import os
import json
from datetime import datetime
from ai_genesis import generate_3d_from_image

class OpenGLRenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # Pygame Setup for OpenGL
        pygame.init()
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("NS-VOXEL-PY (OpenGL Accelerated)")
        
        # OpenGL Scene Setup
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        # FOV 45, Ratio, Near 0.1, Far 100.0
        gluPerspective(45, (width / height), 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Camera Position: Back 10 units
        glTranslatef(0.0, 0.0, -10.0)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Voxel Data
        # Dictionary for sparse storage: (x, y, z) -> (r, g, b)
        self.voxels = {}
        
        # VBOs for Instancing
        self.cube_vertices = np.array([
            # Front face
            -0.5, -0.5,  0.5,
             0.5, -0.5,  0.5,
             0.5,  0.5,  0.5,
            -0.5,  0.5,  0.5,
            # Back face
            -0.5, -0.5, -0.5,
            -0.5,  0.5, -0.5,
             0.5,  0.5, -0.5,
             0.5, -0.5, -0.5,
        ], dtype=np.float32)

        self.cube_indices = np.array([
            0, 1, 2, 2, 3, 0, # Front
            4, 5, 6, 6, 7, 4, # Back
            1, 7, 6, 6, 2, 1, # Right
            4, 0, 3, 3, 5, 4, # Left
            3, 2, 6, 6, 5, 3, # Top
            4, 7, 1, 1, 0, 4  # Bottom
        ], dtype=np.uint32)
        
        # Shader Setup (Simple Pipeline)
        self.shader = self.create_shader()
        
        # Colors
        self.colors = [
            (0, 1, 1),   # Cyan
            (1, 0, 1),   # Magenta
            (0, 1, 0),   # Green
            (1, 0.5, 0), # Orange
            (1, 1, 1)    # White
        ]
        self.color_idx = 0
        self.current_color = self.colors[0]
        self.is_eraser = False

        # Texture ID for Background
        self.bg_texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.bg_texture_id)
        # Texture Parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)

        # Optimization Support
        self.cache_valid = False
        self.pos_array = np.empty((0, 3), dtype=np.float32)
        self.col_array = np.empty((0, 3), dtype=np.float32)
        
        # Navigation (Infinite Canvas)
        self.cam_pos = [0.0, 0.0, 0.0] # X, Y, Z
        
        # OPTIMUS: Pre-computed Tesseract Geometry (save per-frame recalc)
        self.base_corners = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1],  [1, -1, 1],  [1, 1, 1],  [-1, 1, 1]
        ], dtype=np.float32)
        self.line_indices = np.array([
            0,1, 1,2, 2,3, 3,0, # Back
            4,5, 5,6, 6,7, 7,4, # Front
            0,4, 1,5, 2,6, 3,7  # Connections
        ], dtype=np.int32)
        
        # Toast State
        self.toast_message = ""
        self.toast_timer = 0
        
        # Font for HUD
        self.pygame_font = pygame.font.SysFont("monospace", 18)
        
        # OPTIMUS: VBO (Vertex Buffer Objects) for GPU-side geometry
        self.vbo_vertex = glGenBuffers(1)
        self.vbo_color = glGenBuffers(1)
        self.vbo_vertex_count = 0  # Number of vertices in VBO
        
        # OPTIMUS: Recording Mode (Hero Demo)
        self.is_recording = False
        self.recording_frames = []
        self.recording_dir = "recordings"
        os.makedirs(self.recording_dir, exist_ok=True)
        
        # AI 3D Genesis State Machine
        # States: DRAW, GENERATING, MORPHING, VIEWING
        self.genesis_state = "DRAW"
        self.genesis_mesh = None  # Will hold loaded mesh vertices
        self.genesis_particles = None  # Particle positions during morph
        self.genesis_target_vertices = None  # Target mesh vertices
        self.genesis_rotation = [0.0, 0.0, 0.0]  # X, Y, Z rotation angles
        
        # REALISTIC PHYSICS: Rotation velocity & grab state
        self.genesis_velocity = [0.0, 0.0]  # X, Y angular velocity (degrees/frame)
        self.genesis_grabbed = False  # Is user currently grabbing the model?
        self.genesis_friction = 0.98  # Velocity decay per frame (inertia)
        self.genesis_sensitivity = 3.0  # Rotation speed multiplier
        self.genesis_glow = 0.0  # Glow intensity (0-1) when grabbed
        
        # SCANNER: Animated scanning plane (ported from ns-arc)
        self.genesis_scanner_y = 0.0  # Scanner Y position (oscillates)
        
        self.exports_dir = "exports"
        os.makedirs(self.exports_dir, exist_ok=True)

    def create_shader(self):
        return None 
    
    # ... (update_camera_frame, draw_background, cycle_color, clear, add_voxel, remove_voxel, rebuild_arrays, draw_hand_skeleton preserved) ...
    # I will assume I am replacing the whole class or targeted methods.
    # To be safe and concise, I will target specific blocks. 
    # But I need to add cam_pos to __init__.
    # And Fog to draw.
    
    # I'll just use MultiReplace or multiple chunks if needed.
    # Actually, I'll rewrite `draw` entirely to include Fog and Transform.
    # And `__init__`.
    
    # Let's do `__init__` first. 

    def update_camera_frame(self, frame):
        # Frame is BGR from OpenCV, convert to RGB
        # Also need to flip it vertically for OpenGL Texture usually
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 0)
        
        frame_data = frame_rgb.tobytes()
        width, height = frame_rgb.shape[1], frame_rgb.shape[0]
        
        glBindTexture(GL_TEXTURE_2D, self.bg_texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, frame_data)

    def draw_background(self):
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.bg_texture_id)
        glColor3f(1, 1, 1)
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(1, 0)
        glTexCoord2f(1, 1); glVertex2f(1, 1)
        glTexCoord2f(0, 1); glVertex2f(0, 1)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)

    def cycle_color(self):
        self.color_idx = (self.color_idx + 1) % len(self.colors)
        self.current_color = self.colors[self.color_idx]
        print(f"Color Switched: {self.current_color}")

    def clear(self):
        self.voxels = {}
        self.cache_valid = False
        print("Canvas Cleared")

    def add_stroke(self, start_pos, end_pos):
        x1, y1, z1 = start_pos
        x2, y2, z2 = end_pos
        
        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        
        if dist < 5:
            self.add_voxel(x2, y2, z2)
            return
            
        points = int(dist // 2) + 1 # Less points for bigger boxes
        
        for i in range(points):
            t = i / points
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            z = z1 + (z2 - z1) * t
            self.add_voxel(x, y, z)

    def add_voxel(self, x=0, y=0, z=0, color=None, generated_x=None, generated_y=None):
        # Alias
        if generated_x is not None: x = generated_x
        if generated_y is not None: y = generated_y
        
        if self.is_eraser:
            self.remove_voxel(x, y, z)
            return

        # Normalize/Quantize coordinates
        curr_x = (x / self.width) * 10 - 5
        curr_y = -((y / self.height) * 8 - 4)
        curr_z = z 
        
        # World Space
        world_x = curr_x + self.cam_pos[0]
        world_y = curr_y + self.cam_pos[1]
        world_z = curr_z + self.cam_pos[2]
        
        grid_size = 0.1 # Bigger Boxes
        
        sx = round(world_x / grid_size) * grid_size
        sy = round(world_y / grid_size) * grid_size
        sz = round(world_z / grid_size) * grid_size
        
        if color is None: color = self.current_color
        
        key = (sx, sy, sz)
        if key not in self.voxels:
            self.voxels[key] = color
            self.cache_valid = False

    def remove_voxel(self, x, y, z=0):
        curr_x = (x / self.width) * 10 - 5
        curr_y = -((y / self.height) * 8 - 4)
        curr_z = z 
        
        world_x = curr_x + self.cam_pos[0]
        world_y = curr_y + self.cam_pos[1]
        world_z = curr_z + self.cam_pos[2]
        
        grid_size = 0.1
        sx = round(world_x / grid_size) * grid_size
        sy = round(world_y / grid_size) * grid_size
        sz = round(world_z / grid_size) * grid_size
        
        key = (sx, sy, sz)
        if key in self.voxels:
            del self.voxels[key]
            self.cache_valid = False

    def rebuild_arrays(self):
        # ... preserved ...
        pass # Not modifying rebuild_arrays logic here, just context if needed.
        # Check if rebuild_arrays is in range?
        # I'll rely on the fact that I'm replacing the methods.
        # But wait, rebuild_arrays is AFTER these. 
        # I will replace up to remove_voxel.

# ... later ...
# I need to update `draw` too. separate chunk.
            
    def rebuild_arrays(self):
        if not self.voxels:
            self.pos_array = np.empty((0, 3), dtype=np.float32)
            self.col_array = np.empty((0, 3), dtype=np.float32)
        else:
            # Keys to list
            keys = list(self.voxels.keys())
            cols = list(self.voxels.values())
            self.pos_array = np.array(keys, dtype=np.float32)
            self.col_array = np.array(cols, dtype=np.float32)
        
        self.cache_valid = True
    
    def upload_vbo(self):
        """OPTIMUS: Upload static voxel position/color data to GPU buffers."""
        if len(self.pos_array) == 0:
            self.vbo_vertex_count = 0
            return
            
        # Note: Actual vertex data is computed per-frame due to pulsing animation
        # This just marks that we have voxels to render
        self.vbo_vertex_count = len(self.pos_array) * 64  # 64 vertices per tesseract

    def draw_hand_skeleton(self, landmarks):
        # Connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glLineWidth(2)
        glColor3f(0, 1, 1)
        glBegin(GL_LINES)
        for (s, e) in connections:
            if s < len(landmarks) and e < len(landmarks):
                p1 = landmarks[s]
                p2 = landmarks[e]
                glVertex2f(p1[1], p1[2])
                glVertex2f(p2[1], p2[2])
        glEnd()
        
        glPointSize(6)
        glBegin(GL_POINTS)
        for i, lm in enumerate(landmarks):
            if i in [4, 8, 12, 16, 20]:
                glColor3f(1, 1, 1)
            else:
                glColor3f(0, 0.6, 0.6)
            glVertex2f(lm[1], lm[2])
        glEnd()
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)

    def draw(self, frame, tracking_pos=None, is_pinching=False, landmarks_list=None, fps=0.0):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # 0. Background (No Fog)
        self.draw_background()
        
        # 1. World Rendering (With Fog)
        glEnable(GL_FOG)
        glFogi(GL_FOG_MODE, GL_EXP2)
        glFogf(GL_FOG_DENSITY, 0.02) # Thin fog for depth
        glFogfv(GL_FOG_COLOR, [0.0, 0.0, 0.0, 1.0])
        glHint(GL_FOG_HINT, GL_NICEST)
        
        # Camera Transform (Pan)
        glTranslatef(-self.cam_pos[0], -self.cam_pos[1], -self.cam_pos[2] - 10.0) 
        
        # Rebuild Geometry (Upload to VBO when changed)
        if not self.cache_valid:
            self.rebuild_arrays()
            self.upload_vbo()  # OPTIMUS: Upload to GPU
            
        if self.vbo_vertex_count > 0:
            t = time.time()
            pulse = 0.4 + 0.2 * math.sin(t * 3.0)
            s_out = 0.05 # Bigger Scale
            s_in = 0.05 * pulse
            
            N = len(self.pos_array)
            
            # OPTIMUS: Use pre-computed base geometry
            centers = self.pos_array[:, np.newaxis, :]
            corners_out = centers + self.base_corners * s_out
            corners_in = centers + self.base_corners * s_in
            
            lines_out = corners_out[:, self.line_indices, :]
            lines_in  = corners_in[:, self.line_indices, :]
            
            conn_stack = np.stack((corners_out, corners_in), axis=2)
            lines_conn = conn_stack.reshape(N, 16, 3)
            
            all_verts = np.concatenate((lines_out, lines_in, lines_conn), axis=1)
            final_verts = all_verts.reshape(-1, 3).astype(np.float32)
            colors_rep = np.repeat(self.col_array[:, np.newaxis, :], 64, axis=1).reshape(-1, 3).astype(np.float32)
            
            # Upload to VBO (dynamic for pulsing animation)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertex)
            glBufferData(GL_ARRAY_BUFFER, final_verts.nbytes, final_verts, GL_DYNAMIC_DRAW)
            
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_color)
            glBufferData(GL_ARRAY_BUFFER, colors_rep.nbytes, colors_rep, GL_DYNAMIC_DRAW)
            
            # Draw from VBO
            glLineWidth(1)
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertex)
            glVertexPointer(3, GL_FLOAT, 0, None)
            
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_color)
            glColorPointer(3, GL_FLOAT, 0, None)
            
            glDrawArrays(GL_LINES, 0, len(final_verts))
            
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        # AI 3D Genesis: Morph Animation
        if self.genesis_state in ("MORPHING", "VIEWING") and self.genesis_particles is not None:
            self.update_morph_animation()
            self.draw_genesis_particles()
        
        glDisable(GL_FOG)

        # 2. Overlays (Skeletons & GUI)
        if landmarks_list:
            for hand_lms in landmarks_list:
                self.draw_hand_skeleton(hand_lms)
        
        # 3. Cursor / Pinch Indicator
        # Cursor is Screen Space
        if tracking_pos:
            tx, ty = tracking_pos
            wx = (tx / self.width) * 10 - 5
            wy = -((ty / self.height) * 8 - 4)
            
            # Reset Camera
            glLoadIdentity()
            glTranslatef(0,0,-10) 
            glPushMatrix()
            glTranslatef(wx, wy, 0) # Move to Hand Pos
            
            if self.is_eraser:
                # Red Eraser
                glColor3f(1.0, 0.0, 0.0)
                glBegin(GL_TRIANGLES)
                glVertex3f(0, 0.2, 0); glVertex3f(-0.1, 0, 0); glVertex3f(0.1, 0, 0)
                glEnd()
            elif is_pinching:
                # Drawing Active: Glowing Dot
                glPointSize(10)
                glColor3f(1.0, 1.0, 1.0) # Core White
                glBegin(GL_POINTS)
                glVertex3f(0, 0, 0)
                glEnd()
                
                # Glow Ring
                glPointSize(20)
                glColor4f(self.current_color[0], self.current_color[1], self.current_color[2], 0.5)
                glBegin(GL_POINTS)
                glVertex3f(0, 0, 0)
                glEnd()
                
            glPopMatrix()
        
        # 4. HUD Overlay (FPS, Voxels, Toast)
        self.draw_hud(fps, len(self.voxels))

        pygame.display.flip()
    
    def draw_hud(self, fps, voxel_count):
        """Render FPS counter and toast notifications."""
        # FPS Counter
        fps_text = f"FPS: {int(fps)} | Voxels: {voxel_count}"
        text_surface = self.pygame_font.render(fps_text, True, (0, 255, 255))
        
        # Toast (if active)
        if time.time() < self.toast_timer:
            toast_surface = self.pygame_font.render(self.toast_message, True, (255, 255, 255))
        else:
            toast_surface = None
        
        # Blit to screen (Pygame over OpenGL)
        # Note: This requires switching to 2D mode temporarily
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Convert Pygame surface to OpenGL texture
        text_data = pygame.image.tostring(text_surface, "RGBA", False)
        w, h = text_surface.get_size()
        
        glEnable(GL_TEXTURE_2D)
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(10, 10)
        glTexCoord2f(1, 0); glVertex2f(10 + w, 10)
        glTexCoord2f(1, 1); glVertex2f(10 + w, 10 + h)
        glTexCoord2f(0, 1); glVertex2f(10, 10 + h)
        glEnd()
        
        glDeleteTextures([tex_id])
        
        # Toast
        if toast_surface:
            toast_data = pygame.image.tostring(toast_surface, "RGBA", False)
            tw, th = toast_surface.get_size()
            toast_tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, toast_tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, th, 0, GL_RGBA, GL_UNSIGNED_BYTE, toast_data)
            
            # Center toast at bottom
            tx = (self.width - tw) // 2
            ty = self.height - 50
            
            glColor4f(1, 1, 1, 1)
            glBegin(GL_QUADS)
            glTexCoord2f(0, 0); glVertex2f(tx, ty)
            glTexCoord2f(1, 0); glVertex2f(tx + tw, ty)
            glTexCoord2f(1, 1); glVertex2f(tx + tw, ty + th)
            glTexCoord2f(0, 1); glVertex2f(tx, ty + th)
            glEnd()
            
            glDeleteTextures([toast_tex])
        
        glDisable(GL_TEXTURE_2D)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)
    
    def show_toast(self, message, duration=2.0):
        """Display a toast notification for the specified duration."""
        self.toast_message = message
        self.toast_timer = time.time() + duration
    
    def capture_frame(self):
        """OPTIMUS: Capture current OpenGL buffer as numpy array."""
        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        frame = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 3)
        frame = np.flipud(frame)  # OpenGL reads bottom-up
        return frame
    
    def toggle_recording(self):
        """OPTIMUS: Toggle recording mode. On stop, export to GIF."""
        if self.is_recording:
            # Stop recording and export
            self.is_recording = False
            self.export_recording()
            self.show_toast(f"Recording Saved! ({len(self.recording_frames)} frames)")
        else:
            # Start recording
            self.recording_frames = []
            self.is_recording = True
            self.show_toast("Recording Started... (Press R to stop)")
    
    def export_recording(self):
        """OPTIMUS: Export captured frames as animated GIF."""
        if not self.recording_frames:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.recording_dir, f"voxel_demo_{timestamp}.gif")
        
        # Use OpenCV to write frames (GIF via imageio would be better, but cv2 works)
        # Convert to BGR for OpenCV
        frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in self.recording_frames]
        
        # Save as video (MP4) since cv2 doesn't do GIF directly
        filename_mp4 = filename.replace('.gif', '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename_mp4, fourcc, 30.0, (self.width, self.height))
        
        for frame in frames_bgr:
            out.write(frame)
        out.release()
        
        print(f"Recording exported to: {filename_mp4}")
        self.recording_frames = []
    
    def update_morph_animation(self):
        """AI 3D Genesis: Animate particles toward target mesh vertices."""
        if self.genesis_particles is None or self.genesis_target_vertices is None:
            return
            
        # Lerp factor (how fast particles move toward target)
        lerp_speed = 0.05
        
        # Move particles toward targets
        delta = self.genesis_target_vertices - self.genesis_particles
        self.genesis_particles += delta * lerp_speed
        
        # Check if morph is complete (particles close to targets)
        distance = np.linalg.norm(delta, axis=1).mean()
        if distance < 0.05 and self.genesis_state == "MORPHING":
            self.genesis_state = "VIEWING"
            self.show_toast("3D Model Complete! Grab to rotate!")
        
        # PHYSICS: Apply inertia-based rotation when not grabbed
        if self.genesis_state == "VIEWING":
            if not self.genesis_grabbed:
                # Apply velocity to rotation
                self.genesis_rotation[0] += self.genesis_velocity[0]
                self.genesis_rotation[1] += self.genesis_velocity[1]
                
                # Apply friction (gradual slowdown)
                self.genesis_velocity[0] *= self.genesis_friction
                self.genesis_velocity[1] *= self.genesis_friction
                
                # Stop if velocity is very small
                if abs(self.genesis_velocity[0]) < 0.1:
                    self.genesis_velocity[0] = 0
                if abs(self.genesis_velocity[1]) < 0.1:
                    self.genesis_velocity[1] = 0
            
            # Update glow effect (fade in/out)
            if self.genesis_grabbed:
                self.genesis_glow = min(1.0, self.genesis_glow + 0.1)
            else:
                self.genesis_glow = max(0.0, self.genesis_glow - 0.05)
    
    def draw_genesis_particles(self):
        """AI 3D Genesis: Render the morphing/morphed particle cloud (OPTIMIZED)."""
        if self.genesis_particles is None:
            return
            
        num_particles = len(self.genesis_particles)
        
        # OPTIMIZATION: Use depth-based coloring for 3D effect
        if self.genesis_state == "MORPHING":
            # Animated pulsing color during morph
            t = time.time()
            pulse = 0.5 + 0.5 * math.sin(t * 5.0)
            # Add depth-based brightness (further = darker)
            z_values = self.genesis_particles[:, 2]
            z_norm = (z_values - z_values.min()) / (z_values.max() - z_values.min() + 0.001)
            brightness = 0.5 + 0.5 * z_norm  # 0.5 to 1.0
            colors = np.column_stack([
                brightness * pulse,
                brightness,
                brightness
            ]).astype(np.float32)
        else:
            # VIEWING: Gradient based on depth for 3D lighting effect
            z_values = self.genesis_particles[:, 2]
            z_norm = (z_values - z_values.min()) / (z_values.max() - z_values.min() + 0.001)
            
            # GLOW EFFECT when grabbed (brighter, more vibrant)
            glow = self.genesis_glow
            colors = np.column_stack([
                0.1 + 0.2 * z_norm + glow * 0.3,  # R: warmer when grabbed
                0.7 + 0.3 * z_norm + glow * 0.3,  # G: bright green
                0.8 + 0.2 * z_norm + glow * 0.2   # B: blue
            ]).astype(np.float32)
            
            # Clamp colors to valid range
            colors = np.clip(colors, 0, 1)
        
        # Push a fresh matrix for genesis rendering
        glPushMatrix()
        
        # Apply rotation transform
        glRotatef(self.genesis_rotation[0], 1, 0, 0)  # X-axis
        glRotatef(self.genesis_rotation[1], 0, 1, 0)  # Y-axis
        glRotatef(self.genesis_rotation[2], 0, 0, 1)  # Z-axis
        
        # OPTIMIZATION: Enable blending for soft particles
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # OPTIMIZATION: Larger, more visible points with smooth rendering
        glPointSize(12)  # Dense solid appearance
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        
        glVertexPointer(3, GL_FLOAT, 0, self.genesis_particles)
        glColorPointer(3, GL_FLOAT, 0, colors)
        
        glDrawArrays(GL_POINTS, 0, num_particles)
        
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        
        glDisable(GL_POINT_SMOOTH)
        glDisable(GL_BLEND)
        
        glPopMatrix()
        
        # Draw scanner plane (NS-ARC style)
        self.draw_scanner()
    
    def draw_scanner(self):
        """Draw animated scanner plane (OPTIMIZED with vertex arrays)."""
        if self.genesis_particles is None:
            return
        
        # Animate scanner Y position (oscillate)
        t = time.time()
        y = math.sin(t * 2.0) * 1.5  # -1.5 to 1.5
        
        # Pre-computed scanner grid vertices (static pattern, only Y changes)
        hs = 2.0  # half_size
        
        # Grid lines: 5 horizontal + 5 vertical + 4 border = 28 vertices
        scanner_verts = np.array([
            # Grid X lines
            [-hs, y, -hs], [hs, y, -hs],
            [-hs, y, -hs/2], [hs, y, -hs/2],
            [-hs, y, 0], [hs, y, 0],
            [-hs, y, hs/2], [hs, y, hs/2],
            [-hs, y, hs], [hs, y, hs],
            # Grid Z lines
            [-hs, y, -hs], [-hs, y, hs],
            [-hs/2, y, -hs], [-hs/2, y, hs],
            [0, y, -hs], [0, y, hs],
            [hs/2, y, -hs], [hs/2, y, hs],
            [hs, y, -hs], [hs, y, hs],
        ], dtype=np.float32)
        
        # Apply same rotation as genesis model
        glPushMatrix()
        glRotatef(self.genesis_rotation[0], 1, 0, 0)
        glRotatef(self.genesis_rotation[1], 0, 1, 0)
        glRotatef(self.genesis_rotation[2], 0, 0, 1)
        
        # Enable blending for translucent scanner
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # Additive blend
        
        glLineWidth(2)
        glColor4f(1.0, 0.0, 1.0, 0.5)  # Magenta
        
        # OPTIMIZED: Use vertex array instead of immediate mode
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, scanner_verts)
        glDrawArrays(GL_LINES, 0, len(scanner_verts))
        glDisableClientState(GL_VERTEX_ARRAY)
        
        glDisable(GL_BLEND)
        glLineWidth(1)
        
        glPopMatrix()
    
    def export_sketch(self):
        """AI 3D Genesis: Export current canvas as PNG for AI processing."""
        frame = self.capture_frame()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.exports_dir, f"sketch_{timestamp}.png")
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, frame_bgr)
        
        print(f"Sketch exported to: {filename}")
        return filename
    
    def start_genesis(self):
        """AI 3D Genesis: Begin the sketch → 3D → morph pipeline."""
        print(f"[GENESIS] start_genesis called. Current state: {self.genesis_state}")
        
        if self.genesis_state != "DRAW":
            self.show_toast("Already in Genesis mode!")
            print(f"[GENESIS] Blocked - not in DRAW state")
            return
            
        if len(self.voxels) == 0:
            self.show_toast("Draw something first!")
            print(f"[GENESIS] Blocked - no voxels")
            return
        
        print(f"[GENESIS] Starting with {len(self.voxels)} voxels")
        
        # Export current canvas
        sketch_path = self.export_sketch()
        self.show_toast("Morphing to 3D Shape...")
        
        # State transition
        self.genesis_state = "GENERATING"
        print(f"[GENESIS] State changed to GENERATING")
        
        # OPTIMIZED: Shape-aware ellipsoid fill with high density
        def on_mesh_complete(ai_vertices):
            """Callback when AI generates mesh vertices."""
            # SHAPE-AWARE 3D: Create ellipsoid that matches drawing aspect ratio
            
            # Get all voxel positions
            target_vertices = self.pos_array.copy()
            num_voxels = len(target_vertices)
            
            if num_voxels > 0:
                # Find bounding box of drawing
                min_x, max_x = target_vertices[:, 0].min(), target_vertices[:, 0].max()
                min_y, max_y = target_vertices[:, 1].min(), target_vertices[:, 1].max()
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                
                # SHAPE-AWARE: Use separate radii for X and Y to match aspect ratio
                radius_x = (max_x - min_x) / 2 * 0.9  # 90% of bounding box
                radius_y = (max_y - min_y) / 2 * 0.9
                radius_z = min(radius_x, radius_y) * 0.6  # Z-depth proportional
                
                # OPTIMIZED: Higher particle count for solid appearance
                num_solid_particles = 1200  # Dense fill (increased from 800)
                solid_vertices = []
                
                # Generate evenly distributed points in ELLIPSOID (shape-aware)
                for _ in range(num_solid_particles):
                    # Random point in ellipsoid using rejection sampling
                    while True:
                        x = np.random.uniform(-1, 1)
                        y = np.random.uniform(-1, 1)
                        z = np.random.uniform(-1, 1)
                        if x*x + y*y + z*z <= 1:  # Inside unit sphere
                            break
                    
                    # SHAPE-AWARE: Scale by individual axis radii
                    px = center_x + x * radius_x
                    py = center_y + y * radius_y
                    pz = z * radius_z  # Full Z-depth for 3D look
                    solid_vertices.append([px, py, pz])
                
                self.genesis_target_vertices = np.array(solid_vertices, dtype=np.float32)
                
                print(f"[GENESIS] Shape-aware ellipsoid: {radius_x:.1f} x {radius_y:.1f} x {radius_z:.1f}")
            else:
                self.genesis_target_vertices = ai_vertices  # Fallback to sphere
            
            print(f"[GENESIS] CALLBACK: Using {len(self.genesis_target_vertices)} SOLID target vertices")
            
            # Initialize particles at RANDOM positions (they will flow to target)
            num_particles = len(self.genesis_target_vertices)
            # Start particles from edges (for liquid flow effect)
            self.genesis_particles = np.random.uniform(-5, 5, (num_particles, 3)).astype(np.float32)
            
            print(f"[GENESIS] Particles initialized: {len(self.genesis_particles)}")
            self.genesis_state = "MORPHING"
            print(f"[GENESIS] State changed to MORPHING")
            self.show_toast("Liquid Morphing...")
        
        generate_3d_from_image(sketch_path, on_mesh_complete)
        print(f"[GENESIS] generate_3d_from_image called")
    
    def reset_genesis(self):
        """Reset genesis state to DRAW mode."""
        self.genesis_state = "DRAW"
        self.genesis_particles = None
        self.genesis_target_vertices = None
        self.genesis_rotation = [0.0, 0.0, 0.0]
        self.show_toast("Genesis Reset. Draw again!")

    def check_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
                if event.key == K_SPACE:
                    self.clear()
                    self.show_toast("Canvas Cleared!")
                if event.key == K_s:
                    self.save_world()
                    self.show_toast("World Saved!")
                if event.key == K_l:
                    self.load_world()
                    self.show_toast("World Loaded!")
                if event.key == K_r:
                    self.toggle_recording()
                if event.key == K_g:
                    self.start_genesis()
                if event.key == K_c:
                    self.reset_genesis()  # Clear genesis mode
        
        # Capture frame if recording
        if self.is_recording:
            self.recording_frames.append(self.capture_frame())
        
        return True
