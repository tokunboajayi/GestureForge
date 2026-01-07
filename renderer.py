import pygame
import cv2
import numpy as np

class VoxelRenderer:
    def update_camera_frame(self, frame):
        """
        Converts OpenCV BGR frame to Pygame Surface
        """
        # 1. BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Transpose (Rotates image correctly for Pygame coordinate system if needed)
        # OpenCV (0,0) is Top-Left. Pygame (0,0) is Top-Left.
        # But pygame.surfarray.make_surface expects coords as [x][y], while numpy is [row][col] (y, x).
        # So we transpose axes.
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
        
        return pygame.surfarray.make_surface(frame_rgb)

    def draw_isometric_voxel(self, x, y, size=20, color=(0, 255, 255)):
        # Isometric projection approximation
        #   p1
        # p2  p4
        #   p3
        # p5  p7
        #   p6
        
        # Top Face (Diamond)
        p1 = (x, y - size)
        p2 = (x - size, y - size/2)
        p3 = (x, y)
        p4 = (x + size, y - size/2)
        
        # Side Walls
        p5 = (x - size, y + size/2)
        p6 = (x, y + size)
        p7 = (x + size, y + size/2)
        
        # Colors - Dynamic generation based on base color
        # Assuming color is RGB tuple
        r, g, b = color
        
        # Make brighter/darker variants
        top_color = color
        left_color = (max(0, r-50), max(0, g-50), max(0, b-50))
        right_color = (max(0, r-100), max(0, g-100), max(0, b-100))
        edge_color = (min(255, r+50), min(255, g+50), min(255, b+50)) # Neon Highlight
        
        # Fill Faces
        pygame.draw.polygon(self.screen, top_color, [p1, p2, p3, p4])
        pygame.draw.polygon(self.screen, left_color, [p2, p5, p6, p3])
        pygame.draw.polygon(self.screen, right_color, [p4, p3, p6, p7])
        
        # Draw Edges (Neon Glow)
        thickness = 2
        pygame.draw.line(self.screen, edge_color, p1, p2, thickness)
        pygame.draw.line(self.screen, edge_color, p2, p3, thickness)
        pygame.draw.line(self.screen, edge_color, p3, p4, thickness)
        pygame.draw.line(self.screen, edge_color, p4, p1, thickness)
        
        pygame.draw.line(self.screen, edge_color, p2, p5, thickness)
        pygame.draw.line(self.screen, edge_color, p5, p6, thickness)
        pygame.draw.line(self.screen, edge_color, p6, p3, thickness)
        
        pygame.draw.line(self.screen, edge_color, p4, p7, thickness)
        pygame.draw.line(self.screen, edge_color, p7, p6, thickness)

    def draw(self, frame_surface, tracking_center=None, is_pinching=False):
        # 1. Draw Camera Feed
        self.screen.blit(frame_surface, (0, 0))
        
        # 2. Draw Voxels (Isometric)
        # Ideally, sort by Y to handle depth (Painter's Algorithm)
        sorted_voxels = sorted(self.voxels, key=lambda v: v[1])
        
        for v in sorted_voxels:
            x, y, color = v
            # Use stored color if tuple, else default
            c = color if isinstance(color, tuple) or isinstance(color, list) else (0, 255, 255)
            self.draw_isometric_voxel(x, y, size=15, color=c)

        # 3. Draw Cursor (Hand)
        if tracking_center:
            cx, cy = tracking_center
            
            # Cursor Color dependent on mode
            if self.is_eraser:
                cursor_col = (255, 0, 0) # Red for eraser
            else:
                cursor_col = self.current_color

            pygame.draw.circle(self.screen, cursor_col, (cx, cy), 5)
            
            # If drawing, visualize the block being placed ghost
            if is_pinching:
                # Ghost Voxel
                if not self.is_eraser:
                     self.draw_isometric_voxel(cx, cy, size=15, color=cursor_col)
                # If eraser, draw red X? Or just red circle.

        # 4. HUD
        mode_text = "ERASER" if self.is_eraser else "DRAW"
        stats = f"Voxels: {len(self.voxels)} | Mode: {mode_text} | Color: {self.color_names[self.color_idx]}"
        fps_text = self.font.render(stats, True, (255, 255, 255))
        self.screen.blit(fps_text, (10, 10))
        
        controls = "Controls: Pinch=Draw | Space=Clear | C=Color | E=Eraser | S=Save | L=Load"
        ctrl_text = self.font.render(controls, True, (200, 200, 200))
        self.screen.blit(ctrl_text, (10, 30))

        pygame.display.flip()

    def add_voxel(self, x=0, y=0, generated_x=None, generated_y=None):
        # Alias support for faulty main.py call
        if generated_x is not None: x = generated_x
        if generated_y is not None: y = generated_y
        
        # Snap to grid?
        grid_size = 20
        snap_x = round(x / grid_size) * grid_size
        snap_y = round(y / grid_size) * grid_size
        
        # Collision Check
        existing_idx = -1
        for i, v in enumerate(self.voxels):
            if v[0] == snap_x and v[1] == snap_y:
                existing_idx = i
                break
        
        if self.is_eraser:
            if existing_idx != -1:
                self.voxels.pop(existing_idx)
        else:
            # Draw Mode
            if existing_idx == -1:
                self.voxels.append((snap_x, snap_y, self.current_color))

    def clear(self):
        self.voxels = []

    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.clear()
                if event.key == pygame.K_ESCAPE:
                    return False
                
                # New Controls
                if event.key == pygame.K_c:
                    self.cycle_color()
                if event.key == pygame.K_e:
                    self.is_eraser = not self.is_eraser
                if event.key == pygame.K_s:
                    self.save_canvas()
                if event.key == pygame.K_l:
                    self.load_canvas()
        return True

    # --- New Features ---
    def __init__(self, width, height):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("NS-VOXEL-PY")
        
        # Canvas: List of voxels [(x, y, color)]
        self.voxels = []
        
        # Colors
        self.colors = [
            (0, 255, 255), # Cyan
            (255, 0, 255), # Magenta
            (0, 255, 0),   # Green
            (255, 100, 0), # Orange
            (255, 255, 255)# White
        ]
        self.color_names = ["Cyan", "Magenta", "Green", "Orange", "White"]
        self.color_idx = 0
        self.current_color = self.colors[0]
        
        self.is_eraser = False
        
        # Fonts
        self.font = pygame.font.SysFont("Arial", 18)

    def cycle_color(self):
        self.color_idx = (self.color_idx + 1) % len(self.colors)
        self.current_color = self.colors[self.color_idx]

    def save_canvas(self):
        import json
        try:
            with open("canvas.json", "w") as f:
                json.dump(self.voxels, f)
            print("Saved to canvas.json")
        except Exception as e:
            print("Save failed:", e)

    def load_canvas(self):
        import json
        import os
        if os.path.exists("canvas.json"):
            try:
                with open("canvas.json", "r") as f:
                    # Convert lists back to tuples if needed, though list is fine
                    data = json.load(f)
                    self.voxels = [tuple(v) for v in data] 
                print("Loaded canvas.json")
            except Exception as e:
                print("Load failed:", e)
