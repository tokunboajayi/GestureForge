import cv2
from tracker import HandTracker
from renderer_ogl import OpenGLRenderer # Switched to OpenGL
import time
import threading

class AsyncTracker:
    def __init__(self, tracker):
        self.tracker = tracker
        self.latest_data = []
        self.frame_to_process = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    
    def _loop(self):
        while self.running:
            img = None
            with self.lock:
                if self.frame_to_process is not None:
                    img = self.frame_to_process
                    self.frame_to_process = None
            
            if img is not None:
                # Heavy inference
                try:
                    data = self.tracker.find_hands(img, draw=False)
                    self.latest_data = data
                except Exception as e:
                    print(f"Tracking Error: {e}")
            else:
                time.sleep(0.001)

    def update(self, frame):
        with self.lock:
            # We overwrite so we always process only the LATEST frame
            self.frame_to_process = frame.copy() 

    def get_data(self):
        return self.latest_data

    def stop(self):
        self.running = False
        self.thread.join()

def main():
    # 1. Setup Camera (OpenCV)
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 2. Setup Modules
    base_tracker = HandTracker()
    async_tracker = AsyncTracker(base_tracker)
    renderer = OpenGLRenderer(width, height)
    
    print("NS-VOXEL-PY Started (Async Optimized).")
    print(" [ Controls ]")
    print(" - Pinch (Thumb+Index) to Draw")
    print(" - Spacebar to Clear")
    print(" - ESC to Quit")

    # State for triggers
    last_color_switch = 0
    last_wave_time = 0
    left_wrist_history = [] 
    
    # Stroke State
    last_draw_pos = None # (x, y, z)
    last_pan_pos = None # For Left Hand Pan
    last_zoom_pos = None # For Left Hand Zoom
    
    # Two-Hand Genesis State
    last_genesis_time = 0  # Cooldown for two-hand squeeze
    both_hands_pinching = False
    
    # FPS Tracking
    fps = 0.0
    frame_count = 0
    fps_timer = time.time()

    running = True
    while running:
        # A. Read Camera
        success, frame = cap.read()
        if not success:
            continue
            
        frame = cv2.flip(frame, 1) # Mirror
        
        # B. Track Hands (Async)
        async_tracker.update(frame) 
        hands_data = async_tracker.get_data()
        
        # Collect landmarks 
        all_landmarks_raw = [h['landmarks'] for h in hands_data]
        
        tracking_center = None
        is_pinching = False
        
        curr_time = time.time()
        
        # TWO-HAND SQUEEZE DETECTION FOR GENESIS
        left_pinching = False
        right_pinching = False
        for hand in hands_data:
            label = hand['label']
            landmarks = hand['landmarks']
            gestures = base_tracker.detect_gestures(landmarks, label)
            
            # --- RIGHT HAND (Drawing) ---
            if label == "Right":
                # Calculate Pinch Midpoint (More natural drawing)
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                
                mx = (thumb_tip[1] + index_tip[1]) / 2.0
                my = (thumb_tip[2] + index_tip[2]) / 2.0
                mz = (thumb_tip[3] + index_tip[3]) / 2.0 if len(thumb_tip) > 3 else 0 # Z is present now
                
                smoothed_pos = base_tracker.get_smooth_pos(
                    (mx, my, mz),
                    label="Right"
                )
                
                # Current 3D Pos
                cx, cy = smoothed_pos[0], smoothed_pos[1]
                cz = smoothed_pos[2] * 30.0
                curr_pos = (cx, cy, cz)
                tracking_center = (cx, cy)
                
                # Check Pinch
                pinching_now = gestures['index'][0] or gestures['middle'][0]
                
                if gestures['index'][0]: # Draw
                    is_pinching = True
                    renderer.is_eraser = False
                    
                    # Phase 3: UI Interaction Check
                    # Convert tracking center to screen coords equivalent
                    # Note: tracking_center is (cx, cy) in pixels already? No, mx, my are pixels.
                    # BaseTracker.get_smooth_pos returns pixels.
                    
                    ui_hit = renderer.check_ui_hit(cx, cy)
                    
                    if not ui_hit:
                        if last_draw_pos is None:
                            last_draw_pos = curr_pos
                        
                        # Interpolate Stroke
                        renderer.add_stroke(last_draw_pos, curr_pos)
                        last_draw_pos = curr_pos
                    else:
                        last_draw_pos = None # Reset draw if hitting UI
                    
                elif gestures['middle'][0]: # Erase
                    is_pinching = True # Show Cursor
                    renderer.is_eraser = True
                    
                    if last_draw_pos is None:
                        last_draw_pos = curr_pos
                        
                    renderer.add_stroke(last_draw_pos, curr_pos)
                    last_draw_pos = curr_pos
                else:
                    # Not pinching
                    last_draw_pos = None

            # --- LEFT HAND (Navigation & Aux) ---
            elif label == "Left":
                # 1. Pan / Navigate OR GRAB & SPIN Genesis Model (Index Pinch)
                if gestures['index'][0]:
                    l_idx = landmarks[8]
                    curr_pan = (l_idx[1], l_idx[2])
                    
                    if last_pan_pos is not None:
                        dx = (curr_pan[0] - last_pan_pos[0]) * (10.0 / width)
                        dy = -(curr_pan[1] - last_pan_pos[1]) * (8.0 / height)
                        
                        # Check if in Genesis VIEWING mode
                        if renderer.genesis_state == "VIEWING":
                            # GRAB: Set grabbed state
                            renderer.genesis_grabbed = True
                            
                            # SPIN: Apply rotation with sensitivity multiplier
                            rotation_x = dy * renderer.genesis_sensitivity * 50
                            rotation_y = dx * renderer.genesis_sensitivity * 50
                            
                            renderer.genesis_rotation[0] += rotation_x
                            renderer.genesis_rotation[1] += rotation_y
                            
                            # VELOCITY: Track for inertia on release
                            renderer.genesis_velocity[0] = rotation_x
                            renderer.genesis_velocity[1] = rotation_y
                        else:
                            # Normal pan behavior (Smooth)
                            renderer.target_cam_pos[0] -= dx * 1.5
                            renderer.target_cam_pos[1] -= dy * 1.5
                        
                    last_pan_pos = curr_pan
                else:
                    # RELEASE: Stop grabbing
                    if renderer.genesis_grabbed:
                        renderer.genesis_grabbed = False
                    last_pan_pos = None
                
                # DRAG & DROP: Middle pinch to move 3D model
                if gestures['middle'][0] and renderer.genesis_state == "VIEWING":
                    l_mid = landmarks[12]  # Middle fingertip
                    curr_drag = (l_mid[1], l_mid[2])
                    
                    if not renderer.genesis_dragging:
                        renderer.genesis_dragging = True
                        renderer._last_drag_pos = curr_drag
                    else:
                        dx = (curr_drag[0] - renderer._last_drag_pos[0]) * 0.01
                        dy = -(curr_drag[1] - renderer._last_drag_pos[1]) * 0.01
                        
                        # Move genesis model position
                        renderer.genesis_position[0] += dx
                        renderer.genesis_position[1] += dy
                        
                        renderer._last_drag_pos = curr_drag
                else:
                    renderer.genesis_dragging = False

                # 2. Cycle Color (Ring Pinch)
                if gestures['ring'][0]:
                    if curr_time - last_color_switch > 0.5:
                        renderer.cycle_color()
                        last_color_switch = curr_time
                
                # 3. Wave (Clear)
                # Only check if NOT pinching to avoid accidental clears while panning
                if not gestures['index'][0] and not gestures['pinky'][0]:
                    wrist = landmarks[0]
                    left_wrist_history.append((wrist[1], curr_time))
                    left_wrist_history = [p for p in left_wrist_history if curr_time - p[1] < 0.5]
                    
                    if len(left_wrist_history) > 5:
                        xs = [p[0] for p in left_wrist_history]
                        dist = max(xs) - min(xs)
                        if dist > 150 and (curr_time - last_wave_time > 1.0):
                            renderer.clear()
                            last_wave_time = curr_time
                            print("Wave Detected!")
                            
                # 4. Zoom (Pinky Pinch)
                if gestures['pinky'][0]:
                    l_pinky = landmarks[20] # Tip
                    curr_zoom = l_pinky[2] # Y-coord
                    
                    if last_zoom_pos is not None:
                        dy = (curr_zoom - last_zoom_pos) * (8.0 / height) # Scale
                        renderer.target_cam_pos[2] += dy * 2.0 # Smooth Zoom Speed
                        
                    last_zoom_pos = curr_zoom
                else:
                    last_zoom_pos = None
                    
                # Track left hand pinching for two-hand detection
                if gestures['index'][0]:
                    left_pinching = True
            
            # Track right hand pinching for two-hand detection
            if label == "Right" and gestures['index'][0]:
                right_pinching = True

        # TWO-HAND GESTURES: ZOOM & SCALE
        if left_pinching and right_pinching:
            # We have both hands pinching!
            # Calculate distance between pinch points (using index finger tips as proxy)
            # Need to get positions again as they are in local scope
            l_pos = None
            r_pos = None
            
            for hand in hands_data:
                if hand['label'] == "Left":
                    l_pos = hand['landmarks'][8] # Index tip
                elif hand['label'] == "Right":
                    r_pos = hand['landmarks'][8] # Index tip
            
            if l_pos is not None and r_pos is not None:
                # Calculate Euclidean distance between hands
                dx = l_pos[0] - r_pos[0]
                dy = l_pos[1] - r_pos[1]
                dist = (dx*dx + dy*dy)**0.5
                
                # FIRST FRAME of pinch
                if not renderer.is_two_hand_pinching:
                    renderer.is_two_hand_pinching = True
                    renderer.last_pinch_dist = dist
                    renderer.pinch_start_time = curr_time
                else:
                    # CONTINUOUS pinch
                    delta = dist - renderer.last_pinch_dist
                    
                    if renderer.genesis_state == "VIEWING":
                        # EMINENT-SCALE: Scale Genesis Model
                        renderer.genesis_scale = max(0.5, min(3.0, renderer.genesis_scale + delta * 0.005))
                    else:
                        # CAMERA ZOOM: Infinite Canvas (Smooth)
                        renderer.target_cam_pos[2] += delta * 1.5
                    
                    renderer.last_pinch_dist = dist

        else:
            # RESET pinch state
            if renderer.is_two_hand_pinching:
                renderer.is_two_hand_pinching = False
                
                # If short duration, treat as TRIGGER (Genesis)
                if (curr_time - renderer.pinch_start_time < 0.5) and (curr_time - last_genesis_time > 2.0):
                     if renderer.genesis_state != "VIEWING":
                        print("[GENESIS] TWO-HAND SQUEEZE DETECTED! (Triggered)")
                        renderer.start_genesis()
                        last_genesis_time = curr_time
            
            # Slowly return to base scale when not scaling (only for Genesis)
            if renderer.genesis_state == "VIEWING" and renderer.genesis_scale > 1.0:
                 renderer.genesis_scale = max(1.0, renderer.genesis_scale - 0.01)

        # D. Render
        renderer.update_camera_frame(frame)
        renderer.draw(frame, tracking_pos=tracking_center, is_pinching=is_pinching, landmarks_list=all_landmarks_raw, fps=fps)
        
        # E. Events
        running = renderer.check_events()
        
        # F. FPS Calculation
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()

    # Cleanup
    async_tracker.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Exiting...")

if __name__ == "__main__":
    main()
