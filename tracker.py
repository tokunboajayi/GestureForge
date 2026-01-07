import cv2
import mediapipe as mp
import math
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

from filters_euro import PointFilter # Switched to OneEuro

class HandTracker:
    def __init__(self, model_path='hand_landmarker.task'):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2, # Enable 2 Hands
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        
        self.landmarker = HandLandmarker.create_from_options(options)
        self.timestamp_ms = 0
        
        # Filters: Key = "Left" or "Right"
        self.filters = {
            "Left": PointFilter(min_cutoff=1.0, beta=10.0),  # OPTIMUS: β=10 for smoother tracking
            "Right": PointFilter(min_cutoff=1.0, beta=10.0)
        }
        
        # Tip Indices (Thumb, Index)
        self.tip_ids = [4, 8]
    
    def get_smooth_pos(self, index_tip_raw, label="Right"):
        """
        Returns smoothed (x, y, z) for index tip using OneEuroFilter.
        OPTIMIZED: Now uses full 3D filtering (X, Y, Z all smoothed).
        """
        import time
        x = index_tip_raw[0]
        y = index_tip_raw[1]
        z = index_tip_raw[2] if len(index_tip_raw) > 2 else 0
        
        # Create filter if not exists
        if label not in self.filters:
            self.filters[label] = PointFilter(min_cutoff=1.0, beta=5.0)

        # OPTIMIZED: Full 3D filtering
        fx, fy, fz = self.filters[label].filter3d(x, y, z, time.time())
        return (fx, fy, fz)

    def find_hands(self, img, draw=True):
        import time
        # ... existing find_hands code ...

        # Convert to RGB (MediaPipe needs RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create MP Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Calculate timestamp (monotonic)
        # Windows time.time() has low resolution, so we force increment
        curr_ts = int(time.time() * 1000)
        if curr_ts <= self.timestamp_ms:
            curr_ts = self.timestamp_ms + 1
        self.timestamp_ms = curr_ts
        
        # Detect
        # Note: detect_for_video expects ms timestamp
        detection_result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        
        hands_data = [] # List of {'label': 'Right', 'landmarks': [[id,x,y,z], ...]}
        
        if detection_result.hand_landmarks:
            h, w, c = img.shape
            
            for i, hand_lms in enumerate(detection_result.hand_landmarks):
                # Get Handedness
                label = "Right"
                if detection_result.handedness and i < len(detection_result.handedness):
                     # MediaPipe often mirrors. "Left" might appear as "Right" depending on camera flip.
                     # We will trust MediaPipe label for now but might need to flip if camera is flipped.
                     label = detection_result.handedness[i][0].category_name
                
                # Convert to our list format [id, x, y]
                landmarks = []
                for id, lm in enumerate(hand_lms):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # Include Z (relative depth)
                    landmarks.append([id, cx, cy, lm.z])
                    
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                
                # Draw skeleton connections manually since we lost mp_draw
                if draw:
                    connections = mp.solutions.hands.HAND_CONNECTIONS if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands') else []
                    # Logic to draw lines would go here, but points are enough for now.
                    # Or we can just draw lines between known joints if we want.
                
                hands_data.append({'label': label, 'landmarks': landmarks})
                
        return hands_data

    def detect_gestures(self, landmarks):
        """
        Detects pinches between Thumb and other fingers using 3D distance.
        Returns: 
        {
            'index': (bool, cx, cy),
            'middle': (bool, cx, cy),
            'ring': (bool, cx, cy),
            'pinky': (bool, cx, cy)
        }
        """
        gestures = {
            'index': (False, 0, 0),
            'middle': (False, 0, 0),
            'ring': (False, 0, 0),
            'pinky': (False, 0, 0)
        }
        
        if not landmarks or len(landmarks) < 21:
            return gestures

        thumb_tip = landmarks[4]
        x1, y1 = thumb_tip[1], thumb_tip[2]
        z1 = thumb_tip[3] if len(thumb_tip) > 3 else 0

        # Finger Tips: Index(8), Middle(12), Ring(16), Pinky(20)
        fingers = {
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }

        for name, tip_id in fingers.items():
            finger_tip = landmarks[tip_id]
            x2, y2 = finger_tip[1], finger_tip[2]
            z2 = finger_tip[3] if len(finger_tip) > 3 else 0
            
            # OPTIMUS: 3D Euclidean distance (accounts for depth)
            # Z is normalized [-1, 1] range, scale it to similar magnitude as pixels
            z_scale = 100  # Approximate scaling factor
            length_3d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + ((z2 - z1) * z_scale)**2)
            
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Threshold (40px in 2D, ~50 in 3D to account for depth variation)
            if length_3d < 50:
                gestures[name] = (True, cx, cy)
        
        return gestures
