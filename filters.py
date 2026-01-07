import cv2
import numpy as np

class HandKalmanFilter:
    def __init__(self):
        # 4 state variables (x, y, dx, dy), 2 measurement variables (x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Transition Matrix (Physics model)
        # x = x + dx
        # y = y + dy
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        # Measurement Matrix (We observe x, y)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        # Process Noise Covariance (Q) - How much system jitters
        # Lower = smoother but laggy. Higher = faster but noisy.
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Measurement Noise Covariance (R) - How much we trust the sensor
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0 # Trust sensor moderately
        
        # Error Covariance (Posterior)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        self.is_initialized = False

    def predict(self):
        pred = self.kf.predict()
        return (pred[0][0], pred[1][0])

    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        
        if not self.is_initialized:
            # Initialize state with first measurement
            self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.is_initialized = True
        
        self.kf.correct(measurement)
        return self.predict()
