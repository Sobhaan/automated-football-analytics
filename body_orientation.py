# target_orientation.py

import math
from collections import deque, Counter
import numpy as np
import pandas as pd
from ultralytics import YOLO
import cv2
from typing import List, Dict, Any, Tuple, Optional
from soccer.player import Player
import time

# --- Helper Functions ---
def angle_difference(angle1_rad, angle2_rad):
    """Calculates the smallest absolute difference between two angles (radians)."""
    diff = angle1_rad - angle2_rad
    while diff <= -math.pi: diff += 2 * math.pi
    while diff > math.pi: diff -= 2 * math.pi
    return abs(diff)

# --- Estimator Class for a SINGLE Target ID ---
class BodyOrientationEstimator:
    def __init__(self, target_id: int,
                 pose_model_path='yolov8x-pose-p6.pt',
                 forward_vector=(-1, 0.4),
                 smoothing_window=5,
                 pose_conf_threshold=0.5, # Confidence for pose detection WITHIN the crop
                 crop_padding_pixels=10): # Optional padding around crop
        """
        Initializes the orientation estimator for a specific target track ID.
        """
        if not isinstance(target_id, int):
             raise ValueError("target_id must be an integer.")
             
        print(f"Initializing Orientation Estimator for Target ID: {target_id}")
        self.target_id = target_id

        print(f"Loading Pose Model from: {pose_model_path}")
        try:
            self.pose_model = YOLO(pose_model_path)
            print("Pose Model Loaded.")
        except Exception as e:
            print(f"ERROR: Failed to load pose model '{pose_model_path}': {e}")
            raise # Re-raise the exception

        self.forward_vector = np.array(forward_vector, dtype=float)
        norm = np.linalg.norm(self.forward_vector)
        if norm > 1e-6:
            self.forward_angle_rad = math.atan2(self.forward_vector[1], self.forward_vector[0])
        else:
            print("Warning: Forward vector has zero length, using default (up).")
            self.forward_angle_rad = -math.pi / 2 # Angle pointing up

        self.smoothing_window = smoothing_window
        self.pose_conf_threshold = pose_conf_threshold
        self.crop_padding = crop_padding_pixels

        # History only for the target ID
        self.orientation_history: deque = deque(maxlen=self.smoothing_window)

        # Constants for classification
        self.OPEN_THRESHOLD_RAD = math.pi / 8.0   # ~22.5 deg
        self.CLOSED_THRESHOLD_RAD = 7 * math.pi / 8.0 # ~157.5 deg

    def _calculate_single_orientation(self, frame, kpts: np.ndarray, forward_vector =(-0.9978, -0.0665),
                                        prev_shoulder_vector=[0,0]) -> str:
        
        if kpts is None or len(kpts) < 7:
            print("ERROR: Not enough keypoints to calculate orientation.")
            return "Unknown", prev_shoulder_vector
        l_sh, r_sh = np.array(kpts[5,:2]), np.array(kpts[6,:2])
        l_conf, r_conf = kpts[5,2], kpts[6,2]

        shoulder_vector = (r_sh - l_sh)
        
        print(l_conf, r_conf)
        if l_conf < 0.92 or r_conf < 0.92:
            # # print(kpts)
            # if prev_shoulder_vector != [0,0]:
            #     shoulder_vector = prev_shoulder_vector
            
            #     # l_hip, r_hip = np.array(kpts[11,:2]), np.array(kpts[12,:2])
            #     # shoulder_vector = (r_hip - l_hip)
            if l_conf > r_conf:
                perp_vector = np.array((-1, 0))
            else:
                perp_vector = np.array((1, 0))
        else:
            perp_vector = np.array((shoulder_vector[1], -shoulder_vector[0]))

        dot_product = sum(i*j for i, j in zip(perp_vector, forward_vector))
        norm_u = math.sqrt(sum(i**2 for i in perp_vector))
        norm_v = math.sqrt(sum(i**2 for i in forward_vector))
        cos_theta = dot_product / (norm_u * norm_v)
        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)
        
        print(angle_deg)
        label = "Half Open"
        if angle_deg < 75:
            label = "Open"
        elif angle_deg > 110:
            label = "Closed"
        
        return label, shoulder_vector

    def _smooth_orientation(self, current_orientation: str) -> str:
        """Updates history and returns smoothed orientation for the target ID."""
        self.orientation_history.append(current_orientation)
        if len(self.orientation_history) < self.smoothing_window:
            return current_orientation # Not enough history yet
        # Get non-unknown votes from the history window
        known_orientations = [o for o in self.orientation_history if o != "Unknown"]
        if not known_orientations:
            return "Unknown" # No known orientations in window

        # Find majority
        counter = Counter(known_orientations)
        smoothed_orientation, _ = counter.most_common(1)[0]
        return smoothed_orientation

    
    def process_target_player_solo(self, img: np.ndarray, keypoints_rel: np.ndarray,
                               shoulder_vector: np.ndarray) -> Optional[Dict[str, Any]]:
        raw_orientation, shd_vec = self._calculate_single_orientation(img, keypoints_rel, forward_vector=self.forward_vector, prev_shoulder_vector=shoulder_vector)

        # --- 5. Smooth Orientation ---
        smoothed_orientation = self._smooth_orientation(raw_orientation)
        # --- 6. Return Results for Target Player ---
        return {
            'orientation_raw': raw_orientation,
            'orientation_smooth': smoothed_orientation,
            'shoulder_vector': shd_vec
        }
    

