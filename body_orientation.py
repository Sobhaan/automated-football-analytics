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


        self.forward_vector = np.array(forward_vector, dtype=float)
        norm = np.linalg.norm(self.forward_vector)
        self.forward_angle_rad = math.atan2(self.forward_vector[1], self.forward_vector[0])

        self.smoothing_window = smoothing_window
        self.pose_conf_threshold = pose_conf_threshold
        self.crop_padding = crop_padding_pixels

        # History only for the target ID
        self.orientation_history: deque = deque(maxlen=self.smoothing_window)

        # Constants for classification
        self.OPEN_THRESHOLD_RAD = math.pi / 8.0   # ~22.5 deg
        self.CLOSED_THRESHOLD_RAD = 7 * math.pi / 8.0 # ~157.5 deg

    def _calculate_single_orientation(self, frame, kpts: np.ndarray, forward_vector =(-0.9978, -0.0665),
                                        prev_shoulder_vector=[0,0], visualisation = False) -> str:
        
        if kpts is None or len(kpts) < 7:
            print("ERROR: Not enough keypoints to calculate orientation.")
            return "Unknown", prev_shoulder_vector
        l_sh, r_sh = np.array(kpts[5,:2]), np.array(kpts[6,:2])
        l_conf, r_conf = kpts[5,2], kpts[6,2]

        shoulder_vector = (r_sh - l_sh)
        
        if l_conf < 0.65 or r_conf < 0.65:
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
        
        label = "Half Open"
        if angle_deg < 60:
            label = "Open"
        elif angle_deg > 100:
            label = "Closed"

        if visualisation:
            img = frame.copy()
            # print(kpts)
            h, w = img.shape[:2]
            for idx, kp in enumerate(kpts):
            # Convert relative coordinates to absolute
                x_rel, y_rel, conf = kp

                # Draw the keypoint
                # Calculate the end point for the shoulder_vector line
                start_draw_point_x = w // 2
                start_draw_point_y = h // 2
                start_draw_point = (start_draw_point_x, start_draw_point_y-10)
                end_point_shoulder_x = start_draw_point_x + int(shoulder_vector[0] * 20)
                end_point_shoulder_y = start_draw_point_y + int(shoulder_vector[1] * 20)
                end_point_shoulder = (end_point_shoulder_x, end_point_shoulder_y)

                # Calculate the end point for the perp_vector line
                end_point_perp_x = start_draw_point_x + int(perp_vector[0] * 20)
                end_point_perp_y = start_draw_point_y + int(perp_vector[1] * 20)
                end_point_perp = (end_point_perp_x, end_point_perp_y)

                end_forward_x = start_draw_point_x + int(forward_vector[0] * 20)
                end_forward_y = start_draw_point_y + int(forward_vector[1] * 20)
                end_forward = (end_forward_x, end_forward_y)
                # cv2.circle(img, (int(x_rel), int(y_rel)), 2, (0,255,0), 1)
                cv2.line(img, start_draw_point, end_point_perp, (255,0,0), 1)
                cv2.line(img, start_draw_point, end_forward, (0,255,0), 1)
                cv2.line(img, start_draw_point, end_point_shoulder, (0,0,255), 1)
                # Optionally, you can label the keypoint index
                if not math.isnan(float(angle_deg)): 
                    txt = str(int(angle_deg))
                else:
                    txt = "NAN"
                # cv2.putText(img, txt, start_draw_point, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                # cv2.putText(img, str(idx), (int(x_rel), int(y_rel)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
            img = cv2.resize(
                img,
                (int(w * 4), int(h * 4)),
                interpolation=cv2.INTER_LINEAR
            )
            cv2.imshow('Keypoints', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
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
    

