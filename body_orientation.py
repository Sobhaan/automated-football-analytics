# target_orientation.py

import math
from collections import deque, Counter
import numpy as np
import pandas as pd
from ultralytics import YOLO
import cv2
from typing import List, Dict, Any, Tuple, Optional

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

    def _calculate_single_orientation(self, keypoints_xy: np.ndarray) -> str:
        """Calculates orientation from keypoints [K, 2]."""
        try:
            # Indices 5 and 6 are left and right shoulders in YOLOv8-Pose
            if keypoints_xy.shape[0] <= 6: return "Unknown" # Need at least 7 keypoints
            l_shoulder = keypoints_xy[5]
            r_shoulder = keypoints_xy[6]

            # Basic check if keypoints seem valid (e.g., not [0,0])
            # Note: YOLOv8 Pose might output [0,0] for invisible/low-conf keypoints
            if np.all(l_shoulder < 1) or np.all(r_shoulder < 1):
                 return "Unknown"

            shoulder_vector = r_shoulder - l_shoulder # dx, dy
            if np.linalg.norm(shoulder_vector) < 1e-6: return "Unknown" # Avoid division by zero if shoulders overlap

            # Angle of vector PERPENDICULAR to shoulders, pointing forward from chest
            body_facing_angle_rad = math.atan2(-shoulder_vector[1], shoulder_vector[0]) + math.pi

            diff_rad = angle_difference(body_facing_angle_rad, self.forward_angle_rad)

            if diff_rad <= self.OPEN_THRESHOLD_RAD: return "Open"
            elif diff_rad >= self.CLOSED_THRESHOLD_RAD: return "Closed"
            else: return "Half-Open"
        except IndexError:
             # print("Error: Not enough keypoints provided.") # Optional debug
             return "Unknown"
        except Exception as e:
             # print(f"Error calculating orientation: {e}") # Optional debug
             return "Unknown"

    def _smooth_orientation(self, current_orientation: str) -> str:
        """Updates history and returns smoothed orientation for the target ID."""
        self.orientation_history.append(current_orientation)
        if len(self.orientation_history) < self.smoothing_window:
            return current_orientation # Not enough history yet
        try:
            # Get non-unknown votes from the history window
            known_orientations = [o for o in self.orientation_history if o != "Unknown"]
            if not known_orientations:
                return "Unknown" # No known orientations in window

            # Find majority
            counter = Counter(known_orientations)
            smoothed_orientation, _ = counter.most_common(1)[0]
            return smoothed_orientation
        except Exception:
            return current_orientation # Fallback to raw if error

    def process_target_player(self, frame: np.ndarray,
                               player_detections_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Finds the target player, crops, runs pose, calculates/smooths orientation.

        Args:
            frame: The current video frame (NumPy array BGR).
            player_detections_df: DataFrame of ALL tracked players for this frame.
                                 Requires 'id', 'xmin', 'ymin', 'xmax', 'ymax'.

        Returns:
            Dictionary with results for the target player if found in this frame,
            otherwise None. Keys: 'keypoints', 'orientation_raw', 'orientation_smooth'.
            Keypoints are in ABSOLUTE frame coordinates.
        """

        if player_detections_df.empty or 'id' not in player_detections_df.columns:
             # Cannot find target if df is empty or missing id column
             smoothed_orientation = self._smooth_orientation("Unknown") # Update history with Unknown
             return None # Indicate target not processed this frame

        # Find the row for the target player ID
        target_row = player_detections_df[player_detections_df['id'] == self.target_id]

        if target_row.empty:
            # Target player not detected in this frame
            smoothed_orientation = self._smooth_orientation("Unknown") # Update history
            return None # Indicate target not found this frame

        # Extract bbox (use .iloc[0] as there should only be one row for the ID)
        try:
            bbox_data = target_row.iloc[0]
            x1 = int(bbox_data['xmin']); y1 = int(bbox_data['ymin'])
            x2 = int(bbox_data['xmax']); y2 = int(bbox_data['ymax'])
        except KeyError as e:
             print(f"ERROR: Missing bbox column in player_detections_df: {e}.")
             smoothed_orientation = self._smooth_orientation("Unknown") # Update history
             return None # Cannot proceed

        # --- 1. Crop Target Player ---
        frame_h, frame_w = frame.shape[:2]
        pad_x1 = max(0, x1 - self.crop_padding)
        pad_y1 = max(0, y1 - self.crop_padding)
        pad_x2 = min(frame_w, x2 + self.crop_padding)
        pad_y2 = min(frame_h, y2 + self.crop_padding)

        keypoints_abs = None
        raw_orientation = "Unknown"

        if pad_x1 < pad_x2 and pad_y1 < pad_y2:
            cropped_image = frame[pad_y1:pad_y2, pad_x1:pad_x2]

            # --- 2. Run Pose on Crop ---
            try:
                # Make sure crop has valid dimensions before predicting
                if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                    pose_results_crop = self.pose_model.predict(cropped_image, verbose=False, conf=self.pose_conf_threshold)
                else:
                    # print(f"Skipping pose on empty crop for track {self.target_id}") # Debug
                    pose_results_crop = [] # Empty result
            except Exception as e:
                print(f"Error running pose on crop for track {self.target_id}: {e}")
                pose_results_crop = [] 

            # --- 3. Extract Keypoints (Assume first/best detection in crop) ---
            if pose_results_crop and len(pose_results_crop) > 0 and pose_results_crop[0].keypoints is not None:
                keypoints_list = pose_results_crop[0].keypoints.data.cpu().numpy()
                if len(keypoints_list) > 0:
                    keypoints_rel = keypoints_list[0] # Shape [K, 3] relative to crop
                    
                    # --- Convert keypoints to FULL FRAME coordinates ---
                    keypoints_abs = keypoints_rel.copy()
                    keypoints_abs[:, 0] += pad_x1 # Add crop's top-left x
                    keypoints_abs[:, 1] += pad_y1 # Add crop's top-left y

                    # --- 4. Calculate Orientation ---
                    kpts_xy_abs = keypoints_abs[:, :2] # Use absolute coords
                    raw_orientation = self._calculate_single_orientation(kpts_xy_abs)
            # else: Keep raw_orientation as Unknown, keypoints_abs as None

        # --- 5. Smooth Orientation ---
        smoothed_orientation = self._smooth_orientation(raw_orientation)

        # --- 6. Return Results for Target Player ---
        return {
            'keypoints': keypoints_abs, # Absolute keypoints [K, 3] or None
            'orientation_raw': raw_orientation,
            'orientation_smooth': smoothed_orientation
        }