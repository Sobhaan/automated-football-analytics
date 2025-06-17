import numpy as np
import math
import cv2
from ultralytics import YOLO
from IPython.display import display
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
import mxnet as mx
import cv2
import numpy as np
from gluoncv.data.transforms.presets.ssd import transform_test
import time
import os
from collections import deque, Counter

def estimate_head_pose_angles(kps, conf_threshold=0.1, scan_angles_lists=None):
    """
    Enhanced version of your head pose estimation with better yaw calculation.
    This replaces your current estimate_head_pose_angles function.
    """
    # Keypoint indices (same as your original)
    NOSE = 0
    L_EYE = 1
    R_EYE = 2
    L_EAR = 3
    R_EAR = 4
    L_SHOULDER = 5
    R_SHOULDER = 6
    # Initialize with failure values
    yaw, pitch, roll = np.nan, np.nan, np.nan
    
    if kps is None or len(kps) < 7:
        return yaw, pitch, roll
    
    # Helper functions
    def is_valid(idx):
        return idx < len(kps) and kps[idx, 2] >= conf_threshold
    
    def get_point(idx):
        return kps[idx, :2]
    
    # Calculate visibility scores to determine view type
    frontal_score = sum([
        is_valid(NOSE) * 2,
        is_valid(L_EYE),
        is_valid(R_EYE),
    ]) / 4.0
    
    left_profile_score = sum([
        is_valid(L_EAR) * 2,
        is_valid(L_EYE),
        not is_valid(R_EYE),
        not is_valid(R_EAR),
    ]) / 5.0
    
    right_profile_score = sum([
        is_valid(R_EAR) * 2,
        is_valid(R_EYE),
        not is_valid(L_EYE),
        not is_valid(L_EAR),
    ]) / 5.0
    
    back_score = sum([
        is_valid(L_EAR),
        is_valid(R_EAR),
        not is_valid(L_EYE),
        not is_valid(R_EYE),
        not is_valid(NOSE),
    ]) / 5.0
    
    # Determine primary view
    view_scores = {
        'frontal': frontal_score,
        'left_profile': left_profile_score,
        'right_profile': right_profile_score,
        'back': back_score
    }
    primary_view = max(view_scores, key=view_scores.get)
    left_ear, right_ear = kps[3,2], kps[4,2]
    if left_ear < 0.7 or right_ear < 0.7:
        if left_ear > right_ear:
            primary_view = 'left_profile'
        else:
            primary_view = 'right_profile'

    # Estimate based on primary view with enhanced calculations
    if primary_view == 'frontal' and is_valid(NOSE) and is_valid(L_EYE) and is_valid(R_EYE):
        nose = get_point(NOSE)
        l_eye = get_point(L_EYE)
        r_eye = get_point(R_EYE)
        
        eye_center = (l_eye + r_eye) / 2.0
        eye_vector = r_eye - l_eye
        eye_distance = np.linalg.norm(eye_vector)
        
        if eye_distance > 1e-6:
            # Roll calculation (unchanged)
            roll = math.degrees(math.atan2(eye_vector[1], eye_vector[0]))
            if roll > 90:
                roll -= 180
            elif roll < -90:
                roll += 180
            
            # Enhanced yaw calculation
            eye_to_nose = nose - eye_center
            horizontal_ratio = eye_to_nose[0] / eye_distance
            horizontal_ratio = np.clip(horizontal_ratio, -1.0, 1.0)
            
            # Multi-method yaw estimation
            yaw_estimates = []
            weights = []
            
            # Method 1: Nose offset with non-linear mapping
            if abs(horizontal_ratio) < 0.3:
                nose_yaw = math.degrees(math.asin(horizontal_ratio)) * 1.2
            else:
                sign = np.sign(horizontal_ratio)
                abs_ratio = abs(horizontal_ratio)
                nose_yaw = sign * (30 * abs_ratio + 40 * abs_ratio**2 + 20 * abs_ratio**3)
            yaw_estimates.append(nose_yaw)
            weights.append(2.0)
            
            # Method 2: Eye confidence asymmetry
            eye_conf_diff = kps[R_EYE, 2] - kps[L_EYE, 2]
            if abs(eye_conf_diff) > 0.05:  # Lower threshold for head pose
                conf_adjustment = eye_conf_diff * 30.0 * (1.0 - abs(horizontal_ratio))
                yaw_estimates.append(nose_yaw + conf_adjustment)
                weights.append(1.0)
            
            # Method 3: Eye-ear visibility for refinement
            if is_valid(L_EAR) and is_valid(R_EAR):
                # When both ears visible in frontal, person is quite frontal
                ear_visibility_yaw = nose_yaw * 0.8  # Reduce extreme angles
                yaw_estimates.append(ear_visibility_yaw)
                weights.append(0.5)
            elif is_valid(L_EAR) and not is_valid(R_EAR):
                # Left ear visible, turning right
                yaw_estimates.append(abs(nose_yaw) * 1.2 if nose_yaw < 0 else nose_yaw)
                weights.append(0.8)
            elif is_valid(R_EAR) and not is_valid(L_EAR):
                # Right ear visible, turning left
                yaw_estimates.append(-abs(nose_yaw) * 1.2 if nose_yaw > 0 else nose_yaw)
                weights.append(0.8)
            
            # Combine estimates
            yaw = np.average(yaw_estimates, weights=weights)
            
            # Pitch calculation (refined)
            vertical_ratio = eye_to_nose[1] / eye_distance
            adjusted_pitch_ratio = (vertical_ratio - 0.4) * 1.8
            adjusted_pitch_ratio = np.clip(adjusted_pitch_ratio, -1.0, 1.0)
            pitch = -math.degrees(math.asin(adjusted_pitch_ratio)) * 0.8
    
    elif primary_view == 'left_profile':
        yaw = -90.0
        if is_valid(L_EAR) and is_valid(L_EYE):
            l_ear = get_point(L_EAR)
            l_eye = get_point(L_EYE)
            ear_eye_vec = l_eye - l_ear
            if np.linalg.norm(ear_eye_vec) > 1e-6:
                horizontal_offset = ear_eye_vec[0]
                yaw = -90.0 + horizontal_offset * 0.5
                yaw = np.clip(yaw, -135.0, -45.0)
        roll = estimate_roll_from_available_points(kps, conf_threshold)
        pitch = 0.0
    
    elif primary_view == 'right_profile':
        yaw = 90.0
        if is_valid(R_EAR) and is_valid(R_EYE):
            r_ear = get_point(R_EAR)
            r_eye = get_point(R_EYE)
            ear_eye_vec = r_eye - r_ear
            if np.linalg.norm(ear_eye_vec) > 1e-6:
                horizontal_offset = -ear_eye_vec[0]
                yaw = 90.0 + horizontal_offset * 0.5
                yaw = np.clip(yaw, 45.0, 135.0)
        roll = estimate_roll_from_available_points(kps, conf_threshold)
        pitch = 0.0
    
    elif primary_view == 'back':
        yaw = 180.0
        if is_valid(L_EAR) and is_valid(R_EAR):
            ear_conf_diff = kps[R_EAR, 2] - kps[L_EAR, 2]
            yaw = 180.0 + ear_conf_diff * 20.0
        roll = estimate_roll_from_available_points(kps, conf_threshold)
        pitch = 0.0
    
    else:
        # Transitional view - interpolate
        yaw = interpolate_yaw_from_visibility(view_scores)
        roll = estimate_roll_from_available_points(kps, conf_threshold)
        pitch = 0.0
    
    # Clamp values
    if not np.isnan(yaw):
        yaw = np.clip(yaw, -180.0, 180.0)
        pitch = np.clip(pitch, -90.0, 90.0)
        roll = np.clip(roll, -90.0, 90.0)

    yaw, pitch, roll, smooth_list = smooth_scanning([yaw, pitch, roll], scan_angles_lists)
    return yaw, pitch, roll, smooth_list

def estimate_roll_from_available_points(kps, conf_threshold):
    """Helper function to estimate roll from any available horizontal point pairs."""
    PAIRS = [(1, 2), (3, 4), (5, 6)]  # Eyes, Ears, Shoulders
    
    for left_idx, right_idx in PAIRS:
        if (left_idx < len(kps) and right_idx < len(kps) and
            kps[left_idx, 2] >= conf_threshold and kps[right_idx, 2] >= conf_threshold):
            
            left_point = kps[left_idx, :2]
            right_point = kps[right_idx, :2]
            vector = right_point - left_point
            
            if np.linalg.norm(vector) > 1e-6:
                roll = math.degrees(math.atan2(vector[1], vector[0]))
                if roll > 90:
                    roll -= 180
                elif roll < -90:
                    roll += 180
                return roll
    return 0.0

def interpolate_yaw_from_visibility(view_scores):
    """Smoothly interpolate yaw based on visibility scores."""
    yaw_values = {
        'frontal': 0.0,
        'left_profile': -90.0,
        'right_profile': 90.0,
        'back': 180.0
    }
    
    weighted_sum = 0.0
    weight_total = 0.0
    
    for view, score in view_scores.items():
        if score > 0.1:
            weighted_sum += yaw_values[view] * score
            weight_total += score
    
    if weight_total > 0:
        return weighted_sum / weight_total
    return 0.0

    
def keypoints_pose_solo(image, player, detector, pose_net, visualisation=False):
    # Extract bbox
    x1 = int(player.detection.points[0][0])
    y1 = int(player.detection.points[0][1])
    x2 = int(player.detection.points[1][0])
    y2 = int(player.detection.points[1][1])
    
    # Crop
    frame_h, frame_w = image.shape[:2]
    pad_x1 = max(0, x1)
    pad_y1 = max(0, y1)
    pad_x2 = min(frame_w, x2)
    pad_y2 = min(frame_h, y2)
    
    image_cropped = image[pad_y1:pad_y2, pad_x1:pad_x2]
    
    cv_img_rgb = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB)
    with mx.Context(mx.gpu(0)):
        mx_img = mx.nd.array(cv_img_rgb, dtype='uint8')
    
    # Transform for detection
    x, img = transform_test([mx_img], short=512)
    
    # Run detection
    class_IDs, scores, bounding_boxs = detector(x)
    # Pose estimation
    start_time = time.time()
    pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs, thr=0.1, ctx=mx.gpu())
    end_time = time.time()
    if pose_input is None:
        kpts = np.array([[0 for i in range(3)] for i in range(12)])
        return kpts, img
    # Run pose model
    predicted_heatmap = pose_net(pose_input)
    
    # Get coordinates
    pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
    coords = pred_coords[0].asnumpy()
    confs = confidence[0].asnumpy().squeeze(-1)
    
    # Stack
    kpts = np.concatenate([coords, confs[:, None]], axis=1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if visualisation:
        h, w = img.shape[:2]
        for idx, kp in enumerate(kpts):
        # Convert relative coordinates to absolute
            if idx < 7:
                x_rel, y_rel, conf = kp

                # Draw the keypoint
                # Calculate the end point for the shoulder_vector line
                start_draw_point_x = w // 2
                start_draw_point_y = h // 2
                start_draw_point = (start_draw_point_x, start_draw_point_y-10)
                cv2.circle(img, (int(x_rel), int(y_rel)), 1, (0,255,0), 1)
                # Optionally, you can label the keypoint index
                # cv2.putText(img, txt, start_draw_point, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(img, str(idx), (int(x_rel), int(y_rel)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
        img = cv2.resize(
                img,
                (int(w * 4), int(h * 4)),
                interpolation=cv2.INTER_LINEAR
            )
        cv2.imshow('Keypoints', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return kpts, img

def calculate_angular_difference(angle1, angle2):
    """
    Calculate the smallest angular difference between two angles.
    Handles the wrap-around at 0/360 degrees.
    
    Args:
        angle1, angle2: Angles in degrees
    
    Returns:
        Signed angular difference in degrees (-180 to 180)
    """
    diff = angle2 - angle1
    # Normalize to [-180, 180] range
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff

def count_scans(yaw_angles,
                      fps=30,
                      min_angular_velocity=45,
                      min_scan_angle=100,
                      max_scan_duration=1.0):

    # 1. Initial Data Preparation
    original_yaw_angles_for_debug = yaw_angles # Keep a reference if needed for deep debug
    yaw_angles_processed = [x[0] for x in yaw_angles]
    yaw = np.array(yaw_angles_processed)

    # 2. Calculating Angular Velocity
    angular_velocity = np.zeros(len(yaw))
    for i in range(1, len(yaw)):
        angle_diff = calculate_angular_difference(yaw[i-1], yaw[i])
        angular_velocity[i] = angle_diff * fps

    # 3. Smoothing Angular Velocity
    window_size = 3
    if len(angular_velocity) < window_size:
        angular_velocity_smooth = angular_velocity
    else:
        angular_velocity_smooth = np.convolve(angular_velocity,
                                              np.ones(window_size)/window_size,
                                              mode='same')

    # 4. Identifying High-Velocity Frames
    abs_smoothed_velocity = np.abs(angular_velocity_smooth)
    high_velocity_frames = np.where(abs_smoothed_velocity > min_angular_velocity)[0]

    # 5. Grouping and Validating Scans
    scan_count = 0
    detected_scan_frames_list = []
    i = 0
    while i < len(high_velocity_frames):
        scan_start_idx = high_velocity_frames[i]
        scan_end_idx = scan_start_idx

        # Try to extend this scan
        j = i + 1
        while j < len(high_velocity_frames):
            current_hv_frame = high_velocity_frames[j]
            prev_hv_frame = high_velocity_frames[j-1]

            # Condition 1: Gap between consecutive high-velocity frames
            frame_gap = current_hv_frame - prev_hv_frame
            gap_ok = frame_gap <= 3 # Max 2 non-high-velocity frames in between

            # Condition 2: Maximum scan duration
            # Duration is from the start of scan_start_idx to the end of current_hv_frame
            # Number of frames in duration = current_hv_frame - scan_start_idx + 1
            # Time = (Number of frames / fps)
            # Or, more simply, time diff between first and current high-vel frame
            current_scan_duration_frames = current_hv_frame - scan_start_idx # Number of frame intervals
            current_scan_duration_seconds = current_scan_duration_frames / fps # Duration in seconds
            duration_ok = current_scan_duration_seconds <= max_scan_duration

            if gap_ok and duration_ok:
                scan_end_idx = current_hv_frame
                j += 1
            else:
                print(f"[DEBUG]     Stopping scan extension for frame {current_hv_frame}.")
                if not gap_ok: print(f"[DEBUG]       Reason: Frame gap {frame_gap} > 3.")
                if not duration_ok: print(f"[DEBUG]       Reason: Scan duration {current_scan_duration_seconds:.2f}s > {max_scan_duration}s.")
                break
        
        segment_duration_seconds = (scan_end_idx - scan_start_idx) / fps # Duration based on frame indices

        # 6. Validate segment by total angle
        if scan_end_idx > scan_start_idx: # Ensure the segment has some length
            total_angle = 0
            # Summing angular differences for frames from scan_start_idx to scan_end_idx.
            # This involves yaw values from yaw[scan_start_idx] up to yaw[scan_end_idx+1].
            
            # Iterate through the original yaw values for this segment
            for k in range(scan_start_idx, scan_end_idx + 1):
                if k + 1 < len(yaw): # Ensure we don't go out of bounds for yaw[k+1]
                    diff = calculate_angular_difference(yaw[k], yaw[k+1])
                    total_angle += abs(diff)
                else:
                    # This case means scan_end_idx is the last frame of yaw data, so no yaw[k+1]
                    # The loop structure `range(scan_start_idx, scan_end_idx + 1)` means `k` will go up to `scan_end_idx`.
                    # The last difference considered is `yaw[scan_end_idx]` and `yaw[scan_end_idx+1]`.
                    # If `scan_end_idx` is the last frame, this calculation is naturally shorter.
                    # The total_angle will sum changes up to yaw[scan_end_idx-1] to yaw[scan_end_idx].
                    pass # No more differences to add if k is the last frame index

            if total_angle >= min_scan_angle:
                scan_count += 1
                detected_scan_frames_list.append((scan_start_idx, scan_end_idx))
                print(f"✅ [RESULT] Scan DETECTED & COUNTED: Frames {scan_start_idx} to {scan_end_idx}. Total Angle: {total_angle:.2f}°. Current scan_count: {scan_count}")
            else:
                print(f"❌ [RESULT] Segment REJECTED: Frames {scan_start_idx} to {scan_end_idx}. Reason: total_angle ({total_angle:.2f}°) < min_scan_angle ({min_scan_angle}°).")
        else: # scan_end_idx <= scan_start_idx
            print(f"❌ [RESULT] Segment REJECTED: Frames {scan_start_idx} to {scan_end_idx}. Reason: Segment too short or not extended (no high-velocity frames after start).")
        
        i = j # Move to the next unprocessed high-velocity frame

    print(f"\n--- Scanning Finished ---")
    print(f"Final scan_count: {scan_count}")
    return scan_count

def smooth_scanning(current_orientations, yaw_only_smooth) -> str:
        """Updates history and returns smoothed orientation for the target ID."""
        yaw_only = current_orientations[0]
        if yaw_only is not np.nan:
            yaw_only_smooth.append(yaw_only)
        if len(yaw_only_smooth) < 5:
            return current_orientations[0], current_orientations[1], current_orientations[2], yaw_only_smooth # Not enough history yet
        # Get non-unknown votes from the history window
        known_orientations = [o for o in yaw_only_smooth if o != 0]
        if not known_orientations:
            return 0,0,0, yaw_only_smooth # No known orientations in window

        # Find majority
        median_yaw = np.median(yaw_only_smooth)
        smoothed_orientation = median_yaw
        return smoothed_orientation, current_orientations[1], current_orientations[2], yaw_only_smooth  # Return pitch and roll unchanged