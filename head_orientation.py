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

def estimate_head_pose_angles(kps, conf_threshold=0.1):
    """
    Estimates head pose angles from a single frame's keypoints with improved calculations.
    
    Args:
        kps (numpy.ndarray): Keypoint array of shape (17, 3) where each row is [x, y, confidence].
                            Expected keypoint order (COCO format):
                            0: Nose
                            1: Left Eye
                            2: Right Eye
                            3: Left Ear
                            4: Right Ear
                            5: Left Shoulder
                            6: Right Shoulder
                            ... (remaining body keypoints)
        conf_threshold (float): Minimum confidence threshold for using a keypoint.
    
    Returns:
        tuple: (yaw, pitch, roll) in degrees, or (np.nan, np.nan, np.nan) if estimation fails.
    """
    # Initialize with failure values
    yaw, pitch, roll = np.nan, np.nan, np.nan
    
    # Keypoint indices for clarity
    NOSE = 0
    L_EYE = 1
    R_EYE = 2
    L_EAR = 3
    R_EAR = 4
    L_SHOULDER = 5
    R_SHOULDER = 6
    
    # Validate input
    if kps is None or len(kps) < 7:
        return yaw, pitch, roll
    
    # Helper function to check if a keypoint is valid
    def is_valid(idx):
        return idx < len(kps) and kps[idx, 2] >= conf_threshold
    
    # Extract 2D positions for valid keypoints
    def get_point(idx):
        return kps[idx, :2]
    
    # --- Frontal View Estimation (most reliable) ---
    if is_valid(NOSE) and is_valid(L_EYE) and is_valid(R_EYE):
        nose = get_point(NOSE)
        l_eye = get_point(L_EYE)
        r_eye = get_point(R_EYE)
        
        # Calculate eye center and inter-eye distance
        eye_center = (l_eye + r_eye) / 2.0
        eye_vector = r_eye - l_eye
        eye_distance = np.linalg.norm(eye_vector)
        
        if eye_distance > 1e-6:  # Avoid division by zero
            # Roll: angle of the eye line from horizontal
            roll = math.degrees(math.atan2(eye_vector[1], eye_vector[0]))
            # Normalize roll to [-90, 90] range
            if roll > 90:
                roll -= 180
            elif roll < -90:
                roll += 180
            
            # Vector from eye center to nose
            eye_to_nose = nose - eye_center
            
            # Pitch estimation with improved calculation
            # Account for typical facial proportions where nose is below eyes
            vertical_ratio = eye_to_nose[1] / eye_distance
            # In neutral pose, nose is typically 0.3-0.5 eye distances below eye line
            # Adjust for this baseline offset
            adjusted_pitch_ratio = (vertical_ratio - 0.4) * 2.0
            adjusted_pitch_ratio = np.clip(adjusted_pitch_ratio, -1.0, 1.0)
            # Scale down slightly as faces don't tilt as extremely as the math suggests
            pitch = -math.degrees(math.asin(adjusted_pitch_ratio)) * 0.7
            
            # Yaw estimation with improved scaling
            horizontal_ratio = eye_to_nose[0] / eye_distance
            horizontal_ratio = np.clip(horizontal_ratio, -1.0, 1.0)
            # The original formula underestimates yaw, so we scale it up
            # This factor was determined empirically to match typical head turn angles
            yaw = math.degrees(math.asin(horizontal_ratio)) * 1.5
            
            # Confidence-based yaw refinement
            # When the head turns, the farther eye often has lower detection confidence
            eye_conf_diff = kps[R_EYE, 2] - kps[L_EYE, 2]
            # Scale the adjustment based on how frontal the face is
            # (less adjustment when face is more turned)
            conf_adjustment = eye_conf_diff * 25.0 * (1.0 - abs(horizontal_ratio))
            yaw += conf_adjustment
    
    # --- Back View Estimation ---
    elif is_valid(L_EAR) and is_valid(R_EAR) and not is_valid(L_EYE) and not is_valid(R_EYE):
        # Person is facing away from camera
        l_ear = get_point(L_EAR)
        r_ear = get_point(R_EAR)
        
        # Use ear confidence difference to estimate slight head turns
        ear_conf_diff = kps[R_EAR, 2] - kps[L_EAR, 2]
        # Base yaw is 180° (facing away) with adjustment based on which ear is clearer
        yaw = 180.0 + ear_conf_diff * 30.0
        
        # Roll from ear positions
        ear_vector = r_ear - l_ear
        if np.linalg.norm(ear_vector) > 1e-6:
            roll = math.degrees(math.atan2(ear_vector[1], ear_vector[0]))
            if roll > 90:
                roll -= 180
            elif roll < -90:
                roll += 180
        else:
            roll = 0.0
        
        # Pitch is very difficult to estimate from back view
        pitch = 0.0
    
    # --- Left Profile Estimation ---
    elif is_valid(L_EAR) and not is_valid(R_EAR) and not is_valid(R_EYE):
        # Person is looking to their right (left side visible to camera)
        yaw = -90.0  # Base angle for left profile
        
        # Try to refine using eye position if available
        if is_valid(L_EYE):
            l_ear = get_point(L_EAR)
            l_eye = get_point(L_EYE)
            ear_eye_vec = l_eye - l_ear
            
            # Horizontal component indicates how much they're turned toward/away from camera
            if np.linalg.norm(ear_eye_vec) > 1e-6:
                # If eye is forward of ear, person is turning slightly toward camera
                horizontal_offset = ear_eye_vec[0]
                yaw += horizontal_offset * 0.3  # Scale factor for adjustment
                
                # Roll from ear-eye angle
                roll = math.degrees(math.atan2(ear_eye_vec[1], ear_eye_vec[0]))
            else:
                roll = 0.0
        else:
            roll = 0.0
        
        pitch = 0.0  # Hard to estimate from profile
    
    # --- Right Profile Estimation ---
    elif is_valid(R_EAR) and not is_valid(L_EAR) and not is_valid(L_EYE):
        # Person is looking to their left (right side visible to camera)
        yaw = 90.0  # Base angle for right profile
        
        # Try to refine using eye position if available
        if is_valid(R_EYE):
            r_ear = get_point(R_EAR)
            r_eye = get_point(R_EYE)
            ear_eye_vec = r_eye - r_ear
            
            # Horizontal component indicates how much they're turned toward/away from camera
            if np.linalg.norm(ear_eye_vec) > 1e-6:
                # If eye is forward of ear, person is turning slightly toward camera
                horizontal_offset = -ear_eye_vec[0]  # Negative because we're on right side
                yaw += horizontal_offset * 0.3  # Scale factor for adjustment
                
                # Roll from ear-eye angle (adjusted for right side)
                roll = math.degrees(math.atan2(ear_eye_vec[1], ear_eye_vec[0])) - 180.0
                if roll < -90:
                    roll += 180
            else:
                roll = 0.0
        else:
            roll = 0.0
        
        pitch = 0.0  # Hard to estimate from profile
    
    # --- Transitional/Partial View ---
    elif is_valid(L_EAR) and is_valid(R_EAR):
        # Both ears visible but maybe only one eye - person is partially turned
        l_eye_valid = is_valid(L_EYE)
        r_eye_valid = is_valid(R_EYE)
        
        if l_eye_valid and r_eye_valid:
            # Actually a frontal view - recursively call frontal logic
            # (In practice, this case should have been caught by the first condition)
            yaw = 0.0  # Approximate as frontal
        elif l_eye_valid and not r_eye_valid:
            # Turning right (left eye visible, right eye not)
            yaw = -45.0  # Between frontal and left profile
        elif r_eye_valid and not l_eye_valid:
            # Turning left (right eye visible, left eye not)
            yaw = 45.0  # Between frontal and right profile
        else:
            # No eyes visible but both ears - closer to back view
            # Use ear confidences to estimate direction
            ear_conf_diff = kps[R_EAR, 2] - kps[L_EAR, 2]
            yaw = 135.0 + ear_conf_diff * 45.0  # Range from ~90 to ~180
        
        # Roll from ear positions
        l_ear = get_point(L_EAR)
        r_ear = get_point(R_EAR)
        ear_vector = r_ear - l_ear
        if np.linalg.norm(ear_vector) > 1e-6:
            roll = math.degrees(math.atan2(ear_vector[1], ear_vector[0]))
            if roll > 90:
                roll -= 180
            elif roll < -90:
                roll += 180
        else:
            roll = 0.0
        
        pitch = 0.0  # Difficult to estimate in transition
    
    # Clamp all values to reasonable ranges
    if not np.isnan(yaw):
        yaw = np.clip(yaw, -180.0, 180.0)
        pitch = np.clip(pitch, -90.0, 90.0)
        roll = np.clip(roll, -90.0, 90.0)
    return yaw, pitch, roll

    
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
    mx_img = mx.nd.array(cv_img_rgb, dtype='uint8')
    
    # Transform for detection
    x, img = transform_test([mx_img], short=512)
    
    # Run detection
    class_IDs, scores, bounding_boxs = detector(x)
    # Pose estimation
    start_time = time.time()
    pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs, thr=0.1, ctx=mx.gpu())
    end_time = time.time()
    print(f"Pose input preparation time: {end_time - start_time:.2f} seconds")
    
    # Run pose model
    predicted_heatmap = pose_net(pose_input)
    
    # Get coordinates
    pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
    end_time = time.time()
    coords = pred_coords[0].asnumpy()
    confs = confidence[0].asnumpy().squeeze(-1)
    
    # Stack
    kpts = np.concatenate([coords, confs[:, None]], axis=1)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # h, w = img.shape[:2]
    # print(kpts)
    # for idx, kp in enumerate(kpts):
    # # Convert relative coordinates to absolute
    #     if idx < 5:
    #         x_rel, y_rel, conf = kp

    #         # Draw the keypoint
    #         # Calculate the end point for the shoulder_vector line
    #         start_draw_point_x = w // 2
    #         start_draw_point_y = h // 2
    #         cv2.circle(img, (int(x_rel), int(y_rel)), 1, (0,255,0), 1)
    #         # Optionally, you can label the keypoint index

    #         #cv2.putText(img, txt, start_draw_point, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
    #         cv2.putText(img, str(idx), (int(x_rel), int(y_rel)),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)

    # cv2.imshow('Keypoints', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return kpts

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
                min_angular_velocity=90,  # degrees per second
                min_scan_angle=45,        # minimum total angle for a scan
                max_scan_duration=1.0):   # maximum duration of a scan in seconds
    """
    Count the number of scans in a sequence of yaw angles.
    
    Args:
        yaw_angles: List or array of yaw values (degrees) for each frame
        fps: Frames per second of the data
        min_angular_velocity: Minimum angular velocity (deg/s) to consider as scanning
        min_scan_angle: Minimum total angular displacement for a valid scan
        max_scan_duration: Maximum time (seconds) for a single scan motion
        
    Returns:
        int: Number of scans detected
    """
    
    # Convert to numpy array for easier manipulation
    yaw_angles = [x[0] for x in yaw_angles]
    yaw = np.array(yaw_angles)
    

    # Calculate angular velocity for each frame
    angular_velocity = np.zeros(len(yaw))
    
    for i in range(1, len(yaw)):
        # Calculate angular difference accounting for wrap-around
        angle_diff = calculate_angular_difference(yaw[i-1], yaw[i])
        # Convert to angular velocity (degrees per second)
        angular_velocity[i] = angle_diff * fps
    
    # Smooth the angular velocity to reduce noise
    # Using a simple moving average with window size of 3
    window_size = 3
    angular_velocity_smooth = np.convolve(angular_velocity, 
                                         np.ones(window_size)/window_size, 
                                         mode='same')
    
    # Find frames where angular velocity exceeds our threshold
    # These are potential scanning movements
    high_velocity_frames = np.where(np.abs(angular_velocity_smooth) > min_angular_velocity)[0]
    
    if len(high_velocity_frames) == 0:
        return 0
    
    # Group consecutive high-velocity frames into scan events
    scan_count = 0
    i = 0
    
    while i < len(high_velocity_frames):
        # Start of a potential scan
        scan_start_idx = high_velocity_frames[i]
        scan_end_idx = scan_start_idx
        
        # Find all consecutive frames that belong to this scan
        j = i + 1
        while j < len(high_velocity_frames):
            # Check if this frame is close enough to be part of the same scan
            if (high_velocity_frames[j] - high_velocity_frames[j-1] <= 3 and  # Allow small gaps
                (high_velocity_frames[j] - scan_start_idx) / fps <= max_scan_duration):
                scan_end_idx = high_velocity_frames[j]
                j += 1
            else:
                break
        
        # Calculate total angular displacement for this potential scan
        if scan_end_idx > scan_start_idx:
            total_angle = 0
            for k in range(scan_start_idx, min(scan_end_idx + 1, len(yaw) - 1)):
                total_angle += abs(calculate_angular_difference(yaw[k], yaw[k+1]))
            
            # Check if this qualifies as a scan
            if total_angle >= min_scan_angle:
                scan_count += 1
        
        # Move to next unprocessed high-velocity frame
        i = j
    
    return scan_count