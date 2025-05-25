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

def estimate_head_pose_angles(keypoints_per_frame, conf_threshold=0.1, conf_yaw_scale=30.0):
    """
    Estimates head pose angles, attempting different heuristics based on view.

    Args:
        keypoints_per_frame (list): List of keypoint arrays, each shape (1, N, 3).
                                    Indices: 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 
                                             5:LShoulder, 6:RShoulder.
        conf_threshold (float): Minimum confidence for keypoints.
        conf_yaw_scale (float): Yaw adjustment scale based on eye confidence diff.

    Returns:
        list: List of (yaw, pitch, roll) tuples in degrees, or (nan, nan, nan).
    """
    all_angles = []
    
    # Indices
    NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER = 0, 1, 2, 3, 4, 5, 6
    MIN_KP_INDEX = max(NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER)

    for kps_frame_data in keypoints_per_frame:
        yaw, pitch, roll = np.nan, np.nan, np.nan # Default to failure

        # --- Input Handling & Basic Checks ---
        if not isinstance(kps_frame_data, np.ndarray) or kps_frame_data.ndim != 3 or kps_frame_data.shape[0] < 1:
            all_angles.append((yaw, pitch, roll)) 
            continue
        kps = kps_frame_data[0] # Use first detected person

        if kps.ndim != 2 or kps.shape[1] != 3 or kps.shape[0] <= MIN_KP_INDEX:
             all_angles.append((yaw, pitch, roll))
             continue

        # --- Get Keypoint Data ---
        kp_data = {i: {'pt': kps[i][:2], 'conf': kps[i][2]} for i in range(kps.shape[0])}
        
        def is_ok(kp_index):
            return kp_index in kp_data and kp_data[kp_index]['conf'] >= conf_threshold

        # --- View Classification & Calculation ---
        view_type = "undetermined"

        # Frontal View Logic
        if is_ok(NOSE) and is_ok(L_EYE) and is_ok(R_EYE):
            view_type = "frontal"
            nose_pt = kp_data[NOSE]['pt']
            l_eye_pt = kp_data[L_EYE]['pt']
            r_eye_pt = kp_data[R_EYE]['pt']
            
            eye_midpoint = (l_eye_pt + r_eye_pt) / 2.0
            eye_vector = r_eye_pt - l_eye_pt
            eye_dist = np.linalg.norm(eye_vector)

            if eye_dist > 1e-6:
                eye_mid_to_nose_vector = nose_pt - eye_midpoint
                
                # Roll
                roll_rad = math.atan2(eye_vector[1], eye_vector[0])
                roll = math.degrees(roll_rad)
                roll = roll if roll <= 90 else roll - 180 
                roll = roll if roll >= -90 else roll + 180

                # Pitch
                pitch_ratio = np.clip(eye_mid_to_nose_vector[1] / eye_dist, -1.0, 1.0)
                pitch = -math.degrees(math.asin(pitch_ratio)) 

                # Yaw (Geometric + Confidence)
                yaw_ratio = np.clip(eye_mid_to_nose_vector[0] / eye_dist, -1.0, 1.0)
                yaw_geom = math.degrees(math.asin(yaw_ratio))
                
                yaw_adjustment = 0.0
                if conf_yaw_scale > 0:
                    eye_conf_diff = kp_data[R_EYE]['conf'] - kp_data[L_EYE]['conf']
                    yaw_adjustment = -eye_conf_diff * conf_yaw_scale * (1.0 - abs(yaw_ratio))
                
                yaw = yaw_geom + yaw_adjustment
            
        # Back View Logic (Simplified)
        elif is_ok(L_EAR) and is_ok(R_EAR) and is_ok(L_SHOULDER) and is_ok(R_SHOULDER) and not (is_ok(L_EYE) or is_ok(R_EYE)):
            view_type = "back"
            l_ear_pt = kp_data[L_EAR]['pt']
            r_ear_pt = kp_data[R_EAR]['pt']
            l_shoulder_pt = kp_data[L_SHOULDER]['pt']
            r_shoulder_pt = kp_data[R_SHOULDER]['pt']

            yaw = 180.0 # Assume facing directly away
            pitch = 0.0 # Very hard to estimate pitch from back reliably

            # Roll based on ear tilt relative to shoulder tilt
            ear_vector = r_ear_pt - l_ear_pt
            shoulder_vector = r_shoulder_pt - l_shoulder_pt
            norm_ear = np.linalg.norm(ear_vector)
            norm_shoulder = np.linalg.norm(shoulder_vector)

            if norm_ear > 1e-6 and norm_shoulder > 1e-6:
                 roll_rad_ear = math.atan2(ear_vector[1], ear_vector[0])
                 roll_rad_shoulder = math.atan2(shoulder_vector[1], shoulder_vector[0])
                 roll_rad = roll_rad_ear - roll_rad_shoulder
                 roll_rad = (roll_rad + math.pi) % (2 * math.pi) - math.pi # Normalize
                 roll = math.degrees(roll_rad)
            else:
                 roll = 0.0 # Default roll if distances are too small

        # Profile View Logic (Simplified Heuristics)
        # Add more checks (e.g., nose visibility) if needed for better classification
        elif is_ok(L_EAR) and not is_ok(R_EAR) and not is_ok(R_EYE): # Likely Left Profile (person looking right)
             view_type = "left_profile"
             yaw = -90.0
             pitch = 0.0
             # Try roll based on L_Eye / L_Ear if L_Eye visible
             if is_ok(L_EYE):
                 l_eye_pt = kp_data[L_EYE]['pt']
                 l_ear_pt = kp_data[L_EAR]['pt']
                 ear_eye_vec = l_eye_pt - l_ear_pt
                 if np.linalg.norm(ear_eye_vec) > 1e-6:
                      # This assumes ear-eye line is horizontal in neutral pose
                      roll = math.degrees(math.atan2(ear_eye_vec[1], ear_eye_vec[0])) 
                 else: roll = 0.0
             else: roll = 0.0 # Default if eye not visible

        elif is_ok(R_EAR) and not is_ok(L_EAR) and not is_ok(L_EYE): # Likely Right Profile (person looking left)
             view_type = "right_profile"
             yaw = 90.0
             pitch = 0.0
             # Try roll based on R_Eye / R_Ear if R_Eye visible
             if is_ok(R_EYE):
                 r_eye_pt = kp_data[R_EYE]['pt']
                 r_ear_pt = kp_data[R_EAR]['pt']
                 ear_eye_vec = r_eye_pt - r_ear_pt
                 if np.linalg.norm(ear_eye_vec) > 1e-6:
                      roll = math.degrees(math.atan2(ear_eye_vec[1], ear_eye_vec[0]))
                 else: roll = 0.0
             else: roll = 0.0

        # Clamp results before appending
        yaw = np.clip(yaw, -180.0, 180.0)
        pitch = np.clip(pitch, -90.0, 90.0)
        roll = np.clip(roll, -90.0, 90.0)
        
        all_angles.append((yaw, pitch, roll))
        # print(f"Frame {len(all_angles)-1}: View={view_type}, Angles=({yaw:.1f}, {pitch:.1f}, {roll:.1f})") # Debug print

    return all_angles


def keypoints_pose(image, player_detections_df, target_id, model='yolov8x-pose-p6.pt', visualisation=False):
    target_row = player_detections_df[player_detections_df['id'] == target_id]
    if player_detections_df.empty or 'id' not in player_detections_df.columns or target_row.empty:
        return None
    bbox_data = target_row.iloc[0]
    x1 = int(bbox_data['xmin']); y1 = int(bbox_data['ymin'])
    x2 = int(bbox_data['xmax']); y2 = int(bbox_data['ymax'])
    crop_padding = 10
    frame_h, frame_w = image.shape[:2]
    pad_x1 = max(0, x1 - crop_padding)
    pad_y1 = max(0, y1 - crop_padding)
    pad_x2 = min(frame_w, x2 + crop_padding)
    pad_y2 = min(frame_h, y2 + crop_padding)

    if pad_x1 < pad_x2 and pad_y1 < pad_y2:
        image = image[pad_y1:pad_y2, pad_x1:pad_x2]
        model = YOLO(model)
        # images, res = crop_focused_player(target_player_id=target_player_id, video_path=video_path)
        keypoint_list = []
        results = model(image)
        # Create visualization without text
        output_image = image.copy()
        
        keypoints = results[0].keypoints.data.cpu().numpy()
        
        if len(list(keypoints)) == 1:
            print("No keypoints")
            results = model(image, conf=0.01)
            keypoints = results[0].keypoints.data.cpu().numpy()
        if len(list(keypoints)) == 1:
            print("Still no keypoints")
            results = model(image, conf=0.003)
            keypoints = results[0].keypoints.data.cpu().numpy()
        keypoint_list.append(keypoints)
        # Keypoint names
        keypoint_names = [
            "nose", "l_eye", "r_eye", "l_ear", "r_ear", 
            "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
            "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"
        ]

        # Draw keypoints with small labels
        if visualisation:
            for person in keypoints:
                for i, kp in enumerate(person):
                    x, y, conf = kp
                    print("number: ", i, "coords:" ,x, y, "confidence: ", conf, "label name: ",keypoint_names[i])
                    if conf > 0.5:  # Only draw high-confidence keypoints
                        # Draw keypoint
                        cv2.circle(output_image, (int(x), int(y)), 1, (0, 255, 0), -1)
                        # Add small label
                        label = keypoint_names[i]
                        cv2.putText(output_image, str(i), (int(x)+1, int(y)+1), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.18, (0, 0, 255), 1)
                        
            window_name = "Pose Keypoints"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 640, 640)
            cv2.imshow(window_name, output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite('4kbbox.png', output_image)

            # cv2.imwrite('output/orientationorg.jpg', output_image)
        return keypoint_list
    
def keypoints_pose_solo(image, player, model='yolov8x-pose-p6.pt', visualisation=False):    
    detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
    pose_net = model_zoo.get_model('simple_pose_resnet152_v1d', pretrained=True)
    detector.reset_class(["person"], reuse_weights=['person'])
    
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
    pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs, thr=0.1)
    
    # Run pose model
    predicted_heatmap = pose_net(pose_input)
    
    # Get coordinates
    pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
    
    coords = pred_coords[0].asnumpy()
    confs = confidence[0].asnumpy().squeeze(-1)
    
    # Stack
    kpts = np.concatenate([coords, confs[:, None]], axis=1)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # h, w = img.shape[:2]
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