# run.py
# (Includes single-target orientation estimation and Norfair video writing)

import argparse
import cv2
import numpy as np
import PIL # For drawing if Player.draw uses it
from PIL import Image, ImageDraw, ImageFont # For drawing if Draw utils use PIL
import pandas as pd
from collections import deque, Counter # Needed by TargetOrientationEstimator, good to keep accessible
import math
from typing import Dict, Optional, List, Any
import os # For path handling

# --- Ultralytics and Norfair ---
from ultralytics import YOLO
from norfair import Tracker, Video, Detection # Import Detection if needed by Converter/utils
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean, iou_opt

# --- Your Custom Modules ---
# Ensure these paths and classes exist and function correctly
from inference import Converter, YoloV8 
from run_utils import ( 
    get_ball_detections,
    get_main_ball,
    get_player_detections, 
    update_motion_estimator,
)
from soccer import Player, Ball # Player class needs modification
# Team/Match/PassEvent not needed if only drawing players/ball + orientation
# from soccer import Match, Team 
# from soccer.pass_event import Pass 
from soccer.draw import AbsolutePath, Draw # Ensure Draw class/functions exist

# --- Import the Orientation Module ---
from target_orientation import TargetOrientationEstimator # Ensure this file exists

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Soccer Video Analytics with Target Orientation")
parser.add_argument(
    "--video",
    default="../FootieStats/data/HugoSaves2.mp4", # ADJUST PATH AS NEEDED
    type=str,
    help="Path to the input video",
)
parser.add_argument(
    "--ball_model", default="../FootieStats/weights/ball.pt", type=str, help="Path to the ball detection model" # ADJUST PATH AS NEEDED
)
parser.add_argument(
    "--player_model", default="../FootieStats/weights/players.pt", type=str, help="Path to the player detection model" # ADJUST PATH AS NEEDED
)
parser.add_argument(
    "--pose_model", default='yolov8x-pose-p6.pt', type=str, help="Path to the YOLOv8 Pose model"
)
parser.add_argument(
    "--target_id", type=int, default=1, help="Track ID of player for orientation analysis" # Require target ID
)
parser.add_argument( 
    "--output", 
    default="output/orientation_output.mp4", # Default output name
    type=str, 
    help="Path for output video file"
)
# Add other tunable parameters as arguments if desired
parser.add_argument("--player_conf", type=float, default=0.4, help="Confidence threshold for player detection")
parser.add_argument("--iou_thresh", type=float, default=0.8, help="IoU distance threshold for player tracker")
parser.add_argument("--hit_max", type=int, default=75, help="hit_counter_max for player tracker")
parser.add_argument("--smooth_window", type=int, default=7, help="Smoothing window size for orientation")
parser.add_argument("--pose_conf", type=float, default=0.4, help="Confidence threshold for pose detection")
parser.add_argument("--crop_pad", type=int, default=15, help="Padding pixels for player crop")

args = parser.parse_args()

# --- Video Setup with Norfair Writing ---
output_dir = os.path.dirname(args.output)
if output_dir: os.makedirs(output_dir, exist_ok=True) # Ensure directory exists

# Pass output_path directly to Norfair Video
video = Video(input_path=args.video, output_path=args.output) 
fps = video.video_capture.get(cv2.CAP_PROP_FPS)
frame_width = int(video.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Input video: {frame_width}x{frame_height} @ {fps:.2f} FPS")
print(f"Output video will be saved to: {args.output}")

# --- Model Initialization ---
print("Loading object detectors...")
# Pass confidence threshold from args
player_detector = YoloV8(model_path=args.player_model, conf=args.player_conf) 
ball_detector = YoloV8(model_path=args.ball_model)
print("Object detectors loaded.")

# --- Orientation Estimator Setup ---
FORWARD_VECTOR = (-1, 0.4) # Adjust based on your camera angle
estimator = TargetOrientationEstimator(
    target_id=args.target_id, 
    pose_model_path=args.pose_model, 
    forward_vector=FORWARD_VECTOR,
    smoothing_window=args.smooth_window,      
    pose_conf_threshold=args.pose_conf, 
    crop_padding_pixels=args.crop_pad   
)

# --- Tracker Initialization ---
print("Initializing trackers...")
player_tracker = Tracker(
    distance_function="iou_opt",      
    distance_threshold=args.iou_thresh, # From args         
    initialization_delay=3,         
    hit_counter_max=args.hit_max, # From args           
)
ball_tracker = Tracker( 
    distance_function=mean_euclidean, 
    distance_threshold=150,  # Keep fixed or make arg
    initialization_delay=4,   
    hit_counter_max=60,       
)

motion_estimator = MotionEstimator()
coord_transformations = None
# path = AbsolutePath() # Uncomment if drawing ball path

# --- Dictionary to hold active Player objects ---
active_players: Dict[int, Player] = {} 

# --- REMOVED Manual VideoWriter Setup ---

# --- MAIN PROCESSING LOOP ---
print("\nStarting video processing...")
frame_count = 0
try: 
    for i, frame in enumerate(video):
        frame_np = frame # Norfair Video yields BGR NumPy array
        frame_count = i

        # 1. Player and Ball Detection
        # Pass frame_np to detectors
        players_detections_raw = get_player_detections(player_detector, frame_np) 
        ball_detections_raw = get_ball_detections(ball_detector, frame_np)
        
        all_detections_for_motion = []
        if isinstance(players_detections_raw, list): all_detections_for_motion.extend(players_detections_raw)
        if isinstance(ball_detections_raw, list): all_detections_for_motion.extend(ball_detections_raw)

        # 2. Motion Estimation & Tracking Update
        coord_transformations = update_motion_estimator(
            motion_estimator=motion_estimator,
            detections=all_detections_for_motion,
            frame=frame_np,
        )
        player_track_objects = player_tracker.update(
            detections=players_detections_raw, coord_transformations=coord_transformations
        )
        ball_track_objects = ball_tracker.update(
            detections=ball_detections_raw, coord_transformations=coord_transformations
        )

        # 3. Convert Tracked Objects to DataFrame
        player_detections_tracked = Converter.TrackedObjects_to_Detections(player_track_objects)
        ball_detections_tracked = Converter.TrackedObjects_to_Detections(ball_track_objects)
        players_df = Converter.Detections_to_DataFrame(player_detections_tracked)

        # 4. Get Orientation for the Target Player
        target_orientation_data = None 
        if not players_df.empty:
            target_orientation_data = estimator.process_target_player(frame_np, players_df) 

        # 5. Update Active Player Objects Dictionary
        current_frame_track_ids = set()
        if not players_df.empty and 'id' in players_df.columns:
            for _, row in players_df.iterrows():
                track_id = int(row['id'])
                current_frame_track_ids.add(track_id)
                player_data_for_obj = row.to_dict() 
                
                if track_id not in active_players:
                    # Ensure Player class handles team=None and dict for detection
                    active_players[track_id] = Player(id=track_id, detection=player_data_for_obj, team=None) 
                else:
                    active_players[track_id].detection = player_data_for_obj

                # Update target player's orientation state
                if track_id == args.target_id:
                    if target_orientation_data:
                        active_players[track_id].keypoints = target_orientation_data['keypoints']
                        active_players[track_id].orientation = target_orientation_data['orientation_raw']
                        active_players[track_id].smoothed_orientation = target_orientation_data['orientation_smooth']
                    else:
                        # Ensure state is reset/maintained correctly if target detected but orientation fails
                        active_players[track_id].keypoints = None
                        active_players[track_id].orientation = "Unknown"
                        active_players[track_id].smoothed_orientation = estimator._smooth_orientation("Unknown")


        # 6. Remove Stale Player Objects
        lost_track_ids = set(active_players.keys()) - current_frame_track_ids
        for track_id in lost_track_ids: 
            if track_id in active_players: # Check before deleting
                 del active_players[track_id]
        
        # 7. Get Ball Object
        ball = get_main_ball(ball_detections_tracked)
        
        # 8. Draw Visualizations
        frame_final = frame_np # Default to original frame if drawing fails
        # Convert frame BGR NumPy to RGB PIL Image if Draw utils require it
        # If your Draw utils use OpenCV directly, skip this conversion
        frame_pil = PIL.Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))

        # Draw all players - Player.draw_players needs updated implementation
        frame_pil = Player.draw_players(
            players=list(active_players.values()),
            frame=frame_pil,
            id=True,
            draw_orientation=True, # Master switch
            target_player_id=args.target_id # Pass target ID
        )

        # Draw ball (optional)
        if ball:
            # Ensure ball.draw handles PIL Image and RGB color
            frame_pil = ball.draw(frame_pil) # Example green ball (RGB)

        # Convert back to BGR NumPy for Norfair's video.write()
        # draw arrow
        arrow_origin = np.array([50, 50]) 
        arrow_length = 40
        arrow_color = (0, 0, 255) # Red
        arrow_thickness = 10
        fwd_vec = np.array(FORWARD_VECTOR, dtype=float)
        norm = np.linalg.norm(fwd_vec)
        if norm > 1e-6:
            fwd_vec_normalized = fwd_vec / norm
        else:
            fwd_vec_normalized = np.array([0, -1.0]) # Default up

        arrow_end = arrow_origin + fwd_vec_normalized * arrow_length
        arrow_origin_int = tuple(arrow_origin.astype(int))
        arrow_end_int = tuple(arrow_end.astype(int))
        frame_pil = cv2.arrowedLine(np.array(frame_pil), arrow_origin_int, arrow_end_int, 
                        arrow_color, arrow_thickness, tipLength=0.3)
        
        frame_final = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            # frame_final remains frame_np if error occurred

        # 9. Write Frame using Norfair's Video object
        # This assumes Norfair handles potential errors internally
        video.write(frame_final) 

        # Print progress occasionally
        if (i + 1) % 100 == 0:
             print(f"Processed {i + 1} frames...")


    print(f"\nFinished processing {frame_count + 1} frames.")

finally: # Ensure resources are released
    print("Releasing video resources...")
    # No video.release() needed for Norfair object
    # No writer.release() needed
    print(f"Output video should be saved to {args.output} by Norfair.")
    print("Cleanup finished.")