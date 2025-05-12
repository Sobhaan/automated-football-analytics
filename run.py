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
import ast
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
from soccer import Match, Team 
from soccer.pass_event import Pass 
from soccer.draw import AbsolutePath, Draw # Ensure Draw class/functions exist
from inference import HSVClassifier, InertiaClassifier # Keep these
from inference.filters import filters

# --- Import the Orientation Module ---
from body_orientation import BodyOrientationEstimator # Ensure this file exists
from head_orientation import estimate_head_pose_angles, keypoints_pose
from auto_hsv_generator import generate_auto_hsv_filters

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
    "--body_orientation", default=False, type=str, help="Enable body orientation estimation" 
)
parser.add_argument(
    "--scanning", default=False, type=str, help="Enable scanning for target player"
)
parser.add_argument(
    "--target_id", type=int, default=21, help="Track ID of player for orientation analysis" # Require target ID
)
parser.add_argument(
    "--all_players", default=True, type=str, help="Track all players or only target player"
)
parser.add_argument( 
    "--output", 
    default="output/iou.mp4", # Default output name
    type=str, 
    help="Path for output video file"
)
args = parser.parse_args()

body_orientation = ast.literal_eval(str(args.body_orientation))
allplayers = ast.literal_eval(str(args.all_players))
scanning = ast.literal_eval(str(args.scanning))

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
player_conf = 0.4
player_detector = YoloV8(model_path=args.player_model, conf=player_conf) 
ball_detector = YoloV8(model_path=args.ball_model)
print("Object detectors loaded.")


# HSV Filter Arguments
temp_video_reader = cv2.VideoCapture(args.video)
frames_read = 0
initial_frames_for_filters = []
N_FRAMES_FOR_FILTERS = 10 # Number of frames to read for HSV filter generation
if temp_video_reader.isOpened():
    while frames_read < N_FRAMES_FOR_FILTERS:
        ret, frame = temp_video_reader.read()
        initial_frames_for_filters.append(frame)
        frames_read += 1
    temp_video_reader.release()
debug_hsv_image_output_path = "output/hsv_debug_images/" # Create this folder or set to None

auto_generated_filters = generate_auto_hsv_filters(
    player_detector_model=player_detector, # Your initialized player_detector
    frames=initial_frames_for_filters,
    team_names=["Team 1", "Team 2"], # Example team names
    referee_name=None, # Or None if you don't want a referee filter
    detections_conf_threshold=0.4,
    colors_to_define_per_team=2,
)
print(f"Colours: {auto_generated_filters}")
team1_color = [color['lower_hsv'] for team in auto_generated_filters if team.get('name') == 'Team 1' for color in team.get('colors', [])][1]
team2_color = [color['lower_hsv'] for team in auto_generated_filters if team.get('name') == 'Team 2' for color in team.get('colors', [])][1]

hsv_classifier = HSVClassifier(filters=auto_generated_filters)
classifier = InertiaClassifier(classifier=hsv_classifier, inertia=30) 

team1 = Team(
    name="Team 1",
    abbreviation="TM1",
    color=(255, 0, 0),
    board_color=(244, 86, 64),
    text_color=(255, 255, 255),
)
team2 = Team(
    name="Team 2",
    abbreviation="TM2",
    color=(255,255,0), # <-- BGR Green (adjust shade as needed)
    board_color=(153,153,0), 
    text_color=(0, 0, 0), 
)
teams = [team1, team2]
match = Match(home=team1, away=team2, fps=fps)
match.team_possession = team2

# --- Orientation Estimator Setup ---
# Add other tunable parameters as arguments if desired  
smooth_window = 7
pose_conf = 0.4
crop_pad = 15
FORWARD_VECTOR = (-1, 0.4) # Adjust based camera angle
estimator = BodyOrientationEstimator(
    target_id=args.target_id, 
    pose_model_path=args.pose_model, 
    forward_vector=FORWARD_VECTOR,
    smoothing_window=smooth_window,      
    pose_conf_threshold=pose_conf, 
    crop_padding_pixels=crop_pad   
)

# --- Tracker Initialization ---
print("Initializing trackers...")
iou_thresh = 0.8
hit_max = 75
player_tracker = Tracker(
    distance_function="dice",      
    distance_threshold=iou_thresh, # From args         
    initialization_delay=3,         
    hit_counter_max=hit_max, # From args           
)
ball_tracker = Tracker( 
    distance_function="iou", 
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
        if i == 10: break # For testing, remove in production

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
        # players_df = Converter.Detections_to_DataFrame(player_detections_tracked)

        if len(player_detections_tracked) > 0 :
            player_detections_classified = classifier.predict_from_detections(
                detections=player_detections_tracked, # Pass the list of Norfair/Converter Detection objects
                img=frame_np
            )
        else:
            player_detections_classified = []
        players_df = Converter.Detections_to_DataFrame(player_detections_classified)
        

        # 4. Get Orientation for the Target Player
        if body_orientation:
            target_orientation_data = None 
            if not players_df.empty:
                target_orientation_data = estimator.process_target_player(frame_np, players_df) 
        
        if scanning:
            if not players_df.empty:
                keypoints = keypoints_pose(frame_np, players_df, args.target_id)

        player_detections = classifier.predict_from_detections(
        detections=player_detections_tracked,
        img=frame_np,
        )
        count = 0
        # 5. Update Active Player Objects Dictionary
        current_frame_track_ids = set()
        if not players_df.empty and 'id' in players_df.columns:
            for _, row in players_df.iterrows():
                track_id = int(row['id'])
                current_frame_track_ids.add(track_id)
                player_data_for_obj = row.to_dict() 
                team = player_detections[count].data["classification"] 
                count += 1
                if track_id not in active_players and allplayers:
                    # Ensure Player class handles team=None and dict for detection
                    active_players[track_id] = Player(id=track_id, detection=player_data_for_obj, team=team) 
                elif track_id in active_players and allplayers:
                    active_players[track_id].detection = player_data_for_obj

                # Update target player's orientation state
                if track_id == args.target_id and body_orientation:
                    if not allplayers and track_id not in active_players:
                        active_players[track_id] = Player(id=track_id, detection=player_data_for_obj, team=team)
                    elif track_id in active_players and not allplayers:
                        active_players[track_id].detection = player_data_for_obj
                    if target_orientation_data:
                        active_players[track_id].keypoints = target_orientation_data['keypoints']
                        active_players[track_id].orientation = target_orientation_data['orientation_raw']
                        active_players[track_id].smoothed_orientation = target_orientation_data['orientation_smooth']
                    else:
                        # Ensure state is reset/maintained correctly if target detected but orientation fails
                        active_players[track_id].keypoints = None
                        active_players[track_id].orientation = "Unknown"
                        active_players[track_id].smoothed_orientation = estimator._smooth_orientation("Unknown")
                if track_id == args.target_id and scanning:
                    if not allplayers and track_id not in active_players:
                        active_players[track_id] = Player(id=track_id, detection=player_data_for_obj, team=team)
                    elif track_id in active_players and not allplayers:
                        active_players[track_id].detection = player_data_for_obj
                    angles = estimate_head_pose_angles(keypoints, conf_threshold=0.1, conf_yaw_scale=30.0)
                    if not np.isnan(angles[0][0]):
                        active_players[track_id].head_orientation = angles
                    print(f"Head orientation for player {track_id}: {active_players[track_id].head_orientation}, frame {frame_count}") # Debug print


        # 6. Remove Stale Player Objects
        lost_track_ids = set(active_players.keys()) - current_frame_track_ids
        for track_id in lost_track_ids: 
            if track_id in active_players: # Check before deleting
                 del active_players[track_id]
        
        # 7. Get Ball Object
        ball = get_main_ball(ball_detections_tracked)
        

        players = Player.from_detections(detections=player_detections, teams=teams)
        match.update(players, ball, i)


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
            draw_head_orientation=True, # Draw head orientation if scanning
            target_player_id=args.target_id, # Pass target ID
            team_colors=[team1_color, team2_color], # Pass team colors
        )

        # Draw ball (optional)
        if ball:
            # Ensure ball.draw handles PIL Image and RGB color
            frame_pil = ball.draw(frame_pil) # Example green ball (RGB)

        # Draw arrow for orientation - adjust as needed
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
        
        # Draw Frame Number
        frame_pil = cv2.putText(
            np.array(frame_pil), f"Frame: {frame_count}", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
        )
        
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