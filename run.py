# run.py
# (Includes single-target orientation estimation and Norfair video writing)
import warnings
warnings.filterwarnings("ignore")
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
from IPython.display import display # For displaying DataFrames in Jupyter
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
from soccer.draw import AbsolutePath, Draw, draw_arrow # Ensure Draw class/functions exist
# from inference import HSVClassifier, InertiaClassifier # Keep these
from inference.hsv_classifier import HSVClassifier # Ensure this file exists
from inference.inertia_classifier import InertiaClassifier # Ensure this file exists
from inference.filters import filters
import random
# --- Import the Orientation Module ---
from body_orientation import BodyOrientationEstimator # Ensure this file exists
from head_orientation import estimate_head_pose_angles, keypoints_pose
from auto_hsv_generator import generate_auto_hsv_filters
import time
from output import generate_output_df, update_lists
from forward_vector import forward_vector
from auto_id import select_target_player_id_on_first_frame

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Soccer Video Analytics with Target Orientation")
parser.add_argument(
    "--video",
    default="../FootieStats/data/alnassr.mp4", # ADJUST PATH AS NEEDED
    type=str,
    help="Path to the input video",
)
parser.add_argument(
    "--ball_model", default="../FootieStats/weights/ball.engine", type=str, help="Path to the ball detection model" # ADJUST PATH AS NEEDED
)
parser.add_argument(
    "--player_model", default="../FootieStats/weights/players.engine", type=str, help="Path to the player detection model" # ADJUST PATH AS NEEDED
)
parser.add_argument(
    "--pose_model", default='yolov8x-pose-p6.pt', type=str, help="Path to the YOLOv8 Pose model"
)
parser.add_argument(
    "--body_orientation", default=True, type=str, help="Enable body orientation estimation" 
)
parser.add_argument(
    "--scanning", default=False, type=str, help="Enable scanning for target player"
)
# parser.add_argument(
#     "--target_id", type=int, default=3, help="Track ID of player for orientation analysis" # Require target ID
# )
parser.add_argument(
    "--all_players", default=True, type=str, help="Track all players or only target player"
)
parser.add_argument(
    "--automatic_teams", default=True, type=str, help="Enable automatic team color detection"
)
parser.add_argument(
    "--pressure", default=True, type=str, help="Enable pressure detection"
)
parser.add_argument(
    "--visualize", default=True, type=str, help="Enable visualization of players and ball"
)

parser.add_argument( 
    "--output", 
    default="output/asdasdasdasddsasasadas223.mp4", # Default output name
    type=str, 
    help="Path for output video file"
)
args = parser.parse_args()

body_orientation = ast.literal_eval(str(args.body_orientation))
allplayers = ast.literal_eval(str(args.all_players))
scanning = ast.literal_eval(str(args.scanning))
pressure = ast.literal_eval(str(args.pressure))
automatic_teams = ast.literal_eval(str(args.automatic_teams))
visualize = ast.literal_eval(str(args.visualize))

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
if args.automatic_teams:
    temp_video_reader = cv2.VideoCapture(args.video)
    total_frames = int(temp_video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    random_frame_indices = sorted(random.sample(range(total_frames), 30))
    initial_frames_for_filters = []
    for frame_index in random_frame_indices:
        temp_video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = temp_video_reader.read()
        if ret:
            initial_frames_for_filters.append(frame)


    auto_generated_filters = generate_auto_hsv_filters(
        player_detector_model=player_detector, # Your initialized player_detector
        frames=initial_frames_for_filters,
        team_names=["Team 1", "Team 2"], # Example team names
        referee_name=None, # Or None if you don't want a referee filter
    )
    
    filter = auto_generated_filters
    #filter = filters
    hsv_classifier = HSVClassifier(filters=filter)
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
# match.team_possession = team2
# --- Orientation Estimator Setup ---

# --- Tracker Initialization ---
print("Initializing trackers...")
iou_thresh = 0.8
hit_max = 100
player_tracker = Tracker(
    distance_function="iou",      
    distance_threshold=iou_thresh, # From args         
    initialization_delay=2,         
    hit_counter_max=hit_max, # From args           
)
ball_tracker = Tracker( 
    distance_function="iou", 
    distance_threshold=150,  # Keep fixed or make arg
    initialization_delay=0,   
    hit_counter_max=60,       
)

motion_estimator = MotionEstimator()
coord_transformations = None
# path = AbsolutePath() # Uncomment if drawing ball path

id = select_target_player_id_on_first_frame(
        current_video_obj=video,
        player_detector_model=player_detector,
        tracker_obj=player_tracker, 
        motion_estimator_obj=motion_estimator, # Use the instance
        converter_module=Converter, # Pass the imported class/module
        player_module=Player,       # Pass the imported class
        team_module=Team,            # Pass the imported class
        tracker_init_delay_for_selection_frames = player_tracker.initialization_delay
    )
print(id)

# Add other tunable parameters as arguments if desired  
smooth_window = 10
pose_conf = 0.1
crop_pad = 0
#FORWARD_VECTOR = forward_vector() # Adjust based camera angle
FORWARD_VECTOR = (-0.9978, -0.0665)
print(f"FORWARD_VECTOR: {FORWARD_VECTOR}")
estimator = BodyOrientationEstimator(
    target_id=id, 
    pose_model_path=args.pose_model, 
    forward_vector=FORWARD_VECTOR,
    smoothing_window=smooth_window,      
    pose_conf_threshold=pose_conf, 
    crop_padding_pixels=crop_pad   
)

target_missing_frames_count = 0
is_in_reselection_phase = False # Flag to manage reselection state
wait_frames_countdown = 0      # Counter for the "wait" option

# --- Dictionary to hold active Player objects ---
active_players: Dict[int, Player] = {}
old_passes = [] 
passes_list = []
target_pass_list = []

# --- Value lists ---
body_position_list = []
pressure_list = []
turnable_list = []
number_of_scans_list = []

# --- Final Target Values ---
time_list = []
target_bp = []
target_pressure = []
target_turnable = []
target_number_of_scans = []

# --- MAIN PROCESSING LOOP ---
print("\nStarting video processing...")
frame_count = 0
try: 
    for i, frame in enumerate(video):
        frame_np = frame # Norfair Video yields BGR NumPy array
        print("\n")
        print(f"Processing frame {i}...") # Debug print
        print(is_in_reselection_phase)

        if i == 1050: break
        if wait_frames_countdown > 0:
            print(f"Waiting... {wait_frames_countdown} frames remaining.")
            wait_frames_countdown -= 1
            if wait_frames_countdown == 0:
                is_in_reselection_phase = True # Trigger reselection on the next frame
            frame_count = i # Update frame count
            # continue # Skip full processing for this frame

        if is_in_reselection_phase:
            print(f"Target ID {id} lost or wait ended. Triggering re-selection on frame {i}...")
            # Ensure auto_id.py has a function like handle_target_reselection
            from auto_id import handle_target_reselection # Assuming it's defined there

            reselection_result = handle_target_reselection(
                current_frame_np=frame_np, # Pass the current frame
                player_detector_model=player_detector,
                tracker_obj=player_tracker, # IMPORTANT: Use the *same* tracker instance
                motion_estimator_obj=motion_estimator, # And same motion estimator
                converter_module=Converter,
                player_module=Player,
                team_module=Team,
                current_fps=fps,
                lost_target_id=id # Pass the ID that was lost
            )
            is_in_reselection_phase = False # Handled this reselection attempt

            if reselection_result == "WAIT_SIGNAL":
                print("Wait signal received. Waiting for approximately 1 second...")
                wait_frames_countdown = int(fps)
                target_missing_frames_count = 0 # Reset as user is actively managing
                frame_count = i
            elif reselection_result is None: # QUIT_SIGNAL
                print("Quit signal received from re-selection. Finishing video processing.")
                break # Exit the main processing loop
            else: # New target ID selected
                id = reselection_result
                print(f"New Target ID selected: {id}")
                target_missing_frames_count = 0 # Reset for the new target
                # Re-initialize BodyOrientationEstimator for the new target
                estimator = BodyOrientationEstimator(
                    target_id=id,
                    pose_model_path=args.pose_model,
                    forward_vector=FORWARD_VECTOR, # Make sure this is defined
                    smoothing_window=smooth_window,
                    pose_conf_threshold=pose_conf,
                    crop_padding_pixels=crop_pad
                )
        frame_count = i
        # 1. Player and Ball Detection
        # Pass frame_np to detectors
        players_detections_raw = get_player_detections(player_detector, frame_np) 
        ball_detections_raw = get_ball_detections(ball_detector, frame_np)
        detections = ball_detections_raw + players_detections_raw

        # 2. Motion Estimation & Tracking Update
        coord_transformations = update_motion_estimator(
            motion_estimator=motion_estimator,
            detections=detections,
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

        target_is_currently_tracked = any(p_data.data.get('id') == id for p_data in player_detections_tracked)
        if target_is_currently_tracked:
            target_missing_frames_count = 0
        else:
            target_missing_frames_count += 1
            print(f"Target ID {id} missing for {target_missing_frames_count} frames (hit_max: {player_tracker.hit_counter_max}).")

        if target_missing_frames_count > fps*1.5:
            print(f"Target ID {id} officially lost (exceeded hit_counter_max).")
            is_in_reselection_phase = True
        
        if automatic_teams:
            player_detections = classifier.predict_from_detections(
                detections=player_detections_tracked, # Pass the list of Norfair/Converter Detection objects
                img=frame_np
            )
        
        count = 0
        current_frame_track_ids = set()
        
        # 6. Remove Stale Player Objects, otherwise the bbox will be drawn even without player detection
        lost_track_ids = set(active_players.keys()) - current_frame_track_ids
        for track_id in lost_track_ids: 
            if track_id in active_players: # Check before deleting
                 del active_players[track_id]

        # 7. Get Ball Object
        ball = get_main_ball(ball_detections_tracked)
        

        if automatic_teams:
            players = Player.from_detections(detections=player_detections, teams=teams)
        else:
            players = Player.from_detections(detections=player_detections_tracked, teams=teams)

        match.update(players, ball, i, body_orientation, scanning, pressure, frame_np, target_id=id, estimator=estimator)
        body_position_list, pressure_list, turnable_list, number_of_scans_list = update_lists(players, id, body_position_list, pressure_list, turnable_list, number_of_scans_list)
        passes_list = match.passes
        if len(passes_list) > len(old_passes):
            for passs in passes_list:
                if passs not in old_passes:
                    print(f"Pass Detected! {passs.passer_id} -> {passs.receiver_id}. Initiated around frame {passs.initiation_frame}.")
                    if id == passs.receiver_id:
                        time_list.append(passs.initiation_frame * (1/fps))
                        target_bp.append(body_position_list[passs.initiation_frame])
                        target_pressure.append(pressure_list[passs.initiation_frame])
                        target_turnable.append(turnable_list[passs.initiation_frame])
                        target_number_of_scans.append(number_of_scans_list[passs.initiation_frame])
                        target_pass_list.append(passs)
        old_passes = passes_list
        
        
        if visualize:
            start_time = time.time()
            # 8. Draw Visualizations
            frame_final = frame_np # Default to original frame if drawing fails
            # Convert frame BGR NumPy to RGB PIL Image if Draw utils require it
            # If your Draw utils use OpenCV directly, skip this conversion
            frame_pil = PIL.Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))

            # Draw all players - Player.draw_players needs updated implementation
            frame_pil = Player.draw_players(
                players=players,
                frame=frame_pil,
                id=True,
                target_player_id=id, # Pass target ID
            )

            if pressure:
                frame_pil = Player.draw_pressure(
                    players=players,
                    frame=frame_pil,
                    target_id=id,
                )

            if body_orientation:
                frame_pil = Player.draw_body_orientation(
                    players=players,
                    frame=frame_pil,
                    target_id=id,
                )
            
            if scanning:
                frame_pil = Player.draw_scanning(
                    players=players,
                    frame=frame_pil,
                    target_id=id,
                )

            # Draw ball (optional)
            if ball:
                # Ensure ball.draw handles PIL Image and RGB color
                frame_pil = ball.draw(frame_pil) # Example green ball (RGB)

            # Draw arrow for orientation - adjust as needed
            frame_pil = draw_arrow(
                frame_pil=frame_pil,
                fwd_vec = np.array(FORWARD_VECTOR, dtype=float), # Example forward vector
            )
            
            
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
            end_time = time.time()
            print(f"Time taken for drawing: {end_time - start_time:.2f} seconds")

        # Print progress occasionally
        if (i + 1) % 100 == 0:
             print(f"Processed {i + 1} frames...")


    print(f"\nFinished processing {frame_count + 1} frames.")
    
finally: # Ensure resources are released
    df = generate_output_df(
        time=time_list,
        body_position=target_bp,
        pressure=target_pressure,
        turnable=target_turnable,
        number_of_scans=target_number_of_scans
    )
    df.to_csv(args.output.replace(".mp4", ".csv"))
    print("Releasing video resources...")
    # No video.release() needed for Norfair object
    # No writer.release() needed
    print(f"Output video should be saved to {args.output} by Norfair.")
    print("Cleanup finished.")