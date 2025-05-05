import argparse

import cv2
import numpy as np
import PIL
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator, TranslationTransformationGetter # Add TranslationTransformationGetter
from norfair.distances import mean_euclidean

from inference import Converter, HSVClassifier, InertiaClassifier, YoloV8
from inference.filters import filters
from run_utils import (
    get_ball_detections,
    get_main_ball,
    get_player_detections,
    update_motion_estimator,
)
from soccer import Match, Player, Team
from soccer.draw import AbsolutePath
from soccer.pass_event import Pass
from typing import List, Any # Assuming 'Any' for the Pass object type initially
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video",
    default="../FootieStats/data/alnassr.mp4",
    type=str,
    help="Path to the input video",
)
parser.add_argument(
    "--model", default="../FootieStats/weights/ball.pt", type=str, help="Path to the model"
)
parser.add_argument(
    "--passes",
    action="store_true",
    help="Enable pass detection",
)
parser.add_argument(
    "--possession",
    action="store_true",
    help="Enable possession counter",
)
args = parser.parse_args()

video = Video(input_path=args.video, output_path="possession.mp4")
fps = video.video_capture.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}")

# Object Detectors
player_detector = YoloV8(model_path="../FootieStats/weights/players.pt")
ball_detector = YoloV8(model_path=args.model)

# HSV Classifier
hsv_classifier = HSVClassifier(filters=filters)

# Add inertia to classifier
classifier = InertiaClassifier(classifier=hsv_classifier, inertia=50)

# Teams and Match
nas = Team(
    name="Al Nassr",
    abbreviation="NAS",
    color=(255,255,0), # <-- BGR Green (adjust shade as needed)
    board_color=(153,153,0), 
    text_color=(0, 0, 0), 
)
kaw = Team(name="Kawasaki Frontale", abbreviation="KAW", color=(220, 0, 0), # <-- BGR Red (adjust shade as needed)
    board_color=(255, 0, 0),  
    text_color=(255, 255, 255), 
)
teams = [nas, kaw]
match = Match(home=nas, away=kaw, fps=fps)
match.team_possession = nas

# Tracking
player_tracker = Tracker(
    distance_function="iou_opt",       # <-- USE IOU DISTANCE
    distance_threshold=0.7,          # <-- IoU Threshold (1 - min_iou = 0.3). TUNE THIS (0.5 to 0.9)
    initialization_delay=3,          # Keep low for players
    hit_counter_max=45,             # <-- Reduced persistence (Tune this: 30-60)
)

ball_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=150,
    initialization_delay=4,
    hit_counter_max=60,
)

motion_estimator = MotionEstimator()
coord_transformations = None

# Paths
path = AbsolutePath()

# Get Counter img
possession_background = match.get_possession_background()
passes_background = match.get_passes_background()
passes_list = []
length  = 0
for i, frame in enumerate(video):

    # Get Detections
    players_detections = get_player_detections(player_detector, frame)
    ball_detections = get_ball_detections(ball_detector, frame)
    detections = ball_detections + players_detections

    # Update trackers
    coord_transformations = update_motion_estimator(
        motion_estimator=motion_estimator,
        detections=detections,
        frame=frame,
    )

    player_track_objects = player_tracker.update(
        detections=players_detections, coord_transformations=coord_transformations
    )

    ball_track_objects = ball_tracker.update(
        detections=ball_detections, coord_transformations=coord_transformations
    )

    player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
    ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)

    player_detections = classifier.predict_from_detections(
        detections=player_detections,
        img=frame,
    )

    # Match update
    ball = get_main_ball(ball_detections)
    players = Player.from_detections(detections=players_detections, teams=teams)
    match.update(players, ball, i)

    # Draw
    frame = PIL.Image.fromarray(frame)

    if args.possession:
        print(f"Frame {i}: Number of players passed to draw_players: {len(players)}") 
        frame = Player.draw_players(
            players=players, frame=frame, confidence=False, id=True
        )

        frame = path.draw(
            img=frame,
            detection=ball.detection,
            coord_transformations=coord_transformations,
            color=match.team_possession.color,
        )

        frame = match.draw_possession_counter(
            frame, counter_background=possession_background, debug=False
        )

        if ball:
            frame = ball.draw(frame)

    if args.passes:
        pass_list = match.passes
        passes_list.append(match.passes)

        frame = Pass.draw_pass_list(
            img=frame, passes=pass_list, coord_transformations=coord_transformations
        )
        frame = match.draw_passes_counter(
            frame, counter_background=passes_background, debug=False
        )

    frame = np.array(frame)

    # Write video
    video.write(frame)

# with open("passes.pkl", 'wb') as f:
#     pickle.dump(passes_list, f)

# with open("passes.pkl", 'rb') as f:
#         passes_list = pickle.load(f)

# def find_pass_initiation_frames(target_player_id: int, pass_list: List[Pass]) -> List[int]:
#     """ Finds the frame numbers where passes were initiated *to* a specific player. """
#     initiation_frames = []
#     print(f"Searching {len(pass_list)} passes for receiver ID: {target_player_id}")
#     for i, pass_event in enumerate(pass_list[-1]):
#         # --- Access receiver ID and initiation frame (using new attributes) ---
#         receiver_id = None
#         print(f"Pass {i}: {pass_event}")
#         if hasattr(pass_event, 'receiver_id'):
#             receiver_id = pass_event.receiver_id
#         else:
#              if i == 0: print("WARNING: Pass object missing 'receiver_id' attribute.")
#              continue # Skip if missing

#         init_frame = None
#         if hasattr(pass_event, 'initiation_frame'):
#             init_frame = pass_event.initiation_frame
#         else:
#              if i == 0: print("WARNING: Pass object missing 'initiation_frame' attribute.")
#              continue # Skip if missing
       
#         # --- Check if it matches the target player ---
#         if receiver_id == target_player_id:
#             initiation_frames.append(init_frame)
            
#     print(f"Found {len(initiation_frames)} passes initiated to player ID {target_player_id}.")
#     return initiation_frames

# print(find_pass_initiation_frames(target_player_id=1, pass_list=passes_list))
