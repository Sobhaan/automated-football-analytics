#!/usr/bin/env python3
"""Soccer Video Analytics - Main Processing Script"""

import warnings
warnings.filterwarnings("ignore")

# Standard library
import argparse
import ast
import math
import os
import random
import time
from collections import deque, Counter
from typing import Dict, Optional, List, Any

# Third-party libraries
import cv2
import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
from ultralytics import YOLO
from norfair import Tracker, Video, Detection
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean, iou_opt
from norfair.filter import FilterFactory, OptimizedKalmanFilterFactory

# Project imports
from inference import Converter, YoloV8
from inference.hsv_classifier import HSVClassifier
from inference.inertia_classifier import InertiaClassifier
from inference.filters import filters
from run_utils import (
    get_ball_detections,
    get_main_ball,
    get_player_detections,
    update_motion_estimator,
)
from soccer import Player, Ball, Match, Team
from soccer.pass_event import Pass
from soccer.draw import AbsolutePath, Draw, draw_arrow
from body_orientation import BodyOrientationEstimator
from auto_hsv_generator import generate_auto_hsv_filters
from output import generate_output_df, update_lists
from forward_vector import forward_vector
from auto_id import select_target_player_id_on_first_frame


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Soccer Video Analytics with Target Orientation")
    parser.add_argument("--video", default="../FootieStats/data/alnassr.mp4", type=str, help="Path to the input video")
    parser.add_argument("--ball_model", default="../FootieStats/weights/ball.engine", type=str, help="Path to the ball detection model")
    parser.add_argument("--player_model", default="../FootieStats/weights/players.engine", type=str, help="Path to the player detection model")
    parser.add_argument("--visualize", default=True, type=str, help="Enable visualization of players and ball")
    parser.add_argument("--output", default="output/11.mp4", type=str, help="Path for output video file")
    return parser.parse_args()


def setup_video(args):
    """Setup video input and output"""
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    video = Video(input_path=args.video, output_path=args.output)
    fps = video.video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Input video: {frame_width}x{frame_height} @ {fps:.2f} FPS")
    print(f"Output video will be saved to: {args.output}")
    
    return video, fps, frame_width, frame_height


def load_models(args):
    """Load detection models"""
    print("Loading object detectors...")
    player_conf = 0.4
    player_detector = YoloV8(model_path=args.player_model, conf=player_conf)
    ball_detector = YoloV8(model_path=args.ball_model)
    print("Object detectors loaded.")
    return player_detector, ball_detector


def setup_hsv_filters(video_path, player_detector):
    """Setup HSV filters for team classification"""
    temp_video_reader = cv2.VideoCapture(video_path)
    total_frames = int(temp_video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    random_frame_indices = sorted(random.sample(range(total_frames), 60))
    initial_frames_for_filters = []
    
    for frame_index in random_frame_indices:
        temp_video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = temp_video_reader.read()
        if ret:
            initial_frames_for_filters.append(frame)
    
    auto_generated_filters = generate_auto_hsv_filters(
        player_detector_model=player_detector,
        frames=initial_frames_for_filters,
        team_names=["Team 1", "Team 2"],
        referee_name=None,
    )
    
    return auto_generated_filters


def setup_teams_and_match(fps):
    """Setup teams and match objects"""
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
        color=(255, 255, 0),
        board_color=(153, 153, 0),
        text_color=(0, 0, 0),
    )
    teams = [team1, team2]
    match = Match(home=team1, away=team2, fps=fps)
    return teams, match


def setup_trackers():
    """Setup player and ball trackers"""
    print("Initializing trackers...")
    
    iou_thresh = 0.8
    hit_max = 100
    player_tracker = Tracker(
        distance_function="iou",
        distance_threshold=iou_thresh,
        initialization_delay=2,
        hit_counter_max=hit_max,
    )
    
    ball_tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=60,
        initialization_delay=0,
        hit_counter_max=200,
        pointwise_hit_counter_max=10,
        filter_factory=OptimizedKalmanFilterFactory(Q=0.05, R=0.5)
    )
    
    motion_estimator = MotionEstimator()
    return player_tracker, ball_tracker, motion_estimator


def draw_frame_visualizations(frame_np, players, ball, id, FORWARD_VECTOR, frame_count):
    """Draw all visualizations on frame"""
    frame_pil = PIL.Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
    
    frame_pil = Player.draw_players(players=players, frame=frame_pil, id=True, target_player_id=id)
    frame_pil = Player.draw_pressure(players=players, frame=frame_pil, target_id=id)
    frame_pil = Player.draw_body_orientation(players=players, frame=frame_pil, target_id=id)
    frame_pil = Player.draw_scanning(players=players, frame=frame_pil, target_id=id)
    
    if ball:
        frame_pil = ball.draw(frame_pil)
    
    frame_pil = draw_arrow(frame_pil=frame_pil, fwd_vec=np.array(FORWARD_VECTOR, dtype=float))
    
    frame_pil = cv2.putText(
        np.array(frame_pil), f"Frame: {frame_count}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
    )
    
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)


def main():
    # Parse arguments
    args = parse_arguments()
    start_time = time.time()
    visualize = ast.literal_eval(str(args.visualize))
    
    # Setup
    video, fps, frame_width, frame_height = setup_video(args)
    player_detector, ball_detector = load_models(args)
    
    # HSV filters and classifier
    filter = setup_hsv_filters(args.video, player_detector)
    hsv_classifier = HSVClassifier(filters=filter)
    classifier = InertiaClassifier(classifier=hsv_classifier, inertia=30)
    
    # Teams and match
    teams, match = setup_teams_and_match(fps)
    
    # Trackers
    player_tracker, ball_tracker, motion_estimator = setup_trackers()
    coord_transformations = None
    
    # Target player selection
    id = select_target_player_id_on_first_frame(
        current_video_obj=video,
        player_detector_model=player_detector,
        tracker_obj=player_tracker,
        motion_estimator_obj=motion_estimator,
        converter_module=Converter,
        player_module=Player,
        team_module=Team,
        tracker_init_delay_for_selection_frames=player_tracker.initialization_delay
    )
    
    # Orientation estimator
    smooth_window = 10
    pose_conf = 0.1
    crop_pad = 0
    FORWARD_VECTOR = forward_vector()
    
    estimator = BodyOrientationEstimator(
        target_id=id,
        forward_vector=FORWARD_VECTOR,
        smoothing_window=smooth_window,
        pose_conf_threshold=pose_conf,
        crop_padding_pixels=crop_pad
    )
    
    # State variables
    target_missing_frames_count = 0
    is_in_reselection_phase = False
    wait_frames_countdown = 0
    active_players: Dict[int, Player] = {}
    old_passes = []
    passes_list = []
    target_pass_list = []
    
    # Value lists
    body_position_list = []
    pressure_list = []
    turnable_list = []
    scan_angles_list = []
    
    # Final target values
    time_list = []
    target_bp = []
    target_pressure = []
    target_turnable = []
    target_number_of_scans = []
    
    # Main processing loop
    print("\nStarting video processing...")
    frame_count = 0
    
    try:
        for i, frame in enumerate(video):
            frame_np = frame
            print("\n")
            print(f"Processing frame {i}...")
            print(is_in_reselection_phase)
            
            # Handle wait countdown
            if wait_frames_countdown > 0:
                print(f"Waiting... {wait_frames_countdown} frames remaining.")
                wait_frames_countdown -= 1
                if wait_frames_countdown == 0:
                    is_in_reselection_phase = True
                frame_count = i
            
            # Handle reselection
            if is_in_reselection_phase:
                print(f"Target ID {id} lost or wait ended. Triggering re-selection on frame {i}...")
                from auto_id import handle_target_reselection
                
                reselection_result = handle_target_reselection(
                    current_frame_np=frame_np,
                    player_detector_model=player_detector,
                    tracker_obj=player_tracker,
                    motion_estimator_obj=motion_estimator,
                    converter_module=Converter,
                    player_module=Player,
                    team_module=Team,
                    current_fps=fps,
                    lost_target_id=id
                )
                is_in_reselection_phase = False
                
                if reselection_result == "WAIT_SIGNAL":
                    print("Wait signal received. Waiting for approximately 1 second...")
                    wait_frames_countdown = int(fps)
                    target_missing_frames_count = 0
                    frame_count = i
                elif reselection_result is None:
                    print("Quit signal received from re-selection. Finishing video processing.")
                    break
                else:
                    id = reselection_result
                    print(f"New Target ID selected: {id}")
                    FORWARD_VECTOR = forward_vector()
                    target_missing_frames_count = 0
                    estimator = BodyOrientationEstimator(
                        target_id=id,
                        forward_vector=FORWARD_VECTOR,
                        smoothing_window=smooth_window,
                        pose_conf_threshold=pose_conf,
                        crop_padding_pixels=crop_pad
                    )
            
            frame_count = i
            
            # Detection
            players_detections_raw = get_player_detections(player_detector, frame_np)
            ball_detections_raw = get_ball_detections(ball_detector, frame_np)
            detections = ball_detections_raw + players_detections_raw
            
            # Motion estimation and tracking
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
            
            # Convert tracked objects
            player_detections_tracked = Converter.TrackedObjects_to_Detections(player_track_objects)
            ball_detections_tracked = Converter.TrackedObjects_to_Detections(ball_track_objects)
            
            # Check target tracking
            target_is_currently_tracked = any(p_data.data.get('id') == id for p_data in player_detections_tracked)
            if target_is_currently_tracked:
                target_missing_frames_count = 0
            else:
                target_missing_frames_count += 1
                print(f"Target ID {id} missing for {target_missing_frames_count} frames (max: {fps}).")
            
            if target_missing_frames_count > fps:
                print(f"Target ID {id} officially lost (exceeded hit_counter_max).")
                is_in_reselection_phase = True
            
            # Classification
            player_detections = classifier.predict_from_detections(
                detections=player_detections_tracked,
                img=frame_np
            )
            
            # Clean up stale players
            count = 0
            current_frame_track_ids = set()
            lost_track_ids = set(active_players.keys()) - current_frame_track_ids
            for track_id in lost_track_ids:
                if track_id in active_players:
                    del active_players[track_id]
            
            # Get ball and players
            ball = get_main_ball(ball_detections_tracked)
            players = Player.from_detections(detections=player_detections, teams=teams)
            
            # Update match state
            match.update(players, ball, i, scan_angles_list, frame_np, target_id=id, estimator=estimator)
            
            # Update lists
            body_position_list, pressure_list, turnable_list, scan_angles_list = update_lists(
                players, id, body_position_list, pressure_list, turnable_list, scan_angles_list
            )
            
            # Check for passes
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
                            number_of_scans = Match.angles_to_count(scan_angles_list, passs.initiation_frame, fps)
                            target_number_of_scans.append(number_of_scans)
                            target_pass_list.append(passs)
                            print(f"Target Player {id} received a pass at frame {passs.initiation_frame}. Body Position: {body_position_list[passs.initiation_frame]}, Pressure: {pressure_list[passs.initiation_frame]}, Turnable: {turnable_list[passs.initiation_frame]}, Number of Scans: {number_of_scans}")
            old_passes = passes_list
            
            # Visualization
            if visualize:
                frame_final = draw_frame_visualizations(frame_np, players, ball, id, FORWARD_VECTOR, frame_count)
                video.write(frame_final)
        
        print(f"\nFinished processing {frame_count + 1} frames.")
        
    finally:
        # Save output
        df = generate_output_df(
            time=time_list,
            body_position=target_bp,
            pressure=target_pressure,
            turnable=target_turnable,
            number_of_scans=target_number_of_scans
        )
        df.to_csv(args.output.replace(".mp4", ".csv"))
        print(f"Output video should be saved to {args.output} by Norfair.")
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()