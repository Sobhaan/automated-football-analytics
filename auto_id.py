# auto_id.py
import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import PIL
from PIL import Image
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import sys

from inference import Converter # Assuming these are findable from auto_id.py's location
from run_utils import get_player_detections, update_motion_estimator

# --- Global variables for click selection ---
selected_player_id_from_click = None
first_frame_detections_for_click: List[Any] = []

def get_player_id_by_click(event, x, y, flags, param):
    global selected_player_id_from_click, first_frame_detections_for_click
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_id = None
        if not first_frame_detections_for_click:
            print("DEBUG: No detections available for click selection in callback.")
            return

        for det in first_frame_detections_for_click:
            if hasattr(det, 'points') and 'id' in det.data:
                x1, y1 = det.points[0]
                x2, y2 = det.points[1]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    clicked_id = det.data['id']
                    break
        if clicked_id is not None:
            if selected_player_id_from_click != clicked_id: # Only print if it's a new click
                print(f"INFO: Player ID {clicked_id} registered by click. Press ENTER in the selection window to confirm/proceed to terminal.")
            selected_player_id_from_click = clicked_id
        else:
            # Click was not on a player, clear previous click selection
            if selected_player_id_from_click is not None:
                 print("INFO: Click was not on a player. Previous click selection cleared.")
            selected_player_id_from_click = None


def select_target_player_id_on_first_frame( # Renaming for clarity might be good later
    current_video_obj,
    player_detector_model,
    tracker_obj, # This is the main tracker from run.py
    motion_estimator_obj,
    converter_module,
    player_module,
    team_module,
    tracker_init_delay_for_selection_frames: int # New parameter
):
    global selected_player_id_from_click, first_frame_detections_for_click
    
    num_frames_to_process_for_ids = tracker_init_delay_for_selection_frames + 1
    
    print(f"--- Initial ID Selection ---")
    print(f"Processing first {num_frames_to_process_for_ids} frames to allow tracker (init_delay={tracker_init_delay_for_selection_frames}) to assign IDs...")

    last_processed_frame_for_display = None
    current_tracked_objects = [] # To store tracks from the Nth frame

    for frame_num in range(num_frames_to_process_for_ids):
        try:
            current_frame_np = next(iter(current_video_obj))
            if frame_num == num_frames_to_process_for_ids - 1: # If this is the last frame we process
                last_processed_frame_for_display = current_frame_np.copy() # Keep a copy for display
        except StopIteration:
            print(f"Error: Video ended before processing all {num_frames_to_process_for_ids} required initial frames.")
            current_video_obj.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video
            return None # Cannot proceed

        if current_frame_np is None: # Should be caught by StopIteration
            print(f"Error: Got None frame at initial frame {frame_num}.")
            current_video_obj.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return None

        # Process this frame through detection and tracking
        players_detections_raw = get_player_detections(player_detector_model, current_frame_np)
        coord_transformations = update_motion_estimator(
            motion_estimator=motion_estimator_obj,
            detections=players_detections_raw, # Or all detections if your setup needs it
            frame=current_frame_np,
        )
        # Update the main tracker instance. Its state builds up over these N frames.
        current_tracked_objects = tracker_obj.update(
            detections=players_detections_raw, coord_transformations=coord_transformations
        )
        print(f"Processed initial frame {frame_num + 1}/{num_frames_to_process_for_ids}. Tracked objects: {len(current_tracked_objects)}")

    # After processing N frames, current_tracked_objects holds tracks from the Nth frame.
    # These should now have IDs if they met the initialization_delay criteria.
    first_frame_detections_for_click.clear()
    first_frame_detections_for_click.extend(converter_module.TrackedObjects_to_Detections(current_tracked_objects))

    # IMPORTANT: Reset the video object so the main loop in run.py starts from frame 0
    current_video_obj.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if last_processed_frame_for_display is None: # Should not happen if loop completed
        print("Error: Failed to get a frame for display after initial processing.")
        return None

    # --- Drawing and UI logic (uses last_processed_frame_for_display) ---
    frame_pil_display = PIL.Image.fromarray(cv2.cvtColor(last_processed_frame_for_display, cv2.COLOR_BGR2RGB))
    temp_players_for_drawing = []
    detected_ids_for_prompt = []

    default_team = team_module(name="Detected", abbreviation="DET", color=(128, 128, 128))

    for det in first_frame_detections_for_click: # These are tracks from the Nth frame
        if 'id' in det.data:
            p_obj = player_module(id=det.data['id'], detection=det, team=default_team)
            temp_players_for_drawing.append(p_obj)
            detected_ids_for_prompt.append(det.data['id'])

    if not detected_ids_for_prompt:
        print(f"No players were successfully tracked with IDs after {num_frames_to_process_for_ids} frames. Cannot select Target ID.")
        # Optionally show the last frame for debugging
        cv2.imshow(f"No IDs (InitDelay: {tracker_init_delay_for_selection_frames}) - ESC to Quit", last_processed_frame_for_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None

    # --- The rest of your UI for selection (imshow, setMouseCallback, input loop) ---
    # This part remains largely the same as your last working version for auto_id.py,
    # using `last_processed_frame_for_display` and `detected_ids_for_prompt`.
    # Example snippet:
    frame_with_ids_pil = player_module.draw_players(
        players=temp_players_for_drawing,
        frame=frame_pil_display,
        id=True
    )
    frame_with_ids_bgr = cv2.cvtColor(np.array(frame_with_ids_pil), cv2.COLOR_RGB2BGR)
    window_title = f"Select Target (InitDelay:{tracker_init_delay_for_selection_frames}) - Click, then ENTER in Window"
    cv2.imshow(window_title, frame_with_ids_bgr)
    cv2.setMouseCallback(window_title, get_player_id_by_click) # Your existing callback

    print(f"\nAvailable Player IDs (after {num_frames_to_process_for_ids} frames):", sorted(list(set(detected_ids_for_prompt))))
    print("INFO: 1. Click on a player in the window.")
    print("INFO: 2. Press ENTER in the SELECTION WINDOW to proceed to terminal confirmation.")
    print("INFO: Or, press ESC in the SELECTION WINDOW to cancel.")
    
    selected_id_final = None # Initialize
    selected_player_id_from_click = None # Reset for this fresh selection

    # Your existing UI loop (while True: key = cv2.waitKey(30)...) goes here
    # Ensure it uses `detected_ids_for_prompt` for validation.
    # ... (your full UI loop from the last working auto_id.py) ...
    while True: 
        key = cv2.waitKey(30) & 0xFF
        if key == 13: # ENTER in OpenCV window
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(window_title)
                cv2.waitKey(1)
            prompt_msg = "Enter Player ID: "
            if selected_player_id_from_click is not None and selected_player_id_from_click in detected_ids_for_prompt:
                prompt_msg = f"Player {selected_player_id_from_click} clicked. Enter to confirm, or type new ID: "
            try:
                user_typed_input = input(prompt_msg).strip()
                if not user_typed_input: 
                    if selected_player_id_from_click is not None and selected_player_id_from_click in detected_ids_for_prompt:
                        selected_id_final = selected_player_id_from_click
                        print(f"Target ID {selected_id_final} confirmed from click.")
                    else:
                        print("No ID entered. Please re-run to select.")
                        selected_id_final = None 
                else: 
                    potential_id = int(user_typed_input)
                    if potential_id in detected_ids_for_prompt:
                        selected_id_final = potential_id
                        print(f"Target ID {selected_id_final} selected via terminal.")
                    else:
                        print(f"Typed ID {potential_id} is not in available list. Re-run to select.")
                        selected_id_final = None
                break 
            except Exception as e:
                print(f"Terminal input error: {e}. Re-run to select.")
                selected_id_final = None
                break
        elif key == 27: 
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                if selected_player_id_from_click and selected_player_id_from_click in detected_ids_for_prompt:
                    selected_id_final = selected_player_id_from_click
                else: selected_id_final = None
                break
            
    if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) >= 1: # Ensure cleanup
        cv2.destroyWindow(window_title)
        cv2.waitKey(1)
    
    first_frame_detections_for_click.clear()
    # selected_player_id_from_click = None # Reset if needed elsewhere, but function scope usually handles
    
    return selected_id_final


selected_player_id_from_click: Optional[int] = None
first_frame_detections_for_click: List[Any] = []

def get_player_id_by_click(event, x: int, y: int, flags, param):
    global selected_player_id_from_click, first_frame_detections_for_click
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_id_temp = None
        if not first_frame_detections_for_click:
            return
        for det in first_frame_detections_for_click:
            if hasattr(det, 'points') and 'id' in det.data:
                x1, y1 = det.points[0]
                x2, y2 = det.points[1]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    clicked_id_temp = det.data['id']
                    break
        if clicked_id_temp is not None:
            if selected_player_id_from_click != clicked_id_temp:
                 print(f"INFO: Player ID {clicked_id_temp} registered by click. In selection window: Press ENTER to confirm/go to terminal, W for wait, or ESC to quit.")
            selected_player_id_from_click = clicked_id_temp
        else:
            if selected_player_id_from_click is not None:
                print("INFO: Click was not on a player. Previous click selection cleared.")
            selected_player_id_from_click = None

def handle_target_reselection(
    current_frame_np: np.ndarray,
    player_detector_model: Any,
    tracker_obj: Any,
    motion_estimator_obj: Any,
    converter_module: Any,
    player_module: Any,
    team_module: Any,
    current_fps: float,
    lost_target_id: Optional[int]
) -> Optional[Any]: # Returns new ID (int), "WAIT_SIGNAL" (str), or None (for quit)
    global selected_player_id_from_click, first_frame_detections_for_click
    print(f"\n--- Target Reselection ---\nPrevious target ID: {lost_target_id} was lost.")

    reselection_detections_raw = get_player_detections(player_detector_model, current_frame_np)
    reselection_coord_transformations = update_motion_estimator(
        motion_estimator=motion_estimator_obj,
        detections=reselection_detections_raw,
        frame=current_frame_np,
    )
    reselection_player_track_objects = tracker_obj.update(
        detections=reselection_detections_raw, coord_transformations=reselection_coord_transformations
    )

    first_frame_detections_for_click.clear()
    first_frame_detections_for_click.extend(converter_module.TrackedObjects_to_Detections(reselection_player_track_objects))

    frame_pil = PIL.Image.fromarray(cv2.cvtColor(current_frame_np, cv2.COLOR_BGR2RGB))
    temp_players_for_drawing = []
    detected_ids_in_current_frame = []
    default_team = team_module(name="Detected", abbreviation="DET", color=(128, 128, 128))

    for det in first_frame_detections_for_click:
        if 'id' in det.data:
            p_obj = player_module(id=det.data['id'], detection=det, team=default_team)
            temp_players_for_drawing.append(p_obj)
            detected_ids_in_current_frame.append(det.data['id'])

    frame_with_ids_pil = player_module.draw_players(players=temp_players_for_drawing, frame=frame_pil, id=True)
    frame_with_ids_bgr = cv2.cvtColor(np.array(frame_with_ids_pil), cv2.COLOR_RGB2BGR)

    window_title = f"Target Lost (was {lost_target_id}). Reselect. (W: Wait, ESC: Quit)"
    # Add a text prompt on the image itself
    prompt_on_image = "Click Player. Then in THIS window: ENTER to confirm, W for wait, ESC to quit."
    text_y_pos = frame_with_ids_bgr.shape[0] - 20 
    if text_y_pos < 20: text_y_pos = 20 
    cv2.putText(frame_with_ids_bgr, prompt_on_image, (10, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA) 
    cv2.putText(frame_with_ids_bgr, prompt_on_image, (10, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow(window_title, frame_with_ids_bgr)
    cv2.setMouseCallback(window_title, get_player_id_by_click)

    if detected_ids_in_current_frame:
        print("Available IDs in current frame:", sorted(list(set(detected_ids_in_current_frame))))
    else:
        print("No players currently detected in this frame.")
    print("INSTRUCTIONS: Interact with the OpenCV window that popped up.")
    print("  - Click a player if you see one you want.")
    print("  - Then, in that OpenCV window, press:")
    print("    - ENTER: To proceed to terminal input (confirms click or allows typing ID).")
    print("    - 'W' key: To wait for ~1 second and try re-selection again.")
    print("    - 'ESC' key: To quit the program entirely.")

    # This variable will store the action determined by GUI interaction
    gui_action_signal = "CONTINUE_GUI_LOOP" 

    # Reset click for this specific interaction
    # selected_player_id_from_click = None # Already reset in the main selection if this is a new function

    while gui_action_signal == "CONTINUE_GUI_LOOP":
        key = cv2.waitKey(30) & 0xFF

        if key == 27: # ESC key in OpenCV window
            print("ESC pressed in reselection window.")
            gui_action_signal = "QUIT"
            break 
        elif key == ord('w') or key == ord('W'): # 'W' key for WAIT
            print("'Wait' key (W) pressed in reselection window.")
            gui_action_signal = "WAIT"
            break
        elif key == 13: # ENTER pressed in OpenCV window
            print("ENTER pressed in reselection window. Proceeding to terminal.")
            gui_action_signal = "PROCEED_TO_TERMINAL"
            break 
        
        # Check if window was closed by pressing 'X'
        if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
            print("Reselection window was closed by user (X button).")
            gui_action_signal = "QUIT_FROM_X" # Differentiate if needed, or just map to QUIT
            break

    # --- GUI interaction loop has ended. Now, clean up window and handle terminal input if signaled ---
    if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow(window_title)
        cv2.waitKey(20) # Increased waitKey slightly to ensure window closes fully

    # Process the action determined by the GUI loop
    if gui_action_signal == "QUIT" or gui_action_signal == "QUIT_FROM_X":
        print("Quitting reselection and video processing.")
        final_reselection_choice = None
    elif gui_action_signal == "WAIT":
        final_reselection_choice = "WAIT_SIGNAL"
    elif gui_action_signal == "PROCEED_TO_TERMINAL":
        prompt_msg = "Enter new Player ID (or 'wait'/'quit'): "
        if selected_player_id_from_click is not None and selected_player_id_from_click in detected_ids_in_current_frame:
            prompt_msg = f"Player {selected_player_id_from_click} clicked. Press Enter to confirm, or type new ID/wait/quit: "
        
        try:
            user_input = input(prompt_msg).strip().lower()
            if not user_input: # User pressed Enter in terminal
                if selected_player_id_from_click is not None and selected_player_id_from_click in detected_ids_in_current_frame:
                    final_reselection_choice = selected_player_id_from_click
                    print(f"New Target ID {final_reselection_choice} confirmed from click.")
                else:
                    print("No ID entered, and no valid click to confirm. Defaulting to 'wait'.")
                    final_reselection_choice = "WAIT_SIGNAL"
            elif user_input == "wait":
                final_reselection_choice = "WAIT_SIGNAL"
            elif user_input == "quit":
                final_reselection_choice = None # None signals quit to run.py
            else:
                try:
                    potential_id = int(user_input)
                    if potential_id in detected_ids_in_current_frame:
                        final_reselection_choice = potential_id
                        print(f"New Target ID {final_reselection_choice} selected via terminal.")
                    else:
                        print(f"Typed ID {potential_id} is not in available list {detected_ids_in_current_frame}. Defaulting to 'wait'.")
                        final_reselection_choice = "WAIT_SIGNAL"
                except ValueError:
                    print(f"Invalid ID format: '{user_input}'. Defaulting to 'wait'.")
                    final_reselection_choice = "WAIT_SIGNAL"
        except Exception as e: # Catch EOFError etc.
            print(f"Terminal input error: {e}. Defaulting to 'quit'.")
            final_reselection_choice = None
    else: # Should not happen if gui_action_signal is always set
        print("Unknown state after GUI interaction. Defaulting to quit.")
        final_reselection_choice = None

    first_frame_detections_for_click.clear()
    # selected_player_id_from_click = None # Reset this global after each selection cycle

    return final_reselection_choice