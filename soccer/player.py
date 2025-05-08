# soccer/player.py
import numpy as np
import PIL
from PIL import ImageDraw, ImageFont # Example imports for PIL drawing
import cv2 # Keep if needed for other functions
from typing import List, Any, Optional
import pandas as pd

# --- Default Colors ---
DEFAULT_PLAYER_COLOR = (0, 255, 255) # Yellow BGR (Note: PIL uses RGB)
DEFAULT_TEXT_COLOR_PIL = (0, 0, 0)     # Black for text
DEFAULT_PLAYER_COLOR_PIL = (255, 255, 0) # Yellow RGB

# --- Dummy Draw Class (Replace with your actual Draw utility) ---
# This is just a placeholder so the code structure runs.
# Your actual Draw class should handle PIL Image objects.
class Draw:
    @staticmethod
    def rectangle(img: PIL.Image.Image, pt1: tuple, pt2: tuple, color: tuple, thickness: int):
        try:
            draw = ImageDraw.Draw(img)
            # PIL uses RGB, ensure 'color' is RGB tuple e.g. (255, 255, 0) for yellow
            draw.rectangle([pt1, pt2], outline=color, width=thickness)
        except Exception as e:
             print(f"Draw.rectangle error: {e}")
        return img

    @staticmethod
    def text(img: PIL.Image.Image, text: str, pos: tuple, color: tuple, text_color: tuple, font_scale: float):
         try:
             draw = ImageDraw.Draw(img)
             # You might need to load a specific font and scale
             # font = ImageFont.truetype("arial.ttf", size=int(font_scale * 20)) # Example
             font = ImageFont.load_default() # Basic default font
             # Simple text draw - does not handle background color easily
             draw.text(pos, text, fill=text_color, font=font) 
         except Exception as e:
             print(f"Draw.text error: {e}")
         return img
# --- End Dummy Draw Class ---


class Player:
    # Assuming Team class might still exist, but handle if team is None
    def __init__(self, id: int, detection: Any, team: Optional['Team'] = None): 
        self.id = id
        self.team = team # Store team object or None
        self.detection = detection # Store latest detection data (dict or object)
        
        # --- ADD State Attributes ---
        self.keypoints = None # Store absolute keypoints [K, 3] numpy array or None
        self.orientation = "Unknown" # Store raw orientation string
        self.smoothed_orientation = "Unknown" # Store smoothed orientation string
        
        # Store center coordinates if detection provides them easily
        # This helps avoid recalculating frequently
        self._update_center()

    def _update_center(self):
        """Helper to calculate center from detection data."""
        self.center = None
        try:
            if isinstance(self.detection, dict):
                if 'xmin' in self.detection:
                    self.center = (
                        (self.detection['xmin'] + self.detection['xmax']) / 2,
                        (self.detection['ymin'] + self.detection['ymax']) / 2
                    )
            # Add other ways to get center if detection format varies
            # elif hasattr(self.detection, 'points'): ...
        except Exception:
            self.center = None # Ensure it's None if calculation fails

    # --- Keep necessary methods like distance calculation ---
    def distance(self, obj: Any) -> float:
        """Calculate distance to another object (Player or Ball)."""
        # Needs implementation based on how center points are stored/calculated
        if self.center is None or obj.center is None:
            return float('inf')
        return np.linalg.norm(np.array(self.center) - np.array(obj.center))

    def distance_to_ball(self, ball: 'Ball') -> float:
         # Example implementation
         return self.distance(ball)
         
    def closest_foot_to_ball_abs(self, ball: 'Ball') -> Optional[np.ndarray]:
         """Estimate absolute coords of foot closest to ball using keypoints."""
         if self.keypoints is None or ball is None or ball.center is None:
             return None
         try:
             # Keypoint indices for ankles in YOLOv8-Pose
             l_ankle_idx, r_ankle_idx = 15, 16 
             if self.keypoints.shape[0] <= max(l_ankle_idx, r_ankle_idx): return None

             l_ankle = self.keypoints[l_ankle_idx, :2] # Absolute xy
             r_ankle = self.keypoints[r_ankle_idx, :2] # Absolute xy
             
             # Check if keypoints are valid (e.g., > 0)
             l_valid = np.all(l_ankle > 0)
             r_valid = np.all(r_ankle > 0)

             ball_center_np = np.array(ball.center)

             if l_valid and r_valid:
                 # Choose ankle closer to ball center
                 dist_l = np.linalg.norm(l_ankle - ball_center_np)
                 dist_r = np.linalg.norm(r_ankle - ball_center_np)
                 return l_ankle if dist_l <= dist_r else r_ankle
             elif l_valid:
                 return l_ankle
             elif r_valid:
                 return r_ankle
             else: # Neither ankle is valid
                 return None
         except Exception:
             return None # Fallback

    # --- Static method to draw players ---
    @staticmethod
    def draw_players(
        players: List["Player"],
        frame: PIL.Image.Image,
        id: bool = False,
        draw_orientation: bool = False, 
        target_player_id: Optional[int] = None # <-- Add target ID argument
    ) -> PIL.Image.Image:
        
        for player in players:
            if player.detection is None: continue
            
            try:
                # --- Extract BBox Coordinates (Adapt based on actual detection structure) ---
                x1, y1, x2, y2 = -1, -1, -1, -1
                detection_data = player.detection # This is likely a dict now
                if isinstance(detection_data, dict) and 'xmin' in detection_data:
                     # Ensure coords are valid numbers before casting
                     if all(pd.notna(detection_data[k]) for k in ['xmin','ymin','xmax','ymax']):
                          x1 = int(detection_data['xmin']); y1 = int(detection_data['ymin'])
                          x2 = int(detection_data['xmax']); y2 = int(detection_data['ymax'])
                # Add other extraction methods if format varies (e.g., from Norfair object)
                # elif hasattr(player.detection, 'points'): ... 
                
                # Proceed only if valid coordinates were extracted
                if x1 > -1 and x1 < x2 and y1 < y2:
                    # --- Determine Color ---
                    # Use team color if team exists, otherwise default PIL color
                    draw_color = player.team.color_rgb if player.team and hasattr(player.team, 'color_rgb') else DEFAULT_PLAYER_COLOR_PIL # Assumes Team has 'color_rgb' tuple attribute
                    
                    # --- Draw Rectangle ---
                    frame = Draw.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)

                    # --- Prepare Label ---
                    label_items = []
                    if id: label_items.append(f"ID:{player.id}")
                    
                    # --- Draw orientation ONLY for the target player ---
                    if draw_orientation and player.id == target_player_id and player.smoothed_orientation != "Unknown":
                        label_items.append(player.smoothed_orientation) 
                        
                    label = " ".join(label_items)

                    # --- Draw Label ---
                    if label:
                         # Simple text draw using PIL (adjust position/font as needed)
                         draw = ImageDraw.Draw(frame)
                         text_pos = (x1, y1 - 15 if y1 > 15 else y1 + 5) # Position above box
                         try:
                              # Basic font, find better way if needed
                              font = ImageFont.load_default() 
                              # Draw text (no background)
                              draw.text(text_pos, label, fill=DEFAULT_TEXT_COLOR_PIL, font=font) 
                         except Exception as font_e:
                              print(f"Error loading/drawing font: {font_e}")
                              # Fallback draw without font object
                              draw.text(text_pos, label, fill=DEFAULT_TEXT_COLOR_PIL) 

            except Exception as e:
                print(f"Error drawing player ID {player.id}: {e}")

        return frame

# You might need to define the Ball class structure if methods like closest_foot_to_ball_abs use it
# class Ball: # Dummy Ball class for type hints if not imported
#      center: Optional[tuple] = None
#      detection: Any = None
#      # Add other necessary attributes/methods

# # You might need to define the Team class structure if Player methods use it
# class Team: # Dummy Team class for type hints
#     name: str = "Unknown"
#     color_rgb: tuple = (255, 255, 255) # Example attribute needed by drawing
#     # Add other necessary attributes/methods