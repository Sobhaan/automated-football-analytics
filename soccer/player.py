# soccer/player.py
import numpy as np
import PIL
from PIL import ImageDraw, ImageFont # Example imports for PIL drawing
import cv2 # Keep if needed for other functions
from typing import List, Any, Optional
import pandas as pd
import math

from norfair import Detection
# from inference.filters import filters, get_first_color_by_name

from soccer.ball import Ball
from soccer.draw import Draw
from soccer.team import Team

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
        draw = ImageDraw.Draw(img)
        # PIL uses RGB, ensure 'color' is RGB tuple e.g. (255, 255, 0) for yellow
        draw.rectangle([pt1, pt2], outline=color, width=thickness)
        return img

    @staticmethod
    def text(img: PIL.Image.Image, text: str, pos: tuple, color: tuple, text_color: tuple, font_scale: float):
        draw = ImageDraw.Draw(img)
        # You might need to load a specific font and scale
        # font = ImageFont.truetype("arial.ttf", size=int(font_scale * 20)) # Example
        font = ImageFont.load_default() # Basic default font
        # Simple text draw - does not handle background color easily
        draw.text(pos, text, fill=text_color, font=font) 
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
        self.body_orientation = "Unknown" # Store body orientation string
        self.pressure = "None"
        self.scanning = None

        self.head_orientation = None
        # if detection and isinstance(detection, Detection):
        #     if "team" in detection.data:
        #         self.team = detection.data["classification"] 
        # Store center coordinates if detection provides them easily
        # This helps avoid recalculating frequently
        self._update_center()

    def _update_center(self):
        """Helper to calculate center from detection data."""
        self.center = None
        if isinstance(self.detection, dict):
            if 'xmin' in self.detection:
                self.center = (
                    (self.detection['xmin'] + self.detection['xmax']) / 2,
                    (self.detection['ymin'] + self.detection['ymax']) / 2
                )
        elif isinstance(self.detection, Detection):
            self.center = ((self.detection.points[0][0] + self.detection.points[1][0]) / 2,
                           (self.detection.points[0][1] + self.detection.points[1][1]) / 2)


    def get_left_foot(self, points: np.array):
        x1, y1 = points[0]
        x2, y2 = points[1]

        return [x1, y2]

    def get_right_foot(self, points: np.array):
        return points[1]

    @property
    def left_foot(self):
        points = self.detection.points
        left_foot = self.get_left_foot(points)

        return left_foot

    @property
    def right_foot(self):
        points = self.detection.points
        right_foot = self.get_right_foot(points)

        return right_foot

    @property
    def left_foot_abs(self):
        points = self.detection.absolute_points
        left_foot_abs = self.get_left_foot(points)

        return left_foot_abs

    @property
    def right_foot_abs(self):
        points = self.detection.absolute_points
        right_foot_abs = self.get_right_foot(points)

        return right_foot_abs

    @property
    def feet(self) -> np.ndarray:
        return np.array([self.left_foot, self.right_foot])

    def distance_to_ball(self, ball: Ball) -> float:
        """
        Returns the distance between the player closest foot and the ball

        Parameters
        ----------
        ball : Ball
            Ball object

        Returns
        -------
        float
            Distance between the player closest foot and the ball
        """

        if self.detection is None or ball.center is None:
            return None

        left_foot_distance = np.linalg.norm(ball.center - self.left_foot)
        right_foot_distance = np.linalg.norm(ball.center - self.right_foot)

        return min(left_foot_distance, right_foot_distance)

    def distance_to_player(self, opponent) -> float:
        left_foot_distance = np.linalg.norm(np.array(opponent.left_foot) - np.array(self.left_foot))
        right_foot_distance = np.linalg.norm(np.array(opponent.right_foot) - np.array(self.right_foot))

        return min(left_foot_distance, right_foot_distance)


    def closest_foot_to_ball(self, ball: Ball) -> np.ndarray:
        """

        Returns the closest foot to the ball

        Parameters
        ----------
        ball : Ball
            Ball object

        Returns
        -------
        np.ndarray
            Closest foot to the ball (x, y)
        """

        if self.detection is None or ball.center is None:
            return None

        left_foot_distance = np.linalg.norm(ball.center - self.left_foot)
        right_foot_distance = np.linalg.norm(ball.center - self.right_foot)

        if left_foot_distance < right_foot_distance:
            return self.left_foot

        return self.right_foot

    def closest_foot_to_ball_abs(self, ball: Ball) -> np.ndarray:
        """

        Returns the closest foot to the ball

        Parameters
        ----------
        ball : Ball
            Ball object

        Returns
        -------
        np.ndarray
            Closest foot to the ball (x, y)
        """

        if self.detection is None or ball.center_abs is None:
            return None

        left_foot_distance = np.linalg.norm(ball.center_abs - self.left_foot_abs)
        right_foot_distance = np.linalg.norm(ball.center_abs - self.right_foot_abs)

        if left_foot_distance < right_foot_distance:
            return self.left_foot_abs

        return self.right_foot_abs

    def draw(
        self, frame: PIL.Image.Image, confidence: bool = False, id: bool = False
    ) -> PIL.Image.Image:
        """
        Draw the player on the frame

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame to draw on
        confidence : bool, optional
            Whether to draw confidence text in bounding box, by default False
        id : bool, optional
            Whether to draw id text in bounding box, by default False

        Returns
        -------
        PIL.Image.Image
            Frame with player drawn
        """
        if self.detection is None:
            return frame

        if self.team is not None:
            self.detection.data["color"] = self.team.color

        return Draw.draw_detection(self.detection, frame, confidence=confidence, id=id)

    def draw_pointer(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw a pointer above the player

        Parameters
        ----------
        frame : np.ndarray
            Frame to draw on

        Returns
        -------
        np.ndarray
            Frame with pointer drawn
        """
        if self.detection is None:
            return frame

        color = None

        if self.team:
            color = self.team.color

        return Draw.draw_pointer(detection=self.detection, img=frame, color=color)

    # def __str__(self):
    #     return f"Player: {self.feet}, team: {self.team}"

    def __eq__(self, other: "Player") -> bool:
        if isinstance(self, Player) == False or isinstance(other, Player) == False:
            return False

        self_id = self.detection.data["id"]
        other_id = other.detection.data["id"]

        return self_id == other_id
         
    @staticmethod
    def have_same_id(player1: "Player", player2: "Player") -> bool:
        """
        Check if player1 and player2 have the same ids

        Parameters
        ----------
        player1 : Player
            One player
        player2 : Player
            Another player

        Returns
        -------
        bool
            True if they have the same id
        """
        if not player1 or not player2:
            return False
        if "id" not in player1.detection.data or "id" not in player2.detection.data:
            return False
        return player1.detection.data["id"] == player2.detection.data["id"]

    # --- Static method to draw players ---
    @staticmethod
    def draw_players(
        players: List["Player"],
        frame: PIL.Image.Image,
        id: bool = False,
        target_player_id: Optional[int] = None # <-- Add target ID argument
    ) -> PIL.Image.Image:

        for player in players:
            if player.detection is None: 
                print(f"Player {player.id} has no detection data.")
                print(f"frame: {frame}")
                continue
            
          
                # --- Extract BBox Coordinates (Adapt based on actual detection structure) ---
            x1, y1, x2, y2 = -1, -1, -1, -1
            detection_data = player.detection # This is likely a dict now
                    # Ensure coords are valid numbers before casting
            # if all(pd.notna(detection_data[k]) for k in ['xmin','ymin','xmax','ymax']):
            #     x1 = int(detection_data['xmin']); y1 = int(detection_data['ymin'])
            #     x2 = int(detection_data['xmax']); y2 = int(detection_data['ymax'])

            x1 = int(detection_data.points[0][0]); y1 = int(detection_data.points[0][1])
            x2 = int(detection_data.points[1][0]); y2 = int(detection_data.points[1][1])
            # Add other extraction methods if format varies (e.g., from Norfair object)
            # elif hasattr(player.detection, 'points'): ... 
            
            # Proceed only if valid coordinates were extracted
            if x1 > -1 and x1 < x2 and y1 < y2:
                # --- Determine Color ---
                # Use team color if team exists, otherwise default PIL color
                if str(player.team) == "Team 1":
                    draw_color = (0,0,0)
                elif str(player.team) == "Team 2":
                    draw_color = (255,255,255) 
                else:
                    draw_color = DEFAULT_PLAYER_COLOR_PIL # Default color if no team or unknown team
                # --- Draw Rectangle ---
                frame = Draw.rectangle(frame, (x1, y1), (x2, y2), draw_color, 3)

                # --- Prepare Label ---
                label_items = []
                if id: label_items.append(f"ID:{player.detection.data['id']}")        
                    
                label = " ".join(label_items)
                # --- Draw Label ---
                if label:
                        # Simple text draw using PIL (adjust position/font as needed)
                        draw = ImageDraw.Draw(frame)
                        text_pos = (x1, y2 + 15 if y2 > 15 else y2 + 5) # Position below box
                        try:
                            # Basic font, find better way if needed
                            #font = ImageFont.load_default() 
                            font = ImageFont.truetype("arial.ttf", size=int(20))
                            # Draw text (no background)
                            draw.text(text_pos, label, fill=DEFAULT_TEXT_COLOR_PIL, font=font) 
                        except Exception as font_e:
                            # Fallback draw without font object
                            draw.text(text_pos, label, fill=DEFAULT_TEXT_COLOR_PIL) 


        return frame
    


    @staticmethod
    def draw_pressure(players: List["Player"], frame: PIL.Image.Image, target_id: int) -> PIL.Image.Image:
        """
        Draw the pressure on the frame

        Parameters
        ----------
        players : List[Player]
            List of players
        frame : PIL.Image.Image
            Frame to draw on

        Returns
        -------
        PIL.Image.Image
            Frame with pressure drawn
        """
        for player in players:
            if player.pressure == "None" or player.detection.data["id"] != target_id:
            #if player.detection == "None":
                continue
            else:
                x1 = int(player.detection.points[0][0]); y1 = int(player.detection.points[0][1])
                font = ImageFont.truetype("arial.ttf", size=int(20))
                draw = ImageDraw.Draw(frame)
                text_pos = (x1, y1 - 38 if y1 > 38 else y1 + 5) # Position above box
                draw.text(text_pos, player.pressure, fill=DEFAULT_TEXT_COLOR_PIL, font=font) 

        return frame
    
    @staticmethod
    def draw_scanning(players: List["Player"], frame: PIL.Image.Image, target_id: int) -> PIL.Image.Image:
        """
        Draw the pressure on the frame

        Parameters
        ----------
        players : List[Player]
            List of players
        frame : PIL.Image.Image
            Frame to draw on

        Returns
        -------
        PIL.Image.Image
            Frame with pressure drawn
        """
        for player in players:
            if player.scanning is None or player.detection.data["id"] != target_id:
            #if player.detection == "None":
                continue
            else:
                x1 = int(player.detection.points[0][0]); y1 = int(player.detection.points[0][1])
                x2 = int(player.detection.points[1][0]); y2 = int(player.detection.points[1][1])
                origin=(x1+100, y1-100)
                size=250
                origin_int = tuple(map(int, origin))
                yaw, pitch, roll = player.scanning
                pitch_rad = math.radians(pitch)
                yaw_rad = math.radians(yaw)
                roll_rad = math.radians(roll)

                # Rotation matrices
                R_y = np.array([[math.cos(yaw_rad),  0, math.sin(yaw_rad)],
                                [0,                  1, 0               ],
                                [-math.sin(yaw_rad), 0, math.cos(yaw_rad)]])
                R_x = np.array([[1, 0,                  0                ],
                                [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
                                [0, math.sin(pitch_rad), math.cos(pitch_rad)]])
                R_z = np.array([[math.cos(roll_rad), -math.sin(roll_rad), 0],
                                [math.sin(roll_rad), math.cos(roll_rad),  0],
                                [0,                  0,                  1]])

                # Combined rotation matrix (YXZ order assumed from original code)
                R = R_y @ R_x @ R_z

                # Define 3D Axis endpoints
                axis_points_3d = np.float32([
                    [size, 0, 0],    # X-axis end
                    [0, -size, 0],   # Y-axis end (negative Y for "up")
                    [0, 0, size]     # Z-axis end
                ])

                # Rotate the 3D axes
                rotated_axes_3d = R @ axis_points_3d.T
                rotated_axes_3d = rotated_axes_3d.T

                # Project onto 2D image plane (simple orthographic)
                axis_points_2d = (rotated_axes_3d[:, :2] + origin_int).astype(int)

                # Draw the axes on the current frame
                # Make a copy if you don't want to modify the original input frames list
                draw = ImageDraw.Draw(frame)
                start_pt = tuple(map(int, origin))
                end_pt_x = tuple(axis_points_2d[0])
                end_pt_y = tuple(axis_points_2d[1])
                end_pt_z = tuple(axis_points_2d[2])

                text_pos = (x1, y1 - 25 if y1 > 15 else y1 + 15)
                draw.line([start_pt, end_pt_x], fill=(255, 0, 0), width=20)  # X Red (RGB)
                draw.line([start_pt, end_pt_y], fill=(0, 255, 0), width=20)  # Y Green (RGB)
                draw.line([start_pt, end_pt_z], fill=(0, 0, 255), width=2)  # Z Blue (RGB)

        return frame

    @staticmethod
    def draw_body_orientation(players: List["Player"], frame: PIL.Image.Image, target_id: int) -> PIL.Image.Image:
        """
        Draw the pressure on the frame

        Parameters
        ----------
        players : List[Player]
            List of players
        frame : PIL.Image.Image
            Frame to draw on

        Returns
        -------
        PIL.Image.Image
            Frame with pressure drawn
        """
        for player in players:
            if player.body_orientation == "Unknown" or player.detection.data["id"] != target_id:
            #if player.body_orientation == "Unknown":
                continue
            else:
                x1 = int(player.detection.points[0][0]); y1 = int(player.detection.points[0][1])
                font = ImageFont.truetype("arial.ttf", size=int(20))
                draw = ImageDraw.Draw(frame)
                text_pos = (x1, y1 - 20 if y1 > 20 else y1 + 5) # Position above box
                draw.text(text_pos, player.body_orientation, fill=DEFAULT_TEXT_COLOR_PIL, font=font) 

        return frame

    @staticmethod
    def from_detections(
        detections: List[Detection], teams=List[Team]
    ) -> List["Player"]:
        """
        Create a list of Player objects from a list of detections and a list of teams.

        It reads the classification string field of the detection, converts it to a
        Team object and assigns it to the player.

        Parameters
        ----------
        detections : List[Detection]
            List of detections
        teams : List[Team], optional
            List of teams, by default List[Team]

        Returns
        -------
        List[Player]
            List of Player objects
        """
        players = []
        
        for detection in detections:
            if detection is None:
                continue
            if "classification" in detection.data:
                team_name = detection.data["classification"]
                team = Team.from_name(teams=teams, name=team_name)
                detection.data["team"] = team

            player = Player(detection=detection, id=True, team=team)

            players.append(player)

        return players