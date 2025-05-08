from typing import Iterable, List

import numpy as np
import PIL

from soccer.ball import Ball
from soccer.draw import AbsolutePath, PathPoint
from soccer.player import Player
from soccer.team import Team


class Pass:
    def __init__(
        self, start_ball_bbox: np.ndarray, end_ball_bbox: np.ndarray, team: Team,
        passer_id: int,          
        receiver_id: int,        
        initiation_frame: int,   
        reception_frame: int
    ) -> None:
        # Abs coordinates
        self.start_ball_bbox = start_ball_bbox
        self.end_ball_bbox = end_ball_bbox
        self.passer_id = passer_id             
        self.receiver_id = receiver_id          
        self.initiation_frame = initiation_frame  
        self.reception_frame = reception_frame
        self.team = team
        self.draw_abs = AbsolutePath()

    def draw(
        self, img: PIL.Image.Image, coord_transformations: "CoordinatesTransformation"
    ) -> PIL.Image.Image:
        """Draw a pass

        Parameters
        ----------
        img : PIL.Image.Image
            Video frame
        coord_transformations : CoordinatesTransformation
            coordinates transformation

        Returns
        -------
        PIL.Image.Image
            frame with the new pass
        """
        rel_point_start = PathPoint.from_abs_bbox(
            id=0,
            abs_point=self.start_ball_bbox,
            coord_transformations=coord_transformations,
            color=self.team.color,
        )
        rel_point_end = PathPoint.from_abs_bbox(
            id=1,
            abs_point=self.end_ball_bbox,
            coord_transformations=coord_transformations,
            color=self.team.color,
        )

        new_pass = [rel_point_start, rel_point_end]

        pass_filtered = self.draw_abs.filter_points_outside_frame(
            path=new_pass,
            width=img.size[0],
            height=img.size[0],
            margin=3000,
        )

        if len(pass_filtered) == 2:
            img = self.draw_abs.draw_arrow(
                img=img, points=pass_filtered, color=self.team.color, width=6, alpha=150
            )

        return img

    @staticmethod
    def draw_pass_list(
        img: PIL.Image.Image,
        passes: List["Pass"],
        coord_transformations: "CoordinatesTransformation",
    ) -> PIL.Image.Image:
        """Draw all the passes

        Parameters
        ----------
        img : PIL.Image.Image
            Video frame
        passes : List[Pass]
            Passes list to draw
        coord_transformations : CoordinatesTransformation
            Coordinate transformation for the current frame

        Returns
        -------
        PIL.Image.Image
            Drawed frame
        """
        for pass_ in passes:
            img = pass_.draw(img=img, coord_transformations=coord_transformations)

        return img

    def get_relative_coordinates(
        self, coord_transformations: "CoordinatesTransformation"
    ) -> tuple:
        """
        Print the relative coordinates of a pass

        Parameters
        ----------
        coord_transformations : CoordinatesTransformation
            Coordinates transformation

        Returns
        -------
        tuple
            (start, end) of the pass with relative coordinates
        """
        relative_start = coord_transformations.abs_to_rel(self.start_ball_bbox)
        relative_end = coord_transformations.abs_to_rel(self.end_ball_bbox)

        return (relative_start, relative_end)

    def get_center(self, points: np.array) -> tuple:
        """
        Returns the center of the points

        Parameters
        ----------
        points : np.array
            2D points

        Returns
        -------
        tuple
            (x, y) coordinates of the center
        """
        x1, y1 = points[0]
        x2, y2 = points[1]

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        return (center_x, center_y)

    def round_iterable(self, iterable: Iterable) -> Iterable:
        """
        Round all entries from one Iterable object

        Parameters
        ----------
        iterable : Iterable
            Iterable to round

        Returns
        -------
        Iterable
            Rounded Iterable
        """
        return [round(item) for item in iterable]

    def generate_output_pass(
        self, start: np.ndarray, end: np.ndarray, team_name: str,
        passer_id, receiver_id, initiation_frame, reception_frame
    ) -> str:
        """
        Generate a string with the pass information

        Parameters
        ----------
        start : np.ndarray
            The start point of the pass
        end : np.ndarray
            The end point of the pass
        team_name : str
            The team that did this pass

        Returns
        -------
        str
            String with the pass information
        """
        relative_start_point = self.get_center(start)
        relative_end_point = self.get_center(end)

        relative_start_round = self.round_iterable(relative_start_point)
        relative_end_round = self.round_iterable(relative_end_point)

        return f"Start: {relative_start_round}, End: {relative_end_round}, Team: {team_name}, Passer ID: {passer_id}, Receiver ID: {receiver_id}, Initiation Frame: {initiation_frame}, Reception Frame: {reception_frame}"

    def tostring(self, coord_transformations: "CoordinatesTransformation") -> str:
        """
        Get a string with the relative coordinates of this pass

        Parameters
        ----------
        coord_transformations : CoordinatesTransformation
            Coordinates transformation

        Returns
        -------
        str
            string with the relative coordinates
        """
        relative_start, relative_end = self.get_relative_coordinates(
            coord_transformations
        )

        return self.generate_output_pass(relative_start, relative_end, self.team.name)

    def __str__(self):
        return self.generate_output_pass(
            self.start_ball_bbox, self.end_ball_bbox, self.team.name, self.passer_id, self.receiver_id, self.initiation_frame, self.reception_frame
        )


class PassEvent:
    def __init__(self) -> None:
        self.ball = None
        # Player currently closest (might not yet meet possession threshold)
        self.current_closest_player = None 
        # Player currently being tracked for possession streak
        self.player_starting_possession_streak = None 
        self.player_with_ball_counter = 0
        
        # --- State for Pass Detection ---
        # Player who *last* met the possession threshold
        self.last_player_confirmed_possession = None 
        # Frame index when the above player last met the threshold
        self.last_possession_confirmation_frame = -1 
        
        self.player_with_ball_threshold = 3  # Min frames to confirm possession
        # Threshold for detecting change of possession between teams (not used in pass logic below)
        # self.player_with_ball_threshold_dif_team = 4 
        self.current_frame_idx = -1 # Store current frame index

    # Update method now just stores current state
    def update(self, closest_player: Player, ball: Ball, frame_idx: int) -> None:
        """Stores current frame info and updates possession streak counter."""
        self.ball = ball
        self.current_closest_player = closest_player
        self.current_frame_idx = frame_idx

        # Update possession streak counter
        if closest_player and Player.have_same_id(self.player_starting_possession_streak, closest_player):
            # Same player is still closest, increment streak
            self.player_with_ball_counter += 1
        else:
            # Closest player changed (or is None), reset streak
            self.player_with_ball_counter = 1 # Start counter at 1 for the new player
            self.player_starting_possession_streak = closest_player 

    def validate_pass(self, start_player: Player, end_player: Player) -> bool:
        """
        Check if there is a pass between two players of the same team

        Parameters
        ----------
        start_player : Player
            Player that originates the pass
        end_player : Player
            Destination player of the pass

        Returns
        -------
        bool
            Valid pass occurred
        """
        if Player.have_same_id(start_player, end_player):
            return False
        if start_player.team != end_player.team:
            return False

        return True

    def generate_pass(
        self, team: Team, start_pass: np.ndarray, end_pass: np.ndarray,
        passer_id: int,         
        receiver_id: int,       
        initiation_frame: int,  
        reception_frame: int    
   
    ) -> Pass:
        """
        Generate a new pass

        Parameters
        ----------
        team : Team
            Pass team
        start_pass : np.ndarray
            Pass start point
        end_pass : np.ndarray
            Pass end point

        Returns
        -------
        Pass
            The generated instance of the Pass class
        """
        start_pass_bbox = [start_pass, start_pass]
        # end_pass_bbox = [end_pass, end_pass]

        new_pass = Pass(
            start_ball_bbox=start_pass_bbox,
            end_ball_bbox=end_pass,
            team=team,
            passer_id=passer_id,              
            receiver_id=receiver_id,          
            initiation_frame=initiation_frame,   
            reception_frame=reception_frame      
        )

        return new_pass

    def process_pass(self) -> "Pass | None": # Return Pass object if detected, else None
        """
        Checks if the current player confirmed possession and if this constitutes
        a pass from the previously confirmed player. Updates confirmed possession state.
        Returns the generated Pass object if a pass occurred, otherwise None.
        """
        new_pass = None # Initialize as no pass detected this frame

        # Check if the player we've been tracking just confirmed possession
        if self.current_closest_player and self.player_with_ball_counter == self.player_with_ball_threshold:
            
            # This player just confirmed possession (the potential receiver)
            receiver = self.current_closest_player
            receiver_confirmation_frame = self.current_frame_idx

            # Who was the last player confirmed to have possession? (the potential passer)
            passer = self.last_player_confirmed_possession
            passer_confirmation_frame = self.last_possession_confirmation_frame

            # --- Validate Pass Attempt ---
            # Need a valid passer, different players, same team
            if (passer is not None and 
                not Player.have_same_id(passer, receiver) and 
                passer.team == receiver.team):

                # --- Valid Pass Detected! ---
                team = receiver.team # Team performing the pass
                
                if team is None or self.ball is None or self.ball.detection is None:
                    print(f"Frame {self.current_frame_idx}: Error generating pass - Missing team/ball info.")
                else:
                    # Estimate pass start/end points (can be refined)
                    # Start: Passer's foot/center when they were last confirmed?
                    # For simplicity, let's use passer's last known foot position
                    start_coord = passer.closest_foot_to_ball_abs(self.ball) # Needs ball state from THAT frame ideally, but use current ball as approximation
                    if start_coord is None: start_coord = passer.center # Fallback to center
                    
                    # End: Ball position now (when receiver confirmed)
                    end_coord = self.ball.detection.absolute_points 

                    # --- Generate the Pass object ---
                    passer_id = self.last_player_confirmed_possession.detection.data.get("id")
                    receiver_id = self.current_closest_player.detection.data.get("id")
                    # Initiation frame = frame passer was last confirmed (best estimate)
                    initiation_frame = passer_confirmation_frame 
                    reception_frame = receiver_confirmation_frame

                    # Assuming Pass class and generate_pass are updated (see step 3 & 4)
                    new_pass = self.generate_pass(
                        team=team, 
                        start_pass=start_coord, 
                        end_pass=end_coord,
                        passer_id=passer_id,         
                        receiver_id=receiver_id,     
                        initiation_frame=initiation_frame, 
                        reception_frame=reception_frame 
                    )
                    
                    # Add pass to the team's list
                    team.passes.append(new_pass)
                    print(f"Frame {self.current_frame_idx}: Pass Detected! {passer_id} ({passer.team.name}) -> {receiver_id} ({receiver.team.name}). Initiated around frame {initiation_frame}.")

            # --- Update State for Next Frame ---
            # Regardless of whether a pass happened, the current player is now the last confirmed possessor.
            self.last_player_confirmed_possession = receiver
            self.last_possession_confirmation_frame = receiver_confirmation_frame
            
        # If the counter is below threshold, player hasn't confirmed possession yet.
        # If closest_player is None, also no confirmed possession.
            
        return new_pass
