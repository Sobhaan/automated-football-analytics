from typing import List

import numpy as np
import pandas as pd
import torch

from inference.base_detector import BaseDetector
from ultralytics import YOLO

class YoloV8(BaseDetector):
    def __init__(
        self,
        model_path: str = None,
    ):
        """
        Initialize detector

        Parameters
        ----------
        model_path : str, optional
            Path to model, by default None. If it's None, it will download the model with COCO weights
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO(
                "yolov8x"
            )

    def predict(self, input_image: List[np.ndarray]) -> pd.DataFrame:
        """
        Predicts the bounding boxes of the objects in the image

        Parameters
        ----------
        input_image : List[np.ndarray]
            List of input images

        Returns
        -------
        pd.DataFrame
            DataFrame containing the bounding boxes
        """

        result = self.model(input_image, imgsz=1280, device=self.device)

        if result and len(result) > 0:
            results_obj = result[0]  # Get the Results object for the first image
            boxes = results_obj.boxes  # Access the Boxes object

            if boxes is not None and len(boxes) > 0:
                # Extract bounding boxes (xyxy), confidence scores, and class IDs
                xyxy_data = boxes.xyxy.cpu().numpy()
                conf_data = boxes.conf.cpu().numpy()
                cls_data = boxes.cls.cpu().numpy()
                
                # Get the dictionary mapping class IDs to class names
                class_names = results_obj.names
                
                # Prepare data for DataFrame creation
                data_for_df = []
                for i in range(len(xyxy_data)):
                    xmin, ymin, xmax, ymax = xyxy_data[i]
                    confidence = conf_data[i]
                    class_id = int(cls_data[i])
                    # Look up the class name using the class ID
                    name = class_names.get(class_id, f'class_{class_id}') 
                    
                    data_for_df.append([xmin, ymin, xmax, ymax, confidence, class_id, name])
                    
                # Create the Pandas DataFrame with the standard columns
                output_df = pd.DataFrame(data_for_df, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'])
                
            else:
                # No boxes detected, return an empty DataFrame with correct columns
                output_df = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'])
                
        else:
            # No results returned from the model, return an empty DataFrame
            output_df = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'])

        return output_df # Return the created DataFrame
