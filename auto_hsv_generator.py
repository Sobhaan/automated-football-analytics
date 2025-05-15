import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from matplotlib import pyplot as plt

def _get_dominant_colors_from_crop(
    crop_bgr: np.ndarray, 
    k: int = 1, 
    min_saturation: int = 0, min_value: int = 0, max_value: int = 240,
    min_pixels_for_kmeans: int = 0
) -> List[Tuple[int, int, int]]:
    if crop_bgr is None or crop_bgr.size < min_pixels_for_kmeans * 3:
        return []
    img_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask_s = img_hsv[:, :, 1] >= min_saturation
    mask_v = (img_hsv[:, :, 2] >= min_value) & (img_hsv[:, :, 2] <= max_value)
    filtered = img_hsv[mask_s & mask_v]
    if len(filtered) < k:
        filtered = img_hsv.reshape(-1, 3)
        if len(filtered) < k:
            return []
    actual_k = min(k, len(np.unique(filtered, axis=0)))
    if actual_k == 0:
        return []
    try:
        kmeans = KMeans(n_clusters=actual_k, random_state=0, n_init='auto', algorithm='lloyd')
        kmeans.fit(filtered)
        centers = kmeans.cluster_centers_.astype(int)
    except Exception:
        centers = np.unique(filtered, axis=0)[:actual_k].astype(int)
    return [tuple(c) for c in centers]

def get_dominant_hsv_kmeans(image_bgr, k=3):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).reshape(-1,3)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(hsv)
    counts = np.bincount(kmeans.labels_)
    # find the largest cluster
    best = np.argmax(counts)
    center = kmeans.cluster_centers_[best].astype(int)
    return [tuple(center)]




def generate_auto_hsv_filters(
    player_detector_model: Any,
    frames: List[np.ndarray], 
    team_names: List[str] = ["Team 1", "Team 2"],
    referee_name: Optional[str] = "Referee",
    detections_conf_threshold: float = 0.4, 
    colors_to_extract_per_player_crop: int = 1,
    colors_to_define_per_team: int = 1,
    hsv_variance: Tuple[int, int, int] = (10, 60, 60), 
    min_s_filter: int = 0,
    min_v_filter: int = 0
) -> List[Dict[str, Any]]:
    print(f"DEBUG: Generating HSV filters automatically using {len(frames)} frames...")
    all_colors: List[Tuple[int, int, int]] = []

    for frame_idx, frame in enumerate(frames):
        df = player_detector_model.predict(frame)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if 'name' in df.columns:
            df = df[df['name'] == 'soccer-player']
        if 'confidence' in df.columns:
            df = df[df['confidence'] >= detections_conf_threshold]
        for _, row in df.iterrows():
            try:
                x1, y1, x2, y2 = map(int, (row['xmin'], row['ymin'], row['xmax'], row['ymax']))
            except KeyError:
                continue
            h, w = frame.shape[:2]
            # x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x1 >= x2 or y1 >= y2:
                continue
            crop = frame[y1:y2, x1:x2]
            crop_h, crop_w = crop.shape[:2]
            torso_y1_rel = int(crop_h * 0.3) # Start at 10% from top
            torso_y2_rel = int(crop_h * 0.5) # End at 40% from top (keeps middle 35% of height)
            torso_x1_rel = int(crop_w * 0.4) # Crop 30% from left
            torso_x2_rel = int(crop_w * 0.6) # Crop 20% from right (keeps middle 70% of width)
            torso_y1_rel = max(0,torso_y1_rel); torso_y2_rel = min(crop_h,torso_y2_rel)
            torso_x1_rel = max(0,torso_x1_rel); torso_x2_rel = min(crop_w,torso_x2_rel)
            tor = crop[torso_y1_rel:torso_y2_rel, torso_x1_rel:torso_x2_rel]
            # plt.figure(figsize=(4,4))
            # # convert BGR → RGB for display
            # plt.imshow(cv2.cvtColor(tor, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.show()

            colors = _get_dominant_colors_from_crop(tor, k=colors_to_extract_per_player_crop)
            all_colors.extend(colors)

    print(f"DEBUG: Collected {len(all_colors)} HSV samples for clustering.")
    all_np = np.array(all_colors, dtype=np.float32)

    num_teams = len(team_names)
    actual_clusters = min(num_teams, len(all_np))
    kmeans_teams = KMeans(n_clusters=actual_clusters, random_state=0, n_init='auto', algorithm='lloyd')
    labels = kmeans_teams.fit_predict(all_np)
    centers = kmeans_teams.cluster_centers_.astype(int)
    print(f"DEBUG: Team cluster centroids HSV: {centers}")

    h_var, s_var, v_var = hsv_variance
    output = []
    for i in range(actual_clusters):
        team = team_names[i]
        h_c, s_c, v_c = centers[i]
        # circular hue variance
        lower_h = int((h_c - h_var) % 180)
        upper_h = int((h_c + h_var) % 180)
        ls = max(min_s_filter, int(s_c - s_var))
        us = min(255, int(s_c + s_var))
        lv = max(min_v_filter, int(v_c - v_var))
        uv = min(255, int(v_c + v_var))
        color_def = {
            "name": f"{team}",
            "lower_hsv": (lower_h, ls, lv),
            "upper_hsv": (upper_h, us, uv)
        }
        output.append({"name": team, "colors": [color_def]})

    for t in team_names[actual_clusters:]:
        output.append({"name": t, "colors": []})
    if referee_name:
        output.append({"name": referee_name, "colors": [{"name": "dummy_ref", "lower_hsv": (0, 0, 0), "upper_hsv": (179, 50, 50)}]})

    print(f"Final Automatically Generated Filters: {output}")
    return output
