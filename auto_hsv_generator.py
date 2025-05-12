# auto_hsv_generator.py
import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional # Added Optional

# (Your _get_dominant_colors_from_crop function remains the same)
def _get_dominant_colors_from_crop(
    crop_bgr: np.ndarray, 
    k: int = 2, 
    min_saturation: int = 50, min_value: int = 50, max_value: int = 240,
    min_pixels_for_kmeans: int = 100
) -> List[Tuple[int, int, int]]:
    # ... (your existing implementation) ...
    if crop_bgr is None or crop_bgr.size < min_pixels_for_kmeans * 3 : return []
    img_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask_s = img_hsv[:, :, 1] >= min_saturation
    mask_v_lower = img_hsv[:, :, 2] >= min_value
    mask_v_upper = img_hsv[:, :, 2] <= max_value
    combined_mask = mask_s & mask_v_lower & mask_v_upper
    pixels_hsv_filtered = img_hsv[combined_mask]
    if len(pixels_hsv_filtered) < k:
        pixels_hsv_filtered = img_hsv.reshape(-1, 3)
        if len(pixels_hsv_filtered) < k: return []
    actual_k = min(k, len(np.unique(pixels_hsv_filtered, axis=0)))
    if actual_k == 0: return []
    try:
        kmeans = KMeans(n_clusters=actual_k, random_state=0, n_init='auto', algorithm='lloyd')
        kmeans.fit(pixels_hsv_filtered)
        return [tuple(color) for color in kmeans.cluster_centers_.astype(int)]
    except: return [tuple(color) for color in np.unique(pixels_hsv_filtered, axis=0)[:actual_k]]


def generate_auto_hsv_filters(
    player_detector_model: Any,
    frames: List[np.ndarray], 
    team_names: List[str] = ["Team 1", "Team 2"],
    referee_name: Optional[str] = "Referee", # Made optional
    detections_conf_threshold: float = 0.4, 
    colors_to_extract_per_player_crop: int = 2,
    colors_to_define_per_team: int = 1,    
    hsv_variance: Tuple[int, int, int] = (10, 60, 60), 
    min_s_filter: int = 40,
    min_v_filter: int = 40
) -> List[Dict[str, Any]]:
    print(f"DEBUG: Generating HSV filters automatically using {len(frames)} frames...")
    all_player_dominant_colors_hsv = [] 
    
    for frame_idx, frame in enumerate(frames):
        # print(f"  DEBUG: Processing frame {frame_idx+1}/{len(frames)} for filter generation...")
        player_detections_df = player_detector_model.predict(frame) 

        if not isinstance(player_detections_df, pd.DataFrame) or player_detections_df.empty:
            # print(f"  DEBUG: No detections or not a DataFrame in frame {frame_idx}.")
            continue

        df_filtered = player_detections_df.copy()
        if 'name' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['name'] == 'soccer-player']
        if 'confidence' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['confidence'] >= detections_conf_threshold]
        
        # print(f"  DEBUG: Frame {frame_idx} - Found {len(df_filtered)} filtered players.")
        for p_idx, (_, row) in enumerate(df_filtered.iterrows()):
            try:
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            except KeyError: continue
            
            crop_h_orig, crop_w_orig = frame.shape[:2]
            crop_x1=max(0,x1); crop_y1=max(0,y1); crop_x2=min(crop_w_orig,x2); crop_y2=min(crop_h_orig,y2)
            if crop_x1 >= crop_x2 or crop_y1 >= crop_y2: continue
            
            player_crop_bgr = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if player_crop_bgr.size == 0: continue
            
            h_crop, w_crop = player_crop_bgr.shape[:2]
            torso_y1 = int(h_crop * 0.15); torso_y2 = int(h_crop * 0.85)
            torso_crop_bgr = player_crop_bgr[torso_y1:torso_y2, :]
            if torso_crop_bgr.size == 0: torso_crop_bgr = player_crop_bgr
            
            dominant_colors = _get_dominant_colors_from_crop(
                torso_crop_bgr, k=colors_to_extract_per_player_crop
            )
            # --- DEBUG: Dominant colors from each player crop ---
            if dominant_colors:
                print(f"    DEBUG: Frame {frame_idx}, Player Crop {p_idx+1} ({x1},{y1}-{x2},{y2}), Dominant HSV: {dominant_colors}")
            # ---
            all_player_dominant_colors_hsv.extend(dominant_colors)
    
    if not all_player_dominant_colors_hsv:
        # ... (your existing fallback for no colors found) ...
        print("WARNING: No dominant colors extracted from any player across initial frames.")
        dummy_color_def = {"name": "dummy_auto_color", "lower_hsv": (0,0,0), "upper_hsv": (1,1,1)}
        output_filters = [{"name": name, "colors": [dummy_color_def.copy() for _ in range(colors_to_define_per_team)]} for name in team_names]
        if referee_name: output_filters.append({"name": referee_name, "colors": [{"name": "dummy_ref", "lower_hsv": (0,0,0), "upper_hsv": (179,50,50)}]})
        return output_filters

    print(f"  DEBUG: Total {len(all_player_dominant_colors_hsv)} dominant color samples collected across frames for clustering.")
    
    num_teams = len(team_names)
    all_colors_np = np.array(all_player_dominant_colors_hsv, dtype=np.float32)
    unique_colors_for_clustering = np.unique(all_colors_np, axis=0)
    actual_num_clusters = min(num_teams, len(unique_colors_for_clustering))

    if actual_num_clusters == 0: 
        # ... (your existing fallback for no unique colors) ...
        print("WARNING: No unique colors to cluster. Returning dummy filters.")
        dummy_color_def = {"name": "dummy_auto_color", "lower_hsv": (0,0,0), "upper_hsv": (1,1,1)}
        output_filters = [{"name": name, "colors": [dummy_color_def.copy() for _ in range(colors_to_define_per_team)]} for name in team_names]
        if referee_name: output_filters.append({"name": referee_name, "colors": [{"name": "dummy_ref", "lower_hsv": (0,0,0), "upper_hsv": (179,50,50)}]})
        return output_filters
        
    if actual_num_clusters < num_teams:
        print(f"WARNING: Not enough distinct color groups for {num_teams} teams. Found {actual_num_clusters}.")
        team_names_to_use = team_names[:actual_num_clusters]
    else:
        team_names_to_use = team_names

    kmeans_teams = KMeans(n_clusters=actual_num_clusters, random_state=0, n_init='auto', algorithm='lloyd')
    team_assignment_labels = kmeans_teams.fit_predict(unique_colors_for_clustering) 
    
    # --- DEBUG: Print team cluster centroids ---
    print(f"  DEBUG: Main Team Cluster Centroids (HSV): {kmeans_teams.cluster_centers_}")
    # ---

    output_filters = []
    h_var, s_var, v_var = hsv_variance

    for i in range(actual_num_clusters):
        cluster_colors_hsv = unique_colors_for_clustering[team_assignment_labels == i]
        
        # --- DEBUG: Colors assigned to this team cluster ---
        print(f"  DEBUG: Team Cluster {i+1} ('{team_names_to_use[i]}') has these initial dominant colors (HSV):")
        for c in cluster_colors_hsv: print(f"    {c}")
        # ---

        team_color_definitions = []
        if len(cluster_colors_hsv) > 0:
            num_final_defs = min(colors_to_define_per_team, len(cluster_colors_hsv))
            representative_colors = []
            if len(cluster_colors_hsv) >= num_final_defs and num_final_defs > 0 :
                # Re-cluster within this team's colors to get the most representative ones
                try:
                    sub_kmeans = KMeans(n_clusters=num_final_defs, random_state=0, n_init='auto', algorithm='lloyd')
                    sub_kmeans.fit(cluster_colors_hsv)
                    representative_colors = sub_kmeans.cluster_centers_.astype(int)
                except ValueError: # Not enough distinct samples for sub-clustering
                     representative_colors = np.unique(cluster_colors_hsv, axis=0)[:num_final_defs].astype(int)
            else:
                representative_colors = cluster_colors_hsv[:num_final_defs].astype(int) # Take what's available

            # --- DEBUG: Representative colors chosen for this team's filter definitions ---
            print(f"    DEBUG: Representative HSV colors for '{team_names_to_use[i]}' filters: {representative_colors}")
            # ---

            for j, hsv_center in enumerate(representative_colors):
                h,s,v = hsv_center
                lower_h=max(0,int(h-h_var)); upper_h=min(179,int(h+h_var))
                lower_s=max(min_s_filter,int(s-s_var)); upper_s=min(255,int(s+s_var))
                lower_v=max(min_v_filter,int(v-v_var)); upper_v=min(255,int(v+v_var))
                team_color_definitions.append({
                    "name": f"{team_names_to_use[i].replace(' ','_')}_auto_color{j+1}",
                    "lower_hsv": (lower_h, lower_s, lower_v),
                    "upper_hsv": (upper_h, upper_s, upper_v),
                })
        output_filters.append({"name": team_names_to_use[i], "colors": team_color_definitions})
    
    # ... (rest of the function: add placeholders, referee filter) ...
    for k_idx in range(actual_num_clusters, len(team_names)):
        output_filters.append({"name": team_names[k_idx], "colors": []})
    if referee_name:
        output_filters.append({"name": referee_name, "colors": [{"name": "dummy_ref", "lower_hsv": (0,0,0), "upper_hsv": (179,50,50)}]})
            
    print(f"Final Automatically Generated Filters (to be used by HSVClassifier): {output_filters}")
    return output_filters