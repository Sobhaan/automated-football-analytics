import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from matplotlib import pyplot as plt

def _get_dominant_colors_from_crop(
    crop_bgr: np.ndarray, 
    k: int = 1, 
    min_saturation: int = 25, min_value: int = 25, max_value: int = 255,
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
    colors_to_extract_per_player_crop: int = 2, # User confirmed 2 is good
    colors_to_define_per_team: int = 1, # This param seems unused in the color_def creation
    hsv_variance: Tuple[int, int, int] = (15, 60, 60), # Current: (H,S,V). Consider (15, 40, 40) for maroon later
    min_s_filter: int = 20, # Min S for the *final generated filter range*. Consider ~50-70 for maroon.
    min_v_filter: int = 20  # Min V for the *final generated filter range*. Consider ~50-70 for maroon.
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
            
            if x1 >= x2 or y1 >= y2:
                continue
            
            crop = frame[y1:y2, x1:x2]
            crop_h, crop_w = crop.shape[:2]

            # Current torso crop (30%-60% H, 30%-60% W).
            # Consider previously suggested larger crop if this is too small/unrepresentative:
            # torso_y1_rel = int(crop_h * 0.20); torso_y2_rel = int(crop_h * 0.65)
            # torso_x1_rel = int(crop_w * 0.25); torso_x2_rel = int(crop_w * 0.75)
            torso_y1_rel = int(crop_h * 0.3)
            torso_y2_rel = int(crop_h * 0.5)
            torso_x1_rel = int(crop_w * 0.5)
            torso_x2_rel = int(crop_w * 0.6)
            
            torso_y1_rel = max(0, torso_y1_rel); torso_y2_rel = min(crop_h, torso_y2_rel)
            torso_x1_rel = max(0, torso_x1_rel); torso_x2_rel = min(crop_w, torso_x2_rel)
            
            if torso_y1_rel >= torso_y2_rel or torso_x1_rel >= torso_x2_rel: # Check for valid crop
                continue
            tor = crop[torso_y1_rel:torso_y2_rel, torso_x1_rel:torso_x2_rel]

            # Call _get_dominant_colors_from_crop
            # Consider passing slightly higher min_saturation/min_value here for cleaner initial samples
            dominant_colors_from_crop = _get_dominant_colors_from_crop(
                tor,
                k=colors_to_extract_per_player_crop,
                min_saturation=25, # Example: To avoid overly desaturated pixels in crop KMeans
                min_value=25,      # Example: To avoid very dark pixels in crop KMeans
                max_value=255,
                min_pixels_for_kmeans=10 # Consider minimum pixels
            )
            all_colors.extend(dominant_colors_from_crop)

    print(f"DEBUG: Collected {len(all_colors)} HSV samples for clustering.")
    if not all_colors:
        print("ERROR: No colors collected from crops. Cannot generate HSV filters.")
        return []
        
    all_np = np.array(all_colors, dtype=np.float32)
    print(f"DEBUG: Shape of all_np before pre-filtering: {all_np.shape}")

    if all_np.shape[0] == 0: # Should be caught by the check on all_colors
        print("ERROR: all_np is empty. Cannot generate filters.")
        return []

    # --- BEGIN PRE-FILTERING STEP for Team-Level KMeans ---
    all_np_to_cluster = all_np # Default to using all samples

    # Define minimum saturation and value for samples to be considered by the main team KMeans
    # Tune these values; they are crucial for cleaning the input to KMeans.
    pre_filter_min_s = 50  # Example: Maroon should be reasonably saturated.
    pre_filter_min_v = 50  # Example: Maroon shouldn't be almost black.
    
    if all_np.shape[1] == 3: # Ensure it's HSV (3 columns)
        filter_mask = (all_np[:, 1] >= pre_filter_min_s) & (all_np[:, 2] >= pre_filter_min_v)
        all_np_pre_filtered = all_np[filter_mask]
        
        print(f"DEBUG: Shape of all_np after pre-filtering with S>={pre_filter_min_s}, V>={pre_filter_min_v}: {all_np_pre_filtered.shape}")

        num_teams_to_find = len(team_names) # Typically 2

        if all_np_pre_filtered.shape[0] >= num_teams_to_find: # Check if enough samples left for num_teams clusters
            all_np_to_cluster = all_np_pre_filtered
            print(f"DEBUG: Using pre-filtered all_np for team KMeans ({all_np_to_cluster.shape[0]} samples).")
        else:
            # Fallback to unfiltered if too few samples remain, but print a warning
            print(f"WARNING: Pre-filtering left too few samples ({all_np_pre_filtered.shape[0]}) for {num_teams_to_find} teams. Using original unfiltered all_np ({all_np_to_cluster.shape[0]} samples) for team KMeans.")
    else:
        print("DEBUG: all_np does not have 3 columns or is empty; skipping pre-filtering. Using original.")
    # --- END PRE-FILTERING STEP ---

    num_teams = len(team_names)
    # Determine number of clusters based on available data after potential filtering
    actual_clusters = min(num_teams, all_np_to_cluster.shape[0] if all_np_to_cluster.size > 0 else 0)

    if actual_clusters < num_teams:
        print(f"WARNING: Not enough distinct color samples ({all_np_to_cluster.shape[0]} after filtering) to find {num_teams} teams. Will find {actual_clusters} team(s).")
    if actual_clusters == 0:
        print("ERROR: No color samples available for KMeans clustering after filtering. Cannot generate filters.")
        return []

    kmeans_teams = KMeans(n_clusters=actual_clusters, random_state=0, n_init=10, algorithm='lloyd')
    labels = kmeans_teams.fit_predict(all_np_to_cluster)
    centers = kmeans_teams.cluster_centers_.astype(int)
    print(f"DEBUG: Team cluster centroids HSV (from {all_np_to_cluster.shape[0]} samples): {centers}")

    # Visualizing the clustered data (optional, but highly recommended for debugging)
    # This plotting code should use 'all_np_to_cluster', 'labels', and 'centers'
    if all_np_to_cluster.shape[0] > 0 and all_np_to_cluster.shape[1] == 3 and 'labels' in locals() and 'centers' in locals():
        try:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.scatter(all_np_to_cluster[:, 0], all_np_to_cluster[:, 1], c=labels, cmap='viridis', alpha=0.6, s=10)
            plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, color='red', label='Cluster Centers')
            plt.xlabel('Hue')
            plt.ylabel('Saturation')
            plt.title(f'Team Clusters (H vs S) - {all_np_to_cluster.shape[0]} samples')
            plt.xlim(0, 180); plt.ylim(0, 256)
            plt.legend(); plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.scatter(all_np_to_cluster[:, 1], all_np_to_cluster[:, 2], c=labels, cmap='viridis', alpha=0.6, s=10)
            plt.scatter(centers[:, 1], centers[:, 2], marker='X', s=200, color='red', label='Cluster Centers')
            plt.xlabel('Saturation')
            plt.ylabel('Value')
            plt.title(f'Team Clusters (S vs V) - {all_np_to_cluster.shape[0]} samples')
            plt.xlim(0, 256); plt.ylim(0, 256)
            plt.legend(); plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error during plotting team clusters: {e}")


    h_var, s_var, v_var = hsv_variance
    output = []
    for i in range(actual_clusters): # Iterate up to the number of clusters actually found
        team = team_names[i]
        h_c, s_c, v_c = centers[i]
        
        lower_h = int((h_c - h_var)) # Calculate raw lower hue
        upper_h = int((h_c + h_var)) # Calculate raw upper hue

        # Handle hue wraparound for display and range generation
        # The cv2.inRange will be split in HSVClassifier if lower_h_final > upper_h_final after modulo
        # For now, we just calculate the effective range, modulo is for keeping it 0-179 conceptually
        # but the actual range might cross 0.
        
        # For the filter definition, we'll store the potentially wrapped values.
        # The actual splitting logic for cv2.inRange is in HSVClassifier.
        # Here, we define the range bounds correctly.
        
        # Example: center 5, var 10 -> range -5 to 15.
        # For cv2, this means (0, S, V) to (15, S, V) AND (175, S, V) to (179, S, V)
        # The lower_h, upper_h for the filter def should reflect this intent.
        # The current modulo logic handles the numbers but not the split range concept directly.
        # The critical part is that HSVClassifier handles `lower_h_val[0] > upper_h_val[0]`
        
        final_lower_h = int((h_c - h_var) % 180)
        final_upper_h = int((h_c + h_var) % 180)

        # The min_s_filter and min_v_filter are the ABSOLUTE minimums for the generated range
        ls = max(min_s_filter, int(s_c - s_var))
        us = min(255, int(s_c + s_var))
        lv = max(min_v_filter, int(v_c - v_var))
        uv = min(255, int(v_c + v_var))
        
        # Ensure upper is greater than lower for S and V, if not, use centroid or min/max
        if us < ls: us = max(ls, s_c) # Or simply us = ls if you want a point?
        if uv < lv: uv = max(lv, v_c)


        color_def = {
            "name": f"{team}", # Original was f"{team}_color_{j}" if colors_to_define_per_team > 1
            "lower_hsv": (final_lower_h, ls, lv),
            "upper_hsv": (final_upper_h, us, uv)
        }
        output.append({"name": team, "colors": [color_def]})

    # Handle teams for which no cluster was found (if actual_clusters < num_teams)
    for i in range(actual_clusters, num_teams):
        print(f"WARNING: No cluster found for team: {team_names[i]}. Appending empty color definition.")
        output.append({"name": team_names[i], "colors": []}) # Or a default dummy filter

    if referee_name:
        # Check if referee name was accidentally one of the team_names that didn't get a cluster
        ref_already_handled = any(team_entry["name"] == referee_name for team_entry in output)
        if not ref_already_handled:
             output.append({"name": referee_name, "colors": [{"name": "dummy_ref", "lower_hsv": (0, 0, 0), "upper_hsv": (179, 50, 50)}]})


    print(f"Final Automatically Generated Filters: {output}")
    # if 'vis' in globals() and callable(vis): # Check if vis is defined
    vis(output) # Your visualization function
    return output

def vis(updated_color_data):
    # Function to create a sample HSV image and convert to RGB for display
    def create_color_patch(lower_hsv, upper_hsv, size=(100, 100)):
        h = np.linspace(lower_hsv[0], upper_hsv[0], size[1])
        s = np.linspace(lower_hsv[1], upper_hsv[1], size[0])
        v = np.linspace(lower_hsv[2], upper_hsv[2], size[0])
        H, S = np.meshgrid(h, s)
        _, V = np.meshgrid(h, v)
        hsv_img = np.stack((H, S, V), axis=-1).astype(np.uint8)
        rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        return rgb_img

    # Plotting the updated color ranges
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    for i, team in enumerate(updated_color_data):
        color_range = team['colors'][0]
        patch = create_color_patch(color_range['lower_hsv'], color_range['upper_hsv'])
        axes[i].imshow(patch)
        axes[i].set_title(f"{team['name']}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()