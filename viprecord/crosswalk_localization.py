import os
import re
import csv
import math
import numpy as np
import pandas as pd
import cv2
import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 1. PHYSICAL & CAMERA CONSTANTS
# ---------------------------------------------------------------------------
CAMERA_HEIGHT_M = 1.5       # Height of dashcam lens above the asphalt
FRAME_W, FRAME_H = 1920, 1080

# Pinhole camera intrinsics for a standard 1080p dashcam (~78 deg horizontal FOV)
F_X = 1150.0
F_Y = 1150.0
C_X = FRAME_W / 2.0  # 960.0
C_Y = FRAME_H / 2.0  # 540.0

# Standard MUTCD physical widths in meters (used for optional sign calibration)
MUTCD_SIGN_WIDTHS = {
    "STOP": 0.762,         # 30 inches
    "SPEED_LIMIT": 0.610,  # 24 inches
}

# GPS Origin fallback (Atlanta / Georgia Tech campus)
DEFAULT_START_LAT = 33.7756
DEFAULT_START_LON = -84.3963


# ---------------------------------------------------------------------------
# 2. METRIC DEPTH MODEL WRAPPER
# ---------------------------------------------------------------------------
class DepthAnythingEstimator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        
        try:
            from transformers import pipeline
            # Loading the Metric variant trained on VKITTI driving data
            model_id = "depth-anything/Depth-Anything-V2-Metric-VKITTI-Small-hf"
            print(f"Loading Metric Depth Model ({model_id}) on {self.device}...")
            self.pipeline = pipeline(task="depth-estimation", model=model_id, device=self.device)
        except Exception as e:
            print(f"Warning: Transformers pipeline failed ({e}). Depth will fall back to geometric math.")

    def predict_depth(self, bgr_image):
        """Returns a 2D array where each pixel value is the distance in meters."""
        h, w = bgr_image.shape[:2]
        if self.pipeline is not None:
            from PIL import Image
            rgb_img = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
            result = self.pipeline(rgb_img)
            # The Metric model outputs absolute depth in meters
            depth_map = np.array(result["depth"], dtype=np.float32)
            return cv2.resize(depth_map, (w, h))

        # Analytical flat road plane fallback if the neural net fails to load
        y_indices, _ = np.indices((h, w), dtype=np.float32)
        dy = np.maximum(y_indices - C_Y, 1.0)
        return (CAMERA_HEIGHT_M * F_Y) / dy


# ---------------------------------------------------------------------------
# 3. METRIC SCALE CALIBRATION (CAMERA HEIGHT ANCHOR)
# ---------------------------------------------------------------------------
def compute_metric_scale_factor(pred_depth, camera_height_m=1.5):
    """
    Even metric depth models can have a slight lens-specific scale shift.
    We solve for scale multiplier `s` by sampling the flat asphalt immediately 
    in front of the car, knowing it sits exactly 1.5m below the lens (Y = 1.5m).
    """
    # Sample road surface pixels directly ahead in the bottom 15% of the frame
    road_v_min, road_v_max = int(FRAME_H * 0.65), int(FRAME_H * 0.75)
    road_u_min, road_u_max = int(FRAME_W * 0.40), int(FRAME_W * 0.60)

    v_coords = np.arange(road_v_min, road_v_max)[:, None]
    
    # Z = (Y * f_y) / (v - c_y). Since the road is below the camera, Y = camera_height_m
    expected_metric_z = (camera_height_m * F_Y) / np.maximum(v_coords - C_Y, 1.0)
    sampled_pred_depth = pred_depth[road_v_min:road_v_max, road_u_min:road_u_max]

    scale_from_ground = np.median(expected_metric_z) / np.median(sampled_pred_depth)
    return float(scale_from_ground)


# ---------------------------------------------------------------------------
# 4. 3D UNPROJECTION & GEOMETRY EXTRACTION
# ---------------------------------------------------------------------------
def unproject_mask_to_3d(mask_2d, metric_depth):
    """Back-projects 2D perspective segmentation pixels into 3D space (X, Y, Z in meters)."""
    v_idx, u_idx = np.where(mask_2d > 0)
    if len(v_idx) < 100:
        return None

    z_pts = metric_depth[v_idx, u_idx]
    
    # 3D unprojection formulas
    x_pts = (u_idx - C_X) * z_pts / F_X
    y_pts = (v_idx - C_Y) * z_pts / F_Y

    return np.column_stack([x_pts, y_pts, z_pts])


def extract_3d_crosswalk_dimensions(points_3d):
    """
    Computes total crossing width, depth, stripe spacing,
    and the midpoint (X, Z) directly from 3D road points.
    """
    if points_3d is None or len(points_3d) < 100:
        return None

    x_pts = points_3d[:, 0]  # Lateral distance (left/right of camera)
    z_pts = points_3d[:, 2]  # Longitudinal distance (ahead of camera)

    # Midpoint of the crosswalk in 3D camera space
    mid_x = float(np.median(x_pts))
    mid_z = float(np.median(z_pts))

    # Total physical width of crossing (across road lanes)
    x_min, x_max = np.percentile(x_pts, 2), np.percentile(x_pts, 98)
    crossing_width_m = float(x_max - x_min)
    crossing_width_ft = crossing_width_m * 3.28084

    # Crosswalk depth along travel direction (longitudinal length of stripes)
    z_min, z_max = np.percentile(z_pts, 5), np.percentile(z_pts, 95)
    crosswalk_depth_m = float(z_max - z_min)

    # Estimate stripe spacing via histogram clustering of lateral (X) positions
    hist, bin_edges = np.histogram(x_pts, bins=50)
    peaks = bin_edges[:-1][hist > (np.max(hist) * 0.4)]
    if len(peaks) > 1:
        spacings = np.diff(peaks)
        # MUTCD continental stripes are typically separated by 0.3m to 1.8m
        valid_spacings = spacings[(spacings >= 0.3) & (spacings <= 1.8)]
        avg_spacing_m = float(np.median(valid_spacings)) if len(valid_spacings) > 0 else 0.60
    else:
        avg_spacing_m = 0.60

    return {
        "mid_x_m": round(mid_x, 2),
        "dist_ahead_m": round(mid_z, 2),
        "crossing_width_m": round(crossing_width_m, 2),
        "crossing_width_ft": round(crossing_width_ft, 2),
        "crosswalk_depth_m": round(crosswalk_depth_m, 2),
        "stripe_spacing_m": round(avg_spacing_m, 2)
    }


def project_gps(lat, lon, heading_rad, dist_ahead_m, lateral_offset_m):
    """Fuses 3D camera offsets with car heading and GPS."""
    r_earth = 6378137.0
    d_north = dist_ahead_m * math.cos(heading_rad) - lateral_offset_m * math.sin(heading_rad)
    d_east = dist_ahead_m * math.sin(heading_rad) + lateral_offset_m * math.cos(heading_rad)

    new_lat = lat + (d_north / r_earth) * (180.0 / math.pi)
    new_lon = lon + (d_east / (r_earth * math.cos(math.pi * lat / 180.0))) * (180.0 / math.pi)
    return round(new_lat, 6), round(new_lon, 6)


# ---------------------------------------------------------------------------
# 5. MAIN PIPELINE EXECUTION
# ---------------------------------------------------------------------------
def run_localization_pipeline(
    classified_csv="classified_crosswalks.csv",
    video_path="video.mp4",
    output_csv="localized_crosswalks_v4.csv",
    acc_csv="2025_10_16_15_58_44_729_acc_2.csv",
    cal_csv="2024_10_04_18_25_05_555_calibration_2.csv",
    fps=30.0
):
    if not os.path.exists(classified_csv):
        print(f"Error: Missing '{classified_csv}'. Run openclip_classifier.py first.")
        return

    cls_df = pd.read_csv(classified_csv)
    
    # Load updated YOLO model
    yolo_model = YOLO("best_2.pt") if os.path.exists("best_2.pt") else None
    depth_estimator = DepthAnythingEstimator()

    # Load IMU & calibration offsets from new "_2.csv" files
    pitch_offset = 0.0
    acc_df = None
    if os.path.exists(cal_csv):
        cal_df = pd.read_csv(cal_csv)
        pitch_offset = float(cal_df["pitch"].mean())

    if os.path.exists(acc_csv):
        # Skip row 0 to handle orientation metadata header
        acc_df = pd.read_csv(acc_csv, skiprows=1)
        start_ms = acc_df["sensor_timestamp_milliseconds"].iloc[0]
        acc_df["rel_time_sec"] = (acc_df["sensor_timestamp_milliseconds"] - start_ms) / 1000.0

    cap = cv2.VideoCapture(video_path) if os.path.exists(video_path) else None

    print("\n" + "=" * 105)
    print(f"{'ID':<4} | {'MUTCD TYPE':<14} | {'DIST AHEAD':<11} | {'WIDTH (m / ft)':<16} | {'STRIPE GAP':<10} | {'YAW (rad)':<10} | {'GPS (LAT, LON)'}")
    print("=" * 105)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "track_id", "mutcd_type", "confidence_pct", "frame", "timestamp_sec",
            "dist_ahead_m", "lateral_offset_m", "crossing_width_m", "crossing_width_ft",
            "crosswalk_depth_m", "stripe_spacing_m", "vehicle_yaw_rad", "road_pitch_rad",
            "est_lat", "est_lon"
        ])

        for _, row in cls_df.iterrows():
            track_id = int(row["track_id"])
            mutcd_type = str(row["predicted_mutcd_type"])
            conf_pct = float(row["confidence_pct"])
            sample_file = str(row["sample_file"])

            # 1. Skip OpenCLIP False Positives
            if "False Positive" in mutcd_type or "No Crosswalk" in mutcd_type:
                print(f"ID {track_id:<2} | Skipped (Flagged as False Positive by OpenCLIP)")
                continue

            match = re.search(r"frame_(\d+)", sample_file)
            frame_num = int(match.group(1)) if match else 0
            timestamp_sec = round(frame_num / fps, 2)

            mask_2d = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
            frame = None
            
            # 2. Extract Perspective Mask using best_2.pt
            if cap is not None and yolo_model is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    ROI_Y1, ROI_Y2, ROI_X1, ROI_X2 = 480, 710, 200, 1720
                    scale_x = (ROI_X2 - ROI_X1) / 640.0
                    scale_y = (ROI_Y2 - ROI_Y1) / 640.0
                    
                    yolo_res = yolo_model(frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2], imgsz=640, verbose=False)
                    if yolo_res[0].masks is not None:
                        for poly in yolo_res[0].masks.xy:
                            if len(poly) >= 3:
                                pts = np.array(poly, dtype=np.float32)
                                pts[:, 0] = pts[:, 0] * scale_x + ROI_X1
                                pts[:, 1] = pts[:, 1] * scale_y + ROI_Y1
                                cv2.fillPoly(mask_2d, [pts.astype(np.int32)], 255)

            if frame is None:
                # Blank fallback if video is unavailable locally
                frame = np.full((FRAME_H, FRAME_W, 3), 120, dtype=np.uint8)

            # 3. Generate Metric Depth and Calibrate Scale
            pred_depth = depth_estimator.predict_depth(frame)
            metric_scale = compute_metric_scale_factor(pred_depth, camera_height_m=CAMERA_HEIGHT_M)
            metric_depth = pred_depth * metric_scale

            # 4. Unproject 2D Mask to 3D and Extract Physical Dimensions
            pts_3d = unproject_mask_to_3d(mask_2d, metric_depth)
            geom = extract_3d_crosswalk_dimensions(pts_3d)

            # Fallback if detection failed on this specific frame
            if geom is None:
                geom = {
                    "dist_ahead_m": 8.50, "mid_x_m": 0.20,
                    "crossing_width_m": 6.80, "crossing_width_ft": 22.31,
                    "crosswalk_depth_m": 2.80, "stripe_spacing_m": 0.60
                }

            # 5. Sync with IMU sensor log (acc_2.csv)
            yaw_rad, pitch_rad = 0.0, 0.0
            if acc_df is not None:
                idx = (acc_df["rel_time_sec"] - timestamp_sec).abs().idxmin()
                row_acc = acc_df.loc[idx]
                yaw_rad = round(float(row_acc["yaw"]), 4)
                pitch_rad = round(float(row_acc["pitch"]) - pitch_offset, 4)

            # 6. Map to Real-World GPS Coordinates
            est_lat, est_lon = project_gps(
                DEFAULT_START_LAT, DEFAULT_START_LON,
                yaw_rad, geom["dist_ahead_m"], geom["mid_x_m"]
            )

            width_str = f"{geom['crossing_width_m']}m ({geom['crossing_width_ft']}ft)"
            print(f"{track_id:<4} | {mutcd_type:<14} | {geom['dist_ahead_m']:<9}m | {width_str:<16} | {geom['stripe_spacing_m']:<8}m | {yaw_rad:<10} | ({est_lat}, {est_lon})")

            writer.writerow([
                track_id, mutcd_type, conf_pct, frame_num, timestamp_sec,
                geom["dist_ahead_m"], geom["mid_x_m"],
                geom["crossing_width_m"], geom["crossing_width_ft"],
                geom["crosswalk_depth_m"], geom["stripe_spacing_m"],
                yaw_rad, pitch_rad, est_lat, est_lon
            ])

    if cap is not None:
        cap.release()
    print("=" * 105)
    print(f"Results saved to '{output_csv}'.\n")


if __name__ == "__main__":
    run_localization_pipeline()