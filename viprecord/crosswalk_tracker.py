import cv2
import numpy as np
import os
import csv
import shutil
from collections import defaultdict
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# 1. KALMAN FILTER TRACKING CLASSES WITH TEMPORAL CONFIRMATION
# ---------------------------------------------------------------------------
class KalmanTracker:
    def __init__(self, track_id, init_pt):
        self.id = track_id
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        
        # Responsive tuning so smoothed coordinates track acceleration smoothly
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.25
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        self.kf.statePost = np.array([[np.float32(init_pt[0])],
                                      [np.float32(init_pt[1])],
                                      [0.0],
                                      [0.0]], np.float32)
        
        self.age = 1
        self.time_since_update = 0
        self.hits = 1
        self.consecutive_hits = 1
        self.state = "TENTATIVE"  # Starts tentative until confirmed in X frames
        
        self.raw_x = int(init_pt[0])
        self.raw_y = int(init_pt[1])
        self.smooth_x = int(init_pt[0])
        self.smooth_y = int(init_pt[1])

    def predict(self):
        if self.time_since_update > 0:
            self.kf.statePost[2, 0] *= 0.5
            self.kf.statePost[3, 0] *= 0.5
            self.consecutive_hits = 0

        pred = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return pred[:2].flatten()

    def update(self, pt):
        est = self.kf.correct(np.array([[np.float32(pt[0])], [np.float32(pt[1])]]))
        self.time_since_update = 0
        self.hits += 1
        self.consecutive_hits += 1
        
        self.raw_x = int(pt[0])
        self.raw_y = int(pt[1])
        self.smooth_x = int(est[0, 0])
        self.smooth_y = int(est[1, 0])


class TrackingManager:
    def __init__(self, max_age=30, min_hits=8, distance_thresh=160, max_backward_dy=-25):
        self.max_age = max_age
        self.min_hits = min_hits  # Must be visible in >= 8 frames before confirmation
        self.distance_thresh = distance_thresh
        self.max_backward_dy = max_backward_dy
        self.trackers = []
        self.next_id = 1

    def update(self, detections):
        preds = [t.predict() for t in self.trackers]
        unmatched_dets = list(range(len(detections)))

        if len(self.trackers) > 0 and len(detections) > 0:
            cost_matrix = np.full((len(self.trackers), len(detections)), 1e6, dtype=np.float32)
            for i, (trk, p) in enumerate(zip(self.trackers, preds)):
                for j, d in enumerate(detections):
                    dy_raw = d[1] - trk.raw_y
                    dx_raw = abs(d[0] - trk.raw_x)

                    # Kinematic gating: Crosswalks move down the screen (+y) as car drives forward
                    if dy_raw < self.max_backward_dy or dy_raw > 150 or dx_raw > 190:
                        continue

                    dx = (p[0] - d[0]) * 0.5
                    dy = (p[1] - d[1]) * 1.0
                    cost_matrix[i, j] = np.hypot(dx, dy)

            trk_ind, det_ind = linear_sum_assignment(cost_matrix)
            unmatched_dets = []

            for t, d in zip(trk_ind, det_ind):
                if cost_matrix[t, d] < self.distance_thresh:
                    self.trackers[t].update(detections[d])
                    if self.trackers[t].hits >= self.min_hits:
                        self.trackers[t].state = "CONFIRMED"
                else:
                    unmatched_dets.append(d)

            for d in range(len(detections)):
                if d not in det_ind:
                    unmatched_dets.append(d)

        for d in unmatched_dets:
            self.trackers.append(KalmanTracker(self.next_id, detections[d]))
            self.next_id += 1

        # Prune dead tracks and return dead IDs for buffer clean-up
        alive_trackers = []
        dead_ids = []
        for t in self.trackers:
            if t.hits < self.min_hits and t.time_since_update > 5:
                dead_ids.append(t.id)
            elif t.time_since_update <= self.max_age:
                alive_trackers.append(t)
            else:
                dead_ids.append(t.id)
        self.trackers = alive_trackers

        active_tracks = []
        for t in self.trackers:
            if t.time_since_update == 0:
                active_tracks.append({
                    "id": t.id,
                    "x": t.smooth_x,
                    "y": t.smooth_y,
                    "raw_x": t.raw_x,
                    "raw_y": t.raw_y,
                    "hits": t.hits,
                    "confirmed": t.hits >= self.min_hits
                })
        return active_tracks, dead_ids


# ---------------------------------------------------------------------------
# 2. DYNAMIC HOMOGRAPHY & ADVANCED DETECTION FILTERS
# ---------------------------------------------------------------------------
def compute_homography(frame_w=1920, frame_h=1080):
    """Automatically scales homography points to any dashcam resolution (720p, 1080p, 4K)."""
    src_pts = np.float32([
        [frame_w * (100.0 / 1920.0),  frame_h * (705.0 / 1080.0)],  # Bottom-Left
        [frame_w * (1820.0 / 1920.0), frame_h * (705.0 / 1080.0)],  # Bottom-Right
        [frame_w * (480.0 / 1920.0),  frame_h * (480.0 / 1080.0)],  # Top-Left
        [frame_w * (1440.0 / 1920.0), frame_h * (480.0 / 1080.0)]   # Top-Right
    ])
    bev_w, bev_h = 800, 500
    dst_pts = np.float32([
        [0, bev_h],
        [bev_w, bev_h],
        [0, 0],
        [bev_w, 0]
    ])
    H_matrix, _ = cv2.findHomography(src_pts, dst_pts)
    return H_matrix, bev_w, bev_h


def get_unified_crosswalk_detections(
    gray_bev,
    bev_frame=None,
    min_total_area=1800,
    max_total_area=220000,
    min_horizontal_span=150,
    max_vertical_span=350,
    min_aspect_ratio=1.15,
    max_x_gap=380,
    min_paint_std=16.0
):
    """
    Applies morphological closing, aspect ratio gating, fill ratio checks,
    and real-pixel contrast verification to filter false positives.
    """
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (110, 30))
    closed = cv2.morphologyEx(gray_bev, cv2.MORPH_CLOSE, close_kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 250:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out vertical lane lines (tall and skinny)
        if h > w * 2.2 and w < 50:
            continue

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
            blobs.append({
                "cX": cX, "cY": cY, "area": area,
                "x1": x, "x2": x + w, "y1": y, "y2": y + h
            })

    if not blobs:
        return []

    blobs.sort(key=lambda b: b["cY"])
    y_clusters = [[blobs[0]]]
    for b in blobs[1:]:
        if abs(b["cY"] - y_clusters[-1][-1]["cY"]) < 125:
            y_clusters[-1].append(b)
        else:
            y_clusters.append([b])

    final_clusters = []
    for y_cluster in y_clusters:
        y_cluster.sort(key=lambda b: b["x1"])
        sub_cluster = [y_cluster[0]]
        max_x2 = y_cluster[0]["x2"]
        for b in y_cluster[1:]:
            if (b["x1"] - max_x2) <= max_x_gap:
                sub_cluster.append(b)
                max_x2 = max(max_x2, b["x2"])
            else:
                final_clusters.append(sub_cluster)
                sub_cluster = [b]
                max_x2 = b["x2"]
        final_clusters.append(sub_cluster)

    bev_gray_real = cv2.cvtColor(bev_frame, cv2.COLOR_BGR2GRAY) if bev_frame is not None else None

    detections = []
    for cluster in final_clusters:
        total_area = sum(b["area"] for b in cluster)
        x1 = min(b["x1"] for b in cluster)
        x2 = max(b["x2"] for b in cluster)
        y1 = min(b["y1"] for b in cluster)
        y2 = max(b["y2"] for b in cluster)
        span_x = x2 - x1
        span_y = y2 - y1

        if not (min_total_area <= total_area <= max_total_area):
            continue
        if span_x < min_horizontal_span or span_y > max_vertical_span:
            continue
        # Crosswalk must be wider across road than deep along lane
        if (span_x / max(1.0, float(span_y))) < min_aspect_ratio:
            continue

        envelope_area = max(1.0, float(span_x * span_y))
        if (total_area / envelope_area) < 0.18:
            continue

        # Contrast check against underlying road pixels (rejects shadows/dark spots)
        if bev_gray_real is not None:
            roi_real = bev_gray_real[max(0, y1):min(bev_gray_real.shape[0], y2),
                                     max(0, x1):min(bev_gray_real.shape[1], x2)]
            if roi_real.size > 0 and float(np.std(roi_real)) < min_paint_std:
                continue

        avg_x = int(sum(b["cX"] * b["area"] for b in cluster) / total_area)
        avg_y = int(sum(b["cY"] * b["area"] for b in cluster) / total_area)
        detections.append([avg_x, avg_y])

    return detections


# ---------------------------------------------------------------------------
# 3. MAIN TRACKING & BUFFERED CROP SAVING LOOP
# ---------------------------------------------------------------------------
def main(video_path="video.mp4", start_time_sec=0):
    model = YOLO("best.pt")
    tracker = TrackingManager(max_age=30, min_hits=8, distance_thresh=160, max_backward_dy=-25)

    if os.path.exists("crosswalk_crops"):
        shutil.rmtree("crosswalk_crops")
    os.makedirs("crosswalk_crops", exist_ok=True)

    crops_per_id = defaultdict(int)
    pending_crops = defaultdict(list)  # Buffers crops until track reaches min_hits=8
    last_crop_frame = defaultdict(lambda: -100)
    last_crop_y = defaultdict(lambda: -100)
    MAX_CROPS_PER_ID = 3
    MIN_CROP_FRAME_GAP = 5
    MIN_CROP_Y_STEP = 25

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

    if start_time_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)

    H, bev_w, bev_h = compute_homography(frame_w, frame_h)
    frame_count = int(start_time_sec * fps)

    # Dynamic ROI scaled to video resolution
    ROI_Y1 = int(round(frame_h * (480.0 / 1080.0)))
    ROI_Y2 = int(round(frame_h * (710.0 / 1080.0)))
    ROI_X1 = int(round(frame_w * (200.0 / 1920.0)))
    ROI_X2 = int(round(frame_w * (1720.0 / 1920.0)))
    ROI_W = ROI_X2 - ROI_X1
    ROI_H = ROI_Y2 - ROI_Y1

    scale_x = ROI_W / 640.0
    scale_y = ROI_H / 640.0

    with open("tracking_data.csv", "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "timestamp_sec", "track_id", "bev_x", "bev_y"])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            timestamp_sec = round(frame_count / fps, 2)

            road_patch = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
            yolo_input_640 = cv2.resize(road_patch, (640, 640))

            results = model(yolo_input_640, task="segment", imgsz=640, conf=0.28, iou=0.5, verbose=False)
            mask_canvas = np.zeros((frame_h, frame_w), dtype=np.uint8)

            if results[0].masks is not None:
                for poly in results[0].masks.xy:
                    if len(poly) >= 3:
                        pts = np.array(poly, dtype=np.float32)
                        pts[:, 0] = pts[:, 0] * scale_x + ROI_X1
                        pts[:, 1] = pts[:, 1] * scale_y + ROI_Y1
                        if cv2.contourArea(pts) > 0.70 * (ROI_W * ROI_H):
                            continue
                        cv2.fillPoly(mask_canvas, [pts.astype(np.int32)], 255)

            gray_bev = cv2.warpPerspective(mask_canvas, H, (bev_w, bev_h))
            bev_frame = cv2.warpPerspective(frame, H, (bev_w, bev_h))
            bev_display = bev_frame.copy()
            bev_mask_vis = cv2.cvtColor(gray_bev, cv2.COLOR_GRAY2BGR)

            current_detections = get_unified_crosswalk_detections(gray_bev, bev_frame)

            for det in current_detections:
                cv2.circle(bev_mask_vis, (det[0], det[1]), 4, (255, 255, 0), cv2.FILLED)

            tracked_objects, dead_ids = tracker.update(current_detections)

            # Flush out crop buffers for tentative tracks that died before confirmation
            for d_id in dead_ids:
                pending_crops.pop(d_id, None)

            confirmed_count = sum(1 for o in tracked_objects if o["confirmed"])
            if frame_count % 90 == 0:
                print(f"[STATUS] Frame: {frame_count} | t: {timestamp_sec}s | Dets: {len(current_detections)} | Confirmed Tracks: {confirmed_count}")

            for obj in tracked_objects:
                cX, cY, obj_id = obj["x"], obj["y"], obj["id"]
                raw_x, raw_y = obj["raw_x"], obj["raw_y"]
                is_confirmed = obj["confirmed"]

                total_collected = crops_per_id[obj_id] + len(pending_crops[obj_id])
                if (80 <= raw_y <= 430 and 120 <= raw_x <= 680
                        and total_collected < MAX_CROPS_PER_ID
                        and (frame_count - last_crop_frame[obj_id]) >= MIN_CROP_FRAME_GAP
                        and abs(raw_y - last_crop_y[obj_id]) >= MIN_CROP_Y_STEP):
                    y1 = max(0, raw_y - 160)
                    y2 = min(bev_h, raw_y + 160)
                    crop = bev_frame[y1:y2, 0:bev_w]
                    mask_crop = gray_bev[y1:y2, 0:bev_w]

                    if crop.size > 0 and np.count_nonzero(mask_crop) >= 2200:
                        crop_filename = f"crosswalk_crops/id_{obj_id}_frame_{frame_count}.jpg"
                        pending_crops[obj_id].append((crop_filename, crop.copy(), timestamp_sec, raw_y))
                        last_crop_frame[obj_id] = frame_count
                        last_crop_y[obj_id] = raw_y

                if is_confirmed:
                    csv_writer.writerow([frame_count, timestamp_sec, obj_id, cX, cY])

                    while pending_crops[obj_id] and crops_per_id[obj_id] < MAX_CROPS_PER_ID:
                        c_fname, c_img, c_t, c_ry = pending_crops[obj_id].pop(0)
                        cv2.imwrite(c_fname, c_img)
                        crops_per_id[obj_id] += 1
                        print(f"--> [SAVED CROP {crops_per_id[obj_id]}/{MAX_CROPS_PER_ID}] {c_fname} at t={c_t}s (raw_y={c_ry})")

                    for canvas in (bev_mask_vis, bev_display):
                        cv2.circle(canvas, (raw_x, raw_y), 7, (0, 0, 255), cv2.FILLED)
                        cv2.putText(canvas, f"ID: {obj_id}", (raw_x - 25, raw_y - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    for canvas in (bev_mask_vis, bev_display):
                        cv2.circle(canvas, (raw_x, raw_y), 5, (0, 165, 255), cv2.FILLED)
                        cv2.putText(canvas, f"Wait ({obj['hits']}/{tracker.min_hits})", (raw_x - 35, raw_y - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            combined_view = np.hstack((bev_display, bev_mask_vis))
            cv2.imshow("Left: Real Road BEV | Right: Tracked Mask BEV", combined_view)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()