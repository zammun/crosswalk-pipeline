import cv2
import numpy as np
import os
import csv
import shutil
from collections import defaultdict
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment


# KALMAN FILTER TRACKING CLASSES

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
        # Faster responsiveness so smoothed coordinates don't lag behind acceleration
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.25
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        self.kf.statePost = np.array([[np.float32(init_pt[0])],
                                      [np.float32(init_pt[1])],
                                      [0.0],
                                      [0.0]], np.float32)
        self.age = 0
        self.time_since_update = 0
        self.hits = 1
        self.raw_x = int(init_pt[0])
        self.raw_y = int(init_pt[1])
        self.smooth_x = int(init_pt[0])
        self.smooth_y = int(init_pt[1])

    def predict(self):
        if self.time_since_update > 0:
            self.kf.statePost[2, 0] *= 0.5
            self.kf.statePost[3, 0] *= 0.5
        pred = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return pred[:2].flatten()

    def update(self, pt):
        est = self.kf.correct(np.array([[np.float32(pt[0])], [np.float32(pt[1])]]))
        self.time_since_update = 0
        self.hits += 1
        self.raw_x = int(pt[0])
        self.raw_y = int(pt[1])
        self.smooth_x = int(est[0, 0])
        self.smooth_y = int(est[1, 0])


class TrackingManager:
    def __init__(self, max_age=35, min_hits=2, distance_thresh=170, max_backward_dy=-50):
        self.max_age = max_age
        self.min_hits = min_hits
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
                    if dy_raw < self.max_backward_dy or dx_raw > 260:
                        continue

                    dx = (p[0] - d[0]) * 0.5
                    dy = (p[1] - d[1]) * 1.0
                    cost_matrix[i, j] = np.hypot(dx, dy)

            trk_ind, det_ind = linear_sum_assignment(cost_matrix)
            unmatched_dets = []

            for t, d in zip(trk_ind, det_ind):
                if cost_matrix[t, d] < self.distance_thresh:
                    self.trackers[t].update(detections[d])
                else:
                    unmatched_dets.append(d)

            for d in range(len(detections)):
                if d not in det_ind:
                    unmatched_dets.append(d)

        for d in unmatched_dets:
            self.trackers.append(KalmanTracker(self.next_id, detections[d]))
            self.next_id += 1

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        active_tracks = []
        for t in self.trackers:
            if t.time_since_update == 0 and t.hits >= self.min_hits:
                active_tracks.append({
                    "id": t.id,
                    "x": t.smooth_x,
                    "y": t.smooth_y,
                    "raw_x": t.raw_x,
                    "raw_y": t.raw_y
                })
        return active_tracks

# HOMOGRAPHY & CENTROID EXTRACTION

def compute_homography():
    src_pts = np.float32([
        [100, 705],   # Bottom-Left (near car hood)
        [1820, 705],  # Bottom-Right (near car hood)
        [480, 480],   # Top-Left (wide enough for shoulders, low enough to stay on flat road)
        [1440, 480]   # Top-Right (wide enough for shoulders, low enough to stay on flat road)
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
    min_total_area=1600,
    max_total_area=200000,
    min_horizontal_span=130,
    max_vertical_span=350,
    max_x_gap=380  # Bridges across a car blocking the center of the lane
):
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (110, 30))
    closed = cv2.morphologyEx(gray_bev, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 250:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Reject vertical lane lines (tall and skinny)
        if h > w * 2.5 and w < 45:
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
        if abs(b["cY"] - y_clusters[-1][-1]["cY"]) < 130:
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

    detections = []
    for cluster in final_clusters:
        total_area = sum(b["area"] for b in cluster)
        span_x = max(b["x2"] for b in cluster) - min(b["x1"] for b in cluster)
        span_y = max(b["y2"] for b in cluster) - min(b["y1"] for b in cluster)

        if not (min_total_area <= total_area <= max_total_area):
            continue
        if span_x < min_horizontal_span or span_y > max_vertical_span:
            continue

        avg_x = int(sum(b["cX"] * b["area"] for b in cluster) / total_area)
        avg_y = int(sum(b["cY"] * b["area"] for b in cluster) / total_area)
        detections.append([avg_x, avg_y])

    return detections


# MAIN LOGIC
def main():
    model = YOLO("best.pt")
    tracker = TrackingManager(max_age=35, min_hits=2, distance_thresh=170, max_backward_dy=-50)

    if os.path.exists("crosswalk_crops"):
        shutil.rmtree("crosswalk_crops")
    os.makedirs("crosswalk_crops", exist_ok=True)

    crops_per_id = defaultdict(int)
    last_crop_frame = defaultdict(lambda: -100)
    last_crop_y = defaultdict(lambda: -100)
    MAX_CROPS_PER_ID = 3
    MIN_CROP_FRAME_GAP = 5
    MIN_CROP_Y_STEP = 25

    video_path = "video.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Starts at 02:25 right before the red-light crosswalk (02:32) and far-side crosswalk (03:01)
    START_TIME_SEC = 145
    if START_TIME_SEC > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, START_TIME_SEC * 1000)

    H, bev_w, bev_h = compute_homography()
    frame_count = int(START_TIME_SEC * fps)

    ROI_Y1, ROI_Y2 = 480, 710
    ROI_X1, ROI_X2 = 200, 1720
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

            results = model(yolo_input_640, task="segment", imgsz=640, conf=0.25, iou=0.5, verbose=False)
            mask_canvas = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

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

            current_detections = get_unified_crosswalk_detections(gray_bev)

            for det in current_detections:
                cv2.circle(bev_mask_vis, (det[0], det[1]), 4, (255, 255, 0), cv2.FILLED)

            tracked_objects = tracker.update(current_detections)

            if frame_count % 90 == 0:
                print(f"[STATUS] Frame: {frame_count} | t: {timestamp_sec}s | Detections: {len(current_detections)} | Active Tracks: {len(tracked_objects)}")

            for obj in tracked_objects:
                cX, cY, obj_id = obj["x"], obj["y"], obj["id"]
                raw_x, raw_y = obj["raw_x"], obj["raw_y"]
                csv_writer.writerow([frame_count, timestamp_sec, obj_id, cX, cY])

                # Center crop on raw_y (actual current position) and verify mask pixels exist in crop
                if (80 <= raw_y <= 430 and 120 <= raw_x <= 680
                        and crops_per_id[obj_id] < MAX_CROPS_PER_ID
                        and (frame_count - last_crop_frame[obj_id]) >= MIN_CROP_FRAME_GAP
                        and abs(raw_y - last_crop_y[obj_id]) >= MIN_CROP_Y_STEP):
                    y1 = max(0, raw_y - 160)
                    y2 = min(bev_h, raw_y + 160)
                    crop = bev_frame[y1:y2, 0:bev_w]
                    mask_crop = gray_bev[y1:y2, 0:bev_w]

                    # Only save if the crop window actually contains detected crosswalk pixels
                    if crop.size > 0 and np.count_nonzero(mask_crop) >= 2000:
                        crop_filename = f"crosswalk_crops/id_{obj_id}_frame_{frame_count}.jpg"
                        cv2.imwrite(crop_filename, crop)
                        crops_per_id[obj_id] += 1
                        last_crop_frame[obj_id] = frame_count
                        last_crop_y[obj_id] = raw_y
                        print(f"--> [SAVED CROP {crops_per_id[obj_id]}/{MAX_CROPS_PER_ID}] {crop_filename} at t={timestamp_sec}s (raw_y={raw_y})")

                for canvas in (bev_mask_vis, bev_display):
                    cv2.circle(canvas, (raw_x, raw_y), 7, (0, 0, 255), cv2.FILLED)
                    cv2.putText(canvas, f"ID: {obj_id}", (raw_x - 25, raw_y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            combined_view = np.hstack((bev_display, bev_mask_vis))
            cv2.imshow("Left: Real Road BEV | Right: Tracked Mask BEV", combined_view)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()