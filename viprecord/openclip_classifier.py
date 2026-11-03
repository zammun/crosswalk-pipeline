import os
import re
import csv
import cv2
import torch
import open_clip
import numpy as np
from PIL import Image
from collections import defaultdict

# 1. Fallback BEV Crop Generation from video.mp4
def generate_test_crops_from_video(video_path="video.mp4", output_dir="crosswalk_crops"):
    """
    Extracts real multi-frame BEV crops directly from video.mp4 at known crosswalk
    timestamps so Step 3 can be tested independently of Step 2.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    # BEV Homography for video.mp4
    src_pts = np.float32([[100, 705], [1820, 705], [580, 500], [1340, 500]])
    bev_w, bev_h = 800, 500
    dst_pts = np.float32([[0, bev_h], [bev_w, bev_h], [0, 0], [bev_w, 0]])
    H, _ = cv2.findHomography(src_pts, dst_pts)

    # 3 consecutive frames per Track ID to test multi-frame score averaging
    sample_tracks = {
        1: [165, 170, 175],       # Transverse crosswalk (00:05.5) (Actually a False Positive with incorrect timings)
        2: [12118, 12122, 12126], # Yellow continental crosswalk (06:44.1)
        3: [12665, 12667, 12669], # Plain road / lane line false positive (07:02.4) Also a FP
        4: [28545, 28548, 28551], # Yellow ladder crosswalk (15:51.9)
        5: [30335, 30339, 30343]  # White continental crosswalk (16:51.6)
    }

    print("Generating multi-frame BEV test crops from video.mp4...")
    for track_id, frames in sample_tracks.items():
        for fnum in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
            ret, frame = cap.read()
            if ret:
                bev_frame = cv2.warpPerspective(frame, H, (bev_w, bev_h))
                crop = bev_frame[100:420, 0:bev_w]  # Sweet-spot full-width crop
                cv2.imwrite(f"{output_dir}/id_{track_id}_frame_{fnum}.jpg", crop)

    cap.release()
    print(f"Saved 15 multi-frame test crops to '{output_dir}/'.\n")


# 2. STEP 3: OPENCLIP MULTI-FRAME CLASSIFIER
def run_openclip_classification(crops_dir="crosswalk_crops", output_csv="classified_crosswalks.csv"):
    # If Step 2 hasn't produced enough crops yet, auto-generate test crops from video.mp4
    existing_imgs = [f for f in os.listdir(crops_dir) if f.endswith(('.jpg', '.png'))] if os.path.exists(crops_dir) else []
    if len(existing_imgs) < 3:
        generate_test_crops_from_video("video.mp4", crops_dir)
        existing_imgs = [f for f in os.listdir(crops_dir) if f.endswith(('.jpg', '.png'))]

    # Group image files by Track ID (parses "id_X_frame_Y.jpg")
    tracks = defaultdict(list)
    for fname in sorted(existing_imgs):
        match = re.search(r"id_(\d+)_frame_(\d+)", fname)
        if match:
            track_id = int(match.group(1))
            tracks[track_id].append(os.path.join(crops_dir, fname))
        else:
            tracks[-1].append(os.path.join(crops_dir, fname))

    # Load OpenCLIP Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading OpenCLIP (ViT-B-32) on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # MUTCD Crosswalk Categories + Negative Road Class
    mutcd_prompts = {
        "Transverse": "a top-down bird's eye view of a road with two thin parallel horizontal crosswalk lines across the asphalt",
        "Continental": "a top-down bird's eye view of a continental zebra crosswalk with thick painted rectangular blocks or stripes",
        "Ladder": "a top-down bird's eye view of a ladder crosswalk with thick vertical bars connected by two horizontal border lines",
        "No Crosswalk (False Positive)": "a top-down bird's eye view of plain gray asphalt road with a single lane line or tree shadows and no crosswalk"
    }
    labels = list(mutcd_prompts.keys())
    text_tokens = tokenizer(list(mutcd_prompts.values())).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "num_frames_averaged", "predicted_mutcd_type", "confidence_pct", "sample_file"])

            print("\n" + "=" * 65)
            print(f"{'TRACK ID':<10} | {'FRAMES':<8} | {'MUTCD CLASSIFICATION':<28} | {'CONFIDENCE'}")
            print("=" * 65)

            for track_id, img_paths in sorted(tracks.items()):
                prob_list = []

                for path in img_paths:
                    img = Image.open(path).convert("RGB")
                    img_tensor = preprocess(img).unsqueeze(0).to(device)

                    img_features = model.encode_image(img_tensor)
                    img_features /= img_features.norm(dim=-1, keepdim=True)

                    # Cosine similarity -> softmax probabilities
                    probs = (100.0 * img_features @ text_features.T).softmax(dim=-1)
                    prob_list.append(probs.cpu().numpy()[0])

                # Average classification scores across all frames for this Track ID
                avg_probs = np.mean(prob_list, axis=0)
                best_idx = int(np.argmax(avg_probs))
                best_label = labels[best_idx]
                best_conf = round(float(avg_probs[best_idx]) * 100, 2)

                print(f"ID: {track_id:<6} | {len(img_paths):<8} | {best_label:<28} | {best_conf:.2f}%")
                writer.writerow([track_id, len(img_paths), best_label, best_conf, os.path.basename(img_paths[0])])

    print("=" * 65)
    print(f"Saved final classifications to '{output_csv}'.")

if __name__ == "__main__":
    run_openclip_classification()

# Geometric methods
# 