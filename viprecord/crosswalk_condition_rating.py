import os
import re
import csv
import json
import base64
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict

# Optional Anthropic SDK for Claude Vision
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# ---------------------------------------------------------------------------
# 1. QUANTITATIVE CV: PAINT-TO-ASPHALT PIXEL RATIO & CONTRAST ANALYSIS
# ---------------------------------------------------------------------------
def analyze_paint_condition_cv(bev_crop):
    """
    Measures crosswalk paint integrity on a top-down BEV crop:
    - Detects both white and yellow MUTCD paint using adaptive LAB + HSV masks.
    - Builds ideal stripe envelopes (closing + convex hulls) to measure
      intact paint % vs. eroded/missing asphalt % inside the crosswalk boundary.
    - Computes paint-vs-asphalt luminance contrast as a fading/retroreflectivity proxy.
    """
    if bev_crop is None or bev_crop.size == 0:
        return None

    h, w = bev_crop.shape[:2]
    # Focus on the central road band where the crosswalk sits
    roi = bev_crop[int(h * 0.15):int(h * 0.85), int(w * 0.05):int(w * 0.95)]

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    L_chan = lab[:, :, 0].astype(np.float32)

    # Estimate local asphalt brightness robustly using the lower 35th percentile
    asphalt_L = float(np.percentile(L_chan, 35))
    asphalt_std = float(np.std(L_chan[L_chan <= np.percentile(L_chan, 50)]))
    adaptive_thresh = min(225.0, max(asphalt_L + max(28.0, 2.2 * asphalt_std), 130.0))

    # White paint: noticeably brighter than local asphalt with low color saturation
    white_mask = ((L_chan >= adaptive_thresh) & (hsv[:, :, 1] < 65)).astype(np.uint8) * 255

    # Yellow paint: standard MUTCD yellow hue (14-38 in OpenCV HSV)
    yellow_mask = (
        (hsv[:, :, 0] >= 14) & (hsv[:, :, 0] <= 38) &
        (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 120)
    ).astype(np.uint8) * 255

    paint_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Remove 1-2px grain without filling real wear holes
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    paint_clean = cv2.morphologyEx(paint_mask, cv2.MORPH_OPEN, open_k)

    # Reconstruct expected intact stripe boundaries to capture tire-track wear & chipping
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    stripe_envelopes = cv2.morphologyEx(paint_clean, cv2.MORPH_CLOSE, close_k)

    contours, _ = cv2.findContours(stripe_envelopes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_envelope_mask = np.zeros_like(stripe_envelopes)
    for cnt in contours:
        if cv2.contourArea(cnt) >= 200:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(valid_envelope_mask, [hull], -1, 255, -1)

    total_envelope_px = int(np.count_nonzero(valid_envelope_mask))
    if total_envelope_px < 300:
        return {
            "intact_paint_pct": 0.0,
            "eroded_asphalt_pct": 100.0,
            "paint_asphalt_contrast": 0.0,
            "cv_score_1_to_5": 1.0,
            "cv_rating_label": "Critical / Not Visible",
            "sharpness": 0.0
        }

    # Ratio of intact painted pixels inside the reconstructed stripe boundaries
    intact_inside = cv2.bitwise_and(paint_clean, valid_envelope_mask)
    intact_px = int(np.count_nonzero(intact_inside))
    intact_ratio = min(1.0, intact_px / float(total_envelope_px))
    eroded_ratio = 1.0 - intact_ratio

    # Contrast / Fading proxy: LAB L-channel difference between paint and adjacent asphalt ring
    dilated_env = cv2.dilate(valid_envelope_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35)))
    asphalt_ring = cv2.bitwise_and(dilated_env, cv2.bitwise_not(valid_envelope_mask))
    if np.count_nonzero(asphalt_ring) > 100:
        local_asphalt_L = float(np.median(L_chan[asphalt_ring > 0]))
    else:
        local_asphalt_L = asphalt_L

    paint_L = float(np.median(L_chan[intact_inside > 0])) if intact_px > 0 else local_asphalt_L
    contrast = max(0.0, paint_L - local_asphalt_L)
    contrast_norm = min(1.0, contrast / 140.0)  # OpenCV LAB L runs 0-255

    # Weighted 1-to-5 condition score (75% intact paint coverage + 25% brightness contrast)
    composite = 0.75 * intact_ratio + 0.25 * contrast_norm
    score_1_to_5 = round(1.0 + 4.0 * composite, 2)

    # Laplacian variance to pick the sharpest crop for Claude Vision
    sharpness = float(cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

    return {
        "intact_paint_pct": round(intact_ratio * 100.0, 1),
        "eroded_asphalt_pct": round(eroded_ratio * 100.0, 1),
        "paint_asphalt_contrast": round(contrast, 1),
        "cv_score_1_to_5": score_1_to_5,
        "cv_rating_label": score_to_label(score_1_to_5),
        "sharpness": sharpness
    }


def score_to_label(score):
    if score >= 4.4:
        return "Excellent"
    elif score >= 3.7:
        return "Good"
    elif score >= 2.9:
        return "Fair (Monitor)"
    elif score >= 2.1:
        return "Poor (Needs Repaint)"
    return "Critical (Immediate Repaint)"


# ---------------------------------------------------------------------------
# 2. CLAUDE VISION INSPECTOR (QUALITATIVE VERIFICATION)
# ---------------------------------------------------------------------------
def inspect_with_claude_vision(image_path, mutcd_type, cv_metrics, model_name="claude-sonnet-4-5-20250929"):
    """
    Sends the sharpest BEV crop + CV pixel-ratio metrics to Claude Vision
    to grade paint wear, chipping, fading, and maintenance priority.
    Returns None cleanly if ANTHROPIC_API_KEY is not set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not HAS_ANTHROPIC or not api_key:
        return None

    try:
        with open(image_path, "rb") as img_f:
            img_b64 = base64.standard_b64encode(img_f.read()).decode("utf-8")

        media_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        client = anthropic.Anthropic(api_key=api_key)

        prompt = (
            f"You are a transportation infrastructure inspector grading a top-down Bird's-Eye View (BEV) "
            f"crop of a {mutcd_type} crosswalk.\n"
            f"Automated computer vision measurements on this crosswalk:\n"
            f"- Intact Paint Ratio: {cv_metrics['intact_paint_pct']}%\n"
            f"- Eroded/Missing Asphalt Inside Stripes: {cv_metrics['eroded_asphalt_pct']}%\n"
            f"- Paint-to-Asphalt Luminance Contrast (0-255): {cv_metrics['paint_asphalt_contrast']}\n\n"
            f"Inspect the image for paint fading/retroreflectivity loss, tire-track wear, chipping, "
            f"asphalt cracking across stripes, or partial occlusion.\n"
            f"Respond ONLY with valid JSON using these exact keys:\n"
            f'{{"claude_score_1_to_5": <float 1.0-5.0>, '
            f'"claude_rating_label": "<Excellent|Good|Fair|Poor|Critical>", '
            f'"defects": ["<list of observed issues or None>"], '
            f'"maintenance_note": "<concise 1-sentence recommendation>"}}'
        )

        message = client.messages.create(
            model=model_name,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_b64
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        )

        raw_text = message.content[0].text.strip()
        # Strip markdown fences if present
        raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
        return json.loads(raw_text)

    except Exception as e:
        print(f"  [Claude Vision Warning] Skipped for {os.path.basename(image_path)}: {e}")
        return None


# ---------------------------------------------------------------------------
# 3. MAIN STEP 5 EXECUTION & MASTER INVENTORY MERGE
# ---------------------------------------------------------------------------
def run_condition_rating(
    classified_csv="classified_crosswalks.csv",
    localized_csv_candidates=("localized_crosswalks_v4.csv", "localized_crosswalks.csv"),
    crops_dir="crosswalk_crops",
    output_csv="condition_ratings.csv",
    master_csv="final_crosswalk_inventory.csv",
    use_claude=True
):
    if not os.path.exists(classified_csv):
        print(f"Error: '{classified_csv}' not found. Run openclip_classifier.py first.")
        return

    cls_df = pd.read_csv(classified_csv)

    # Index all saved BEV crops by Track ID
    crops_by_id = defaultdict(list)
    if os.path.exists(crops_dir):
        for fname in sorted(os.listdir(crops_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                m = re.search(r"id_(\d+)_frame_(\d+)", fname)
                if m:
                    crops_by_id[int(m.group(1))].append(os.path.join(crops_dir, fname))

    claude_active = use_claude and HAS_ANTHROPIC and bool(os.environ.get("ANTHROPIC_API_KEY"))
    mode_str = "Hybrid (CV Pixel-Ratio + Claude Vision)" if claude_active else "Quantitative CV Pixel-Ratio Only (set ANTHROPIC_API_KEY to enable Claude Vision)"
    print(f"\nRunning Step 5 Condition Rating | Mode: {mode_str}")

    print("\n" + "=" * 105)
    print(f"{'ID':<4} | {'MUTCD TYPE':<13} | {'INTACT %':<9} | {'ERODED %':<9} | {'CONTRAST':<9} | {'FINAL SCORE':<12} | {'CONDITION RATING'}")
    print("=" * 105)

    rating_rows = []
    for _, row in cls_df.iterrows():
        track_id = int(row["track_id"])
        mutcd_type = str(row["predicted_mutcd_type"])

        # Skip false positives filtered out in Step 3
        if "False Positive" in mutcd_type or "No Crosswalk" in mutcd_type:
            print(f"ID {track_id:<2} | Skipped (Flagged as False Positive by OpenCLIP)")
            continue

        img_paths = crops_by_id.get(track_id, [])
        if not img_paths and pd.notna(row.get("sample_file")):
            fallback_path = os.path.join(crops_dir, str(row["sample_file"]))
            if os.path.exists(fallback_path):
                img_paths = [fallback_path]

        frame_metrics = []
        best_img_path = None
        best_sharpness = -1.0

        for path in img_paths:
            img = cv2.imread(path)
            metrics = analyze_paint_condition_cv(img)
            if metrics is not None:
                frame_metrics.append(metrics)
                if metrics["sharpness"] > best_sharpness:
                    best_sharpness = metrics["sharpness"]
                    best_img_path = path

        if frame_metrics:
            avg_intact = round(float(np.mean([m["intact_paint_pct"] for m in frame_metrics])), 1)
            avg_eroded = round(float(np.mean([m["eroded_asphalt_pct"] for m in frame_metrics])), 1)
            avg_contrast = round(float(np.mean([m["paint_asphalt_contrast"] for m in frame_metrics])), 1)
            cv_score = round(float(np.mean([m["cv_score_1_to_5"] for m in frame_metrics])), 2)
        else:
            avg_intact, avg_eroded, avg_contrast, cv_score = 0.0, 100.0, 0.0, 1.0

        cv_summary = {
            "intact_paint_pct": avg_intact,
            "eroded_asphalt_pct": avg_eroded,
            "paint_asphalt_contrast": avg_contrast,
            "cv_score_1_to_5": cv_score,
            "cv_rating_label": score_to_label(cv_score)
        }

        # Optional Claude Vision qualitative check on the sharpest BEV crop
        claude_res = None
        if claude_active and best_img_path is not None:
            claude_res = inspect_with_claude_vision(best_img_path, mutcd_type, cv_summary)

        if claude_res is not None:
            claude_score = float(claude_res.get("claude_score_1_to_5", cv_score))
            # Blend 50% objective CV score + 50% Claude Vision inspector score
            final_score = round(0.5 * cv_score + 0.5 * claude_score, 2)
            final_label = claude_res.get("claude_rating_label", score_to_label(final_score))
            defects = "; ".join(claude_res.get("defects", ["None"]))
            note = claude_res.get("maintenance_note", "")
        else:
            claude_score = ""
            final_score = cv_score
            final_label = score_to_label(final_score)
            defects = "High erosion/fading detected (CV)" if avg_eroded >= 30 else "Minimal wear (CV)"
            note = "Schedule repainting" if final_score < 2.9 else "No immediate maintenance required"

        print(f"{track_id:<4} | {mutcd_type:<13} | {avg_intact:<8}% | {avg_eroded:<8}% | {avg_contrast:<9} | {final_score:<12} | {final_label}")

        rating_rows.append({
            "track_id": track_id,
            "mutcd_type": mutcd_type,
            "frames_rated": len(frame_metrics),
            "intact_paint_pct": avg_intact,
            "eroded_asphalt_pct": avg_eroded,
            "paint_asphalt_contrast": avg_contrast,
            "cv_score_1_to_5": cv_score,
            "claude_score_1_to_5": claude_score,
            "final_condition_score": final_score,
            "condition_rating": final_label,
            "defects_flagged": defects,
            "maintenance_note": note,
            "inspected_crop": os.path.basename(best_img_path) if best_img_path else ""
        })

    print("=" * 105)

    # Save Step 5 results
    rating_df = pd.DataFrame(rating_rows)
    rating_df.to_csv(output_csv, index=False)
    print(f"Saved Step 5 condition ratings to '{output_csv}'.")

    # Merge with Step 4 Localization output into one master 5-step asset inventory
    loc_csv = next((p for p in localized_csv_candidates if os.path.exists(p)), None)
    if loc_csv is not None and not rating_df.empty:
        loc_df = pd.read_csv(loc_csv)
        merge_cols = [c for c in rating_df.columns if c not in ("mutcd_type",)]
        master_df = pd.merge(loc_df, rating_df[merge_cols], on="track_id", how="left")
        master_df.to_csv(master_csv, index=False)
        print(f"Merged Steps 1-5 into master inventory: '{master_csv}'.\n")
    else:
        print("Note: Run crosswalk_localization_4.py first to auto-merge into 'final_crosswalk_inventory.csv'.\n")


if __name__ == "__main__":
    run_condition_rating()