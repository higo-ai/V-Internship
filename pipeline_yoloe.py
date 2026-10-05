"""
VidVRD Standalone Panoramic Full-Frame Tracking Pipeline (pipeline_yoloe.py)
-------------------------------------------------------------------------------
100% Unified YOLOE-26m Architecture with 60 classes from configs/s_objects.json
- Synchronized Robust Temporal Tracklet Stitching (TTS) with pipeline_roi_crop.py
- Canonical Foreground Person & Scene Object ID assignment ([1] person, [2] person, ...)
- Full-Frame Panoramic Set-of-Marks Rendering (NO ROI crop/cluster boxes)
- Sleek Alpha-Blending Badges (70% tint, 30% transparency), Soft Bounding Boxes
- Zero Interior Redundant ID Occlusion (eliminated face/torso label clutter)
- Codec: avc1 (H.264) for universal Windows Media Player / Web / VS Code playback
- Exports high-quality annotated video to: data/video_processed/{video_id}_yoloe_annotated.mp4
- Extracts clean visual evaluation preview frames to: data/preview/{video_id}/{video_id}_yoloe/
-------------------------------------------------------------------------------
"""

import os
import sys
import json
import cv2
import math
import numpy as np
import torch
import argparse
from ultralytics import YOLO

# Spatial Box Utilities
from modules.spatial_clustering import (
    compute_box_edge_distance,
    compute_box_iou
)

STATIC_FIXTURE_CLASSES = {
    "table", "bench", "chair", "refrigerator", "sofa", "bed",
    "toilet", "sink", "microwave", "oven", "screen", "stool",
    "stop_sign", "traffic_light", "electric_fan", "faucet"
}

# ------------------------------------------------------------------------------
# CLI ARGUMENTS
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="VidVRD Panoramic Full-Frame Tracking Pipeline (YOLOE-26m)")
parser.add_argument("--video", type=str, default="data/videos/video7.mp4", help="Path to input surveillance video")
parser.add_argument("--start_sec", type=float, default=114.0, help="Start time in seconds for golden segment")
parser.add_argument("--end_sec", type=float, default=130.0, help="End time in seconds for golden segment")
parser.add_argument("--num_frames", type=int, default=8, help="Number of preview frames to extract for inspection")
parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold for interactive object detection (default: 0.20)")
parser.add_argument("--iou", type=float, default=0.35, help="IoU threshold for ByteTrack person tracking (default: 0.35)")
parser.add_argument("--stride", type=int, default=1, help="Frame step stride for object detection (default: 1)")
parser.add_argument("--device", type=str, default="auto", help="Compute device: 'auto', 'cuda', or 'cpu'")
parser.add_argument("--preview_dir", type=str, default=None, help="Directory to save full-frame preview images")
cli_args, _ = parser.parse_known_args()

DEVICE = "cuda" if (cli_args.device == "auto" and torch.cuda.is_available()) else ("cpu" if cli_args.device == "auto" else cli_args.device)

# ------------------------------------------------------------------------------
# 1. PATHS AND CONFIGURATIONS
# ------------------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(base_dir, cli_args.video) if not os.path.isabs(cli_args.video) else cli_args.video
if not os.path.exists(video_path):
    alt_data = os.path.join(base_dir, "data", "videos", os.path.basename(video_path))
    if os.path.exists(alt_data):
        video_path = alt_data
    elif alt_data.lower().endswith(".avi") and os.path.exists(alt_data[:-4] + ".mp4"):
        video_path = alt_data[:-4] + ".mp4"

video_basename = os.path.splitext(os.path.basename(video_path))[0]
output_video_dir = os.path.join(base_dir, "data", "video_processed")
os.makedirs(output_video_dir, exist_ok=True)
output_video_path = os.path.join(output_video_dir, f"{video_basename}_yoloe_annotated.mp4")

# Preview directory setup
if cli_args.preview_dir:
    preview_dir = cli_args.preview_dir
else:
    preview_dir = os.path.join(base_dir, "data", "preview", video_basename, f"{video_basename}_yoloe")
os.makedirs(preview_dir, exist_ok=True)

# Clean previous preview frames in target directory
for f in os.listdir(preview_dir):
    if f.endswith(".jpg") or f.endswith(".png"):
        os.remove(os.path.join(preview_dir, f))

START_SEC = cli_args.start_sec
END_SEC = cli_args.end_sec
NUM_PREVIEW_FRAMES = cli_args.num_frames
CONF_THRESH = cli_args.conf
IOU_THRESH = cli_args.iou

print("=" * 80)
print("VIDVRD STANDALONE PANORAMIC FULL-FRAME TRACKING PIPELINE (YOLOE-26M)")
print(f"Target Video:        {video_path}")
print(f"Temporal Window:     {START_SEC}s -> {END_SEC}s (Duration: {END_SEC - START_SEC:.1f}s)")
print(f"Compute Device:      {DEVICE.upper()}")
print(f"Confidence Thresh:   {CONF_THRESH}")
print(f"IoU Thresh:          {IOU_THRESH}")
print(f"Output Video:        {output_video_path}")
print(f"Preview Frames Dir:  {preview_dir}")
print("=" * 80)

# Load Official Project Taxonomies
with open(os.path.join(base_dir, "configs", "s_objects.json"), "r", encoding="utf-8") as f:
    allowed_objects_60 = json.load(f)

with open(os.path.join(base_dir, "configs", "relations.json"), "r", encoding="utf-8") as f:
    relations_list = json.load(f)

# ------------------------------------------------------------------------------
# 2. MODEL INITIALIZATION
# ------------------------------------------------------------------------------
yoloe_weights = os.path.join(base_dir, "models", "yoloe-26m-seg.pt")
if not os.path.exists(yoloe_weights):
    yoloe_weights = os.path.join(base_dir, "yoloe-26m-seg.pt")
model = YOLO(yoloe_weights)
model.to(DEVICE)
model.set_classes(allowed_objects_60)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
start_frame = int(START_SEC * fps)
end_frame = min(int(END_SEC * fps), total_frames - 1)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

ret, first_f = cap.read()
if not ret:
    raise RuntimeError(f"Cannot read video at frame {start_frame}")
orig_h, orig_w = first_f.shape[:2]
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# ------------------------------------------------------------------------------
# 3. PASS 1: FORWARD TRACKING & TRAJECTORY ACCUMULATION
# ------------------------------------------------------------------------------
print("\n--- Pass 1: Forward Tracking & Detection ---")
bytetrack_config = os.path.join(base_dir, "configs", "custom_bytetrack.yaml")
COLOR_PALETTE = [
    (0, 165, 255), (50, 205, 50), (255, 191, 0), (238, 130, 238),
    (255, 20, 147), (0, 255, 255), (147, 112, 219), (0, 215, 255)
]

frame_detections = []
active_object_tracklets = []
raw_clean_frames = {}

for f_idx in range(start_frame, end_frame + 1):
    ret, frame = cap.read()
    if not ret:
        break
    raw_clean_frames[f_idx] = (f_idx / fps, frame)

    track_results = model.track(
        source=frame,
        persist=True,
        tracker=bytetrack_config,
        conf=0.20,
        iou=IOU_THRESH,
        verbose=False,
        device=DEVICE
    )

    frame_persons = []
    frame_objects = []
    if track_results and track_results[0].boxes:
        d_boxes = track_results[0].boxes.xyxy.cpu().numpy().astype(int)
        d_clses = track_results[0].boxes.cls.cpu().numpy().astype(int)
        d_confs = track_results[0].boxes.conf.cpu().numpy().astype(float)
        d_ids = track_results[0].boxes.id.cpu().numpy().astype(int) if track_results[0].boxes.id is not None else [None] * len(d_boxes)

        for b, c_idx, c_val, tid in zip(d_boxes, d_clses, d_confs, d_ids):
            c_name = allowed_objects_60[c_idx]
            bw = b[2] - b[0]
            bh = b[3] - b[1]
            area = bw * bh
            if c_name == "person":
                # Filter out distant doorway background noise specks (area < 800 or height < 50)
                if tid is not None and area >= 800 and bh >= 50:
                    frame_persons.append((b, tid, "person", area))
            else:
                if c_name not in STATIC_FIXTURE_CLASSES and c_val >= CONF_THRESH and area >= 200:
                    frame_objects.append((b, c_name, c_val))

    # Spatial Object Association with Open-Vocabulary Class Frequency Fusion
    for b_det, c_name, c_val in frame_objects:
        matched_tr = None
        best_dist = float("inf")
        c_det = ((b_det[0] + b_det[2]) / 2.0, (b_det[1] + b_det[3]) / 2.0)

        for tr in active_object_tracklets:
            gap = f_idx - tr["last_frame"]
            if gap <= 40:
                last_b = tr["frame_map"][tr["last_frame"]]
                c_last = ((last_b[0] + last_b[2]) / 2.0, (last_b[1] + last_b[3]) / 2.0)
                center_dist = math.hypot(c_det[0] - c_last[0], c_det[1] - c_last[1])
                iou = compute_box_iou(b_det, last_b)
                if iou >= 0.20 or center_dist <= 60.0 or (gap <= 10 and center_dist <= 90.0):
                    if center_dist < best_dist:
                        best_dist = center_dist
                        matched_tr = tr

        if matched_tr is not None:
            matched_tr["frame_map"][f_idx] = b_det
            matched_tr["confs"].append(c_val)
            matched_tr["class_votes"][c_name] = matched_tr["class_votes"].get(c_name, 0) + 1
            matched_tr["last_frame"] = f_idx
        else:
            active_object_tracklets.append({
                "class_votes": {c_name: 1},
                "frame_map": {f_idx: b_det},
                "confs": [c_val],
                "first_frame": f_idx,
                "last_frame": f_idx
            })

    frame_detections.append({
        "frame_idx": f_idx,
        "persons": frame_persons,
        "objects": frame_objects
    })

cap.release()

# ------------------------------------------------------------------------------
# 4. STABLE PERSON IDENTIFIERS & TRACKLET REFINEMENT
# ------------------------------------------------------------------------------
print("\n--- Stable Person Identifiers & Tracklet Refinement (TTS) ---")
person_tid_stats = {}
for fd in frame_detections:
    for b, tid, c, area in fd["persons"]:
        if tid not in person_tid_stats:
            person_tid_stats[tid] = {"count": 0, "total_area": 0, "frames": {}}
        person_tid_stats[tid]["count"] += 1
        person_tid_stats[tid]["total_area"] += area
        person_tid_stats[tid]["frames"][fd["frame_idx"]] = b

# Temporal Tracklet Stitching (TTS) for Persons across temporary tracking dropouts
raw_person_tracklets = [
    {"id": tid, "frames": stats["frames"], "total_area": stats["total_area"]}
    for tid, stats in person_tid_stats.items()
    if stats["count"] >= 20 and (stats["total_area"] / stats["count"]) >= 1500
]
raw_person_tracklets.sort(key=lambda t: min(t["frames"].keys()))

stitched_persons = []
for tr in raw_person_tracklets:
    tr_start = min(tr["frames"].keys())
    matched_sp = None
    for sp in stitched_persons:
        # Check temporal and spatial continuity across dropouts, concurrent duplicates, or tracker ID flicker
        common_f = set(tr["frames"].keys()) & set(sp["frames"].keys())
        if common_f:
            # Overlapping tracklets: merge if spatial duplicate of the same body (mean IoU >= 0.35)
            ious = [compute_box_iou(tr["frames"][f], sp["frames"][f]) for f in common_f]
            if np.mean(ious) >= 0.35:
                matched_sp = sp
                break
        else:
            # Non-concurrent tracklets (sequential or tracker flicker)
            min_dt = float("inf")
            best_pair = None
            for f_a in tr["frames"]:
                for f_b in sp["frames"]:
                    dt = abs(f_a - f_b)
                    if dt < min_dt:
                        min_dt = dt
                        best_pair = (tr["frames"][f_a], sp["frames"][f_b])
                        if dt == 1:
                            break
                if min_dt == 1:
                    break
            
            # Brief dropout or tracker alternating flicker: dt <= 15 frames (~0.5s)
            if min_dt <= 15 and best_pair is not None:
                b1, b2 = best_pair
                dist = compute_box_edge_distance(b1, b2)
                iou = compute_box_iou(b1, b2)
                if dist <= 35.0 or iou >= 0.30:
                    matched_sp = sp
                    break

    if matched_sp is not None:
        matched_sp["frames"].update(tr["frames"])
        matched_sp["total_area"] += tr["total_area"]
        matched_sp["orig_tids"].append(tr["id"])
    else:
        stitched_persons.append({
            "orig_tids": [tr["id"]],
            "frames": dict(tr["frames"]),
            "total_area": tr["total_area"]
        })

stitched_persons.sort(key=lambda p: p["total_area"], reverse=True)
canonical_persons = stitched_persons[:3]  # keep top foreground actors
person_entities = {}
for i, cp in enumerate(canonical_persons):
    p_id = f"[{i + 1}]"
    person_entities[p_id] = {
        "class": "person",
        "frame_map": cp["frames"],
        "type": "person",
        "numeric_id": i + 1
    }
num_persons = len(person_entities)

# Object tracklet cross-class merging & stationary occlusion gap interpolation
persistent_raw_objects = [tr for tr in active_object_tracklets if len(tr["frame_map"]) >= 25]
persistent_raw_objects.sort(key=lambda tr: min(tr["frame_map"].keys()))
merged_objects = []

for tr in persistent_raw_objects:
    t_start = min(tr["frame_map"].keys())
    b_start = tr["frame_map"][t_start]
    c_start = ((b_start[0] + b_start[2]) / 2.0, (b_start[1] + b_start[3]) / 2.0)
    c_class = max(tr["class_votes"].items(), key=lambda kv: kv[1])[0]
    
    matched = None
    for mo in merged_objects:
        mo_end = max(mo["frame_map"].keys())
        b_end = mo["frame_map"][mo_end]
        c_end = ((b_end[0] + b_end[2]) / 2.0, (b_end[1] + b_end[3]) / 2.0)
        mo_class = max(mo["class_votes"].items(), key=lambda kv: kv[1])[0]
        
        gap = t_start - mo_end
        dist = math.hypot(c_start[0] - c_end[0], c_start[1] - c_end[1])
        
        # Merge if same/compatible class (e.g. handbag <-> backpack) and either:
        # (a) Spatial overlap/stationary match (dist <= 45px)
        # (b) Sequential human transport continuity (gap <= 50 frames and dist <= 160px)
        is_bag_family = (c_class in ["backpack", "handbag"]) and (mo_class in ["backpack", "handbag"])
        is_same_class = (c_class == mo_class) or is_bag_family
        
        if is_same_class:
            if dist <= 45.0 or (-5 <= gap <= 50 and dist <= 160.0):
                matched = mo
                break
                
    if matched is not None:
        matched["frame_map"].update(tr["frame_map"])
        matched["confs"].extend(tr.get("confs", []))
        for cn, count in tr["class_votes"].items():
            matched["class_votes"][cn] = matched["class_votes"].get(cn, 0) + count
    else:
        merged_objects.append(tr)

# Interpolate occlusion gaps and hold stationary position
for mo in merged_objects:
    sorted_fs = sorted(mo["frame_map"].keys())
    if sorted_fs:
        min_f, max_f = sorted_fs[0], sorted_fs[-1]
        for f in range(min_f + 1, max_f):
            if f not in mo["frame_map"]:
                prev_f = max(k for k in sorted_fs if k < f)
                next_f = min(k for k in sorted_fs if k > f)
                alpha = (f - prev_f) / (next_f - prev_f)
                b_prev = np.array(mo["frame_map"][prev_f], dtype=float)
                b_next = np.array(mo["frame_map"][next_f], dtype=float)
                b_interp = (1.0 - alpha) * b_prev + alpha * b_next
                mo["frame_map"][f] = b_interp.astype(int)
        
        # Stationary Forward-Fill:
        # 1. Permanently stationary scene objects (e.g. parked bicycle, total_disp < 35px)
        # 2. Placed objects: objects transported and placed down on a surface/ground (tail_disp < 20px, not near border)
        boxes_arr = np.array(list(mo["frame_map"].values()))
        cxs = (boxes_arr[:, 0] + boxes_arr[:, 2]) / 2.0
        cys = (boxes_arr[:, 1] + boxes_arr[:, 3]) / 2.0
        total_disp = float(math.hypot(np.ptp(cxs), np.ptp(cys)))

        tail_fs = sorted_fs[-min(10, len(sorted_fs)):]
        tail_boxes = np.array([mo["frame_map"][f] for f in tail_fs])
        tail_cxs = (tail_boxes[:, 0] + tail_boxes[:, 2]) / 2.0
        tail_cys = (tail_boxes[:, 1] + tail_boxes[:, 3]) / 2.0
        tail_disp = float(math.hypot(np.ptp(tail_cxs), np.ptp(tail_cys)))

        last_b = mo["frame_map"][max_f]
        is_near_border = (last_b[0] < 20 or last_b[1] < 20 or 
                          last_b[2] > orig_w - 20 or last_b[3] > orig_h - 20)

        if (total_disp < 35.0) or (tail_disp < 20.0 and not is_near_border):
            print(f"  [Stationary Forward-Fill] Object '{c_class}' resting at frame {max_f} (tail_disp={tail_disp:.1f}px, near_border={is_near_border}). Holding position through frame {end_frame}.")
            for f in range(max_f + 1, end_frame + 1):
                mo["frame_map"][f] = last_b

next_obj_id = num_persons + 1
object_entities = {}
for mo in merged_objects:
    o_id = f"[{next_obj_id}]"
    final_class = max(mo["class_votes"].items(), key=lambda kv: kv[1])[0]
    boxes_arr = np.array(list(mo["frame_map"].values()))
    cxs = (boxes_arr[:, 0] + boxes_arr[:, 2]) / 2.0
    cys = (boxes_arr[:, 1] + boxes_arr[:, 3]) / 2.0
    disp = float(math.hypot(np.ptp(cxs), np.ptp(cys)))
    object_entities[o_id] = {
        "class": final_class,
        "frame_map": mo["frame_map"],
        "type": "object",
        "displacement": disp,
        "numeric_id": next_obj_id
    }
    next_obj_id += 1

all_entities = {**person_entities, **object_entities}

print(f"Registered Total Entities: {len(all_entities)} ({len(person_entities)} persons, {len(object_entities)} scene objects)")
for eid, edata in all_entities.items():
    print(f"  - Entity {eid}: class='{edata['class']}', frames_visible={len(edata['frame_map'])}")

# ------------------------------------------------------------------------------
# 5. PASS 2: CLEAN FULL-FRAME SET-OF-MARKS RENDERING & VIDEO EXPORT
# ------------------------------------------------------------------------------
print("\n--- Pass 2: Clean Set-of-Marks Panoramic Video Rendering ---")
# Use 'avc1' (H.264) codec with fallback to 'mp4v' for seamless Windows Media Player / browser compatibility
fourcc = cv2.VideoWriter_fourcc(*"avc1")
out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (orig_w, orig_h))
if not out_writer.isOpened():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (orig_w, orig_h))

rendered_frames = []

for f_idx in range(start_frame, end_frame + 1):
    f_sec, raw_img = raw_clean_frames[f_idx]
    annotated_frame = raw_img.copy()

    # Draw all entities visible in this frame
    for eid, edata in all_entities.items():
        if f_idx in edata["frame_map"]:
            box = edata["frame_map"][f_idx]
            x1, y1, x2, y2 = box
            num_id = edata["numeric_id"]
            color = COLOR_PALETTE[num_id % len(COLOR_PALETTE)]
            c_name = edata["class"]

            # 1. 1.5px Soft Bounding Box Border (anti-aliased visual weight with alpha blending)
            overlay_b = annotated_frame.copy()
            cv2.rectangle(
                overlay_b,
                (max(0, x1 - 1), max(0, y1 - 1)),
                (min(orig_w - 1, x2 + 1), min(orig_h - 1, y2 + 1)),
                color,
                2
            )
            annotated_frame = cv2.addWeighted(overlay_b, 0.45, annotated_frame, 0.55, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

            # 2. Compact label badge with Alpha Blending (70% opacity, 30% background transparency)
            label_text = f"{eid} {c_name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, 1)

            badge_h = text_h + 8
            badge_w = text_w + 10
            if y1 >= badge_h + 2:
                by1 = y1 - badge_h
                by2 = y1
                ty = y1 - 4
            else:
                by1 = y1
                by2 = min(orig_h, y1 + badge_h)
                ty = y1 + text_h + 2

            bx1 = x1
            if bx1 + badge_w > orig_w:
                bx1 = max(0, orig_w - badge_w - 2)
            bx2 = min(orig_w, bx1 + badge_w)

            # Local Alpha Blending (70% tint, 30% background)
            sub = annotated_frame[by1:by2, bx1:bx2]
            overlay_badge = np.full_like(sub, color)
            annotated_frame[by1:by2, bx1:bx2] = cv2.addWeighted(overlay_badge, 0.70, sub, 0.30, 0)

            # Outline for the badge
            cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)

            # 3. Bold White Text (CTRL+B faux-bold double pass)
            cv2.putText(annotated_frame, label_text, (bx1 + 5, ty), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, label_text, (bx1 + 6, ty), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

            # Redundant interior duplicate ID completely eliminated to prevent face/torso occlusion

    out_writer.write(annotated_frame)
    rendered_frames.append((f_idx, f_sec, annotated_frame))

out_writer.release()
print(f"Annotated panoramic clip written to: {output_video_path} ({len(rendered_frames)} frames)")

# ------------------------------------------------------------------------------
# 6. EXTRACT PREVIEW FRAMES FOR RAPID VISUAL VERIFICATION
# ------------------------------------------------------------------------------
if NUM_PREVIEW_FRAMES > 0 and len(rendered_frames) > 0:
    print(f"\n--- Extracting {NUM_PREVIEW_FRAMES} Preview Frames to {preview_dir} ---")
    step = (len(rendered_frames) - 1) / (NUM_PREVIEW_FRAMES - 1) if NUM_PREVIEW_FRAMES > 1 else 0
    for i in range(NUM_PREVIEW_FRAMES):
        idx = int(round(i * step))
        f_idx, f_sec, img = rendered_frames[idx]
        preview_fn = f"frame_{i+1:02d}_{f_sec:.2f}s.jpg"
        preview_path = os.path.join(preview_dir, preview_fn)
        cv2.imwrite(preview_path, img)
        print(f"  [Preview {i+1}/{NUM_PREVIEW_FRAMES}] Saved {preview_fn} (frame {f_idx}, {f_sec:.2f}s)")

print("=" * 80)
print("PANORAMIC FULL-FRAME TRACKING PIPELINE COMPLETE.")
print("=" * 80)
