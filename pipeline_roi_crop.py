"""
Unified Forward Pipeline with Task 2: Spatial Clustering & Dynamic ROI Zoom Crop
=================================================================================
VidVRD Task 2 Implementation:
1. Unified YOLOE-26m Open-Vocabulary Object & Person Tracking (ByteTrack).
2. Spatio-Temporal Interaction Clustering (DBSCAN / Proximity-based Connected Components).
3. Dynamic Stabilized Union Bounding Box Envelope with Adaptive Context Padding (~20%).
4. Multi-Scale High-Resolution Zoom Crop Generation (eliminates background distractor clutter).
5. Cluster-Specific VLM Payload Generation with Zero Cheating / Zero Hardcoding.

Author: VinFast Computer Vision Center AI Trainee Program
Mentors: Luong Manh Tu & Tran Minh Thanh
"""

import os
import sys
import cv2
import json
import math
import argparse
import numpy as np
import torch
from ultralytics import YOLO

# Import Task 2 Spatial Clustering Module
from modules.spatial_clustering import (
    cluster_entities_spatially,
    compute_cluster_union_boxes,
    render_cluster_zoom_frames
)

# ------------------------------------------------------------------------------
# CLI ARGUMENTS
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="VidVRD Task 2: Spatial Clustering & Dynamic ROI Zoom Crop Pipeline")
parser.add_argument("--video", type=str, default="data/videos/video1.mp4", help="Path to input surveillance video")
parser.add_argument("--start_sec", type=float, default=44.0, help="Start time in seconds for golden segment")
parser.add_argument("--end_sec", type=float, default=52.0, help="End time in seconds for golden segment")
parser.add_argument("--num_frames", type=int, default=8, help="Number of clean VLM frames to sample")
parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold for interactive object detection")
parser.add_argument("--iou", type=float, default=0.35, help="IoU threshold for ByteTrack person tracking")
parser.add_argument("--stride", type=int, default=1, help="Frame step stride for object detection")
parser.add_argument("--device", type=str, default="auto", help="Compute device: 'auto', 'cuda', or 'cpu'")
parser.add_argument("--proximity_thresh", type=float, default=None, help="Spatial proximity threshold in pixels (default: dynamic ~15% diagonal)")
parser.add_argument("--padding_ratio", type=float, default=0.20, help="Adaptive context padding ratio for ROI crop (default: 0.20)")
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
vlm_frames_dir = os.path.join(base_dir, "data", "frames", f"{video_basename}_yoloe")
payloads_dir = os.path.join(base_dir, "data", "payloads")
os.makedirs(payloads_dir, exist_ok=True)
os.makedirs(vlm_frames_dir, exist_ok=True)

START_SEC = cli_args.start_sec
END_SEC = cli_args.end_sec
NUM_VLM_FRAMES = cli_args.num_frames
CONF_THRESH = cli_args.conf
IOU_THRESH = cli_args.iou
FRAME_STRIDE = cli_args.stride

print("=" * 80)
print(f"VIDVRD TASK 2 PIPELINE: SPATIAL CLUSTERING & DYNAMIC ROI ZOOM CROP")
print(f"Target Video:        {video_path}")
print(f"Temporal Window:     {START_SEC}s -> {END_SEC}s (Duration: {END_SEC - START_SEC:.1f}s)")
print(f"Target VLM Frames:   {NUM_VLM_FRAMES} frames")
print(f"Compute Device:      {DEVICE.upper()}")
print(f"Adaptive Padding:    {cli_args.padding_ratio * 100:.0f}%")
print("=" * 80)

# Load Vocabularies
with open(os.path.join(base_dir, "configs", "s_objects.json"), "r", encoding="utf-8") as f:
    allowed_objects_60 = json.load(f)

with open(os.path.join(base_dir, "configs", "relations.json"), "r", encoding="utf-8") as f:
    relations_list = json.load(f)

# ------------------------------------------------------------------------------
# 2. MODEL INITIALIZATION
# ------------------------------------------------------------------------------
yoloe_weights = os.path.join(base_dir, "yoloe-26m-seg.pt")
if not os.path.exists(yoloe_weights):
    yoloe_weights = os.path.join(base_dir, "weights", "yoloe-26m-seg.pt")
model_weights = yoloe_weights
model = YOLO(model_weights)
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

    # Single forward pass for both human tracking and object detection (avoids Ultralytics classes mutation bug)
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
            if c_name == "person":
                if tid is not None:
                    frame_persons.append((b, tid, "person"))
            else:
                if c_val >= CONF_THRESH:
                    frame_objects.append((b, c_name, c_val))

    # Object Association
    for b_det, c_name, c_val in frame_objects:
        matched_tr = None
        best_iou = 0.20
        for tr in active_object_tracklets:
            if tr["class"] == c_name and (f_idx - tr["last_frame"]) <= 12:
                last_b = tr["frame_map"][tr["last_frame"]]
                # compute IoU
                x1 = max(b_det[0], last_b[0])
                y1 = max(b_det[1], last_b[1])
                x2 = min(b_det[2], last_b[2])
                y2 = min(b_det[3], last_b[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                a1 = max(0, b_det[2] - b_det[0]) * max(0, b_det[3] - b_det[1])
                a2 = max(0, last_b[2] - last_b[0]) * max(0, last_b[3] - last_b[1])
                u = a1 + a2 - inter
                iou = (inter / u) if u > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    matched_tr = tr
        if matched_tr is not None:
            matched_tr["frame_map"][f_idx] = b_det
            matched_tr["confs"].append(c_val)
            matched_tr["last_frame"] = f_idx
        else:
            active_object_tracklets.append({
                "class": c_name,
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

# ------------------------------------------------------------------------------
# 4. STABLE PERSON IDENTIFIERS & TRACKLET REFINEMENT
# ------------------------------------------------------------------------------
all_tids = {}
for fd in frame_detections:
    for b, tid, c in fd["persons"]:
        all_tids[tid] = all_tids.get(tid, 0) + 1

stable_person_ids = [tid for tid, count in all_tids.items() if count >= 10]
stable_person_ids.sort(key=lambda x: all_tids[x], reverse=True)
num_persons = min(3, len(stable_person_ids))
canonical_person_map = {orig_tid: i + 1 for i, orig_tid in enumerate(sorted(stable_person_ids[:num_persons]))}

person_entities = {}
for p_idx, (orig_tid, c_tid) in enumerate(canonical_person_map.items()):
    p_id = f"[{c_tid}]"
    p_fmap = {}
    for fd in frame_detections:
        for b, tid, c in fd["persons"]:
            if tid == orig_tid:
                p_fmap[fd["frame_idx"]] = b
    person_entities[p_id] = {
        "class": "person",
        "frame_map": p_fmap,
        "type": "person"
    }

# Object tracklet cross-class merging
persistent_objects = [tr for tr in active_object_tracklets if len(tr["frame_map"]) >= 25]
merged_objects = []
for tr in persistent_objects:
    matched = None
    for mo in merged_objects:
        common_f = set(tr["frame_map"].keys()) & set(mo["frame_map"].keys())
        if len(common_f) >= 8:
            ious = []
            for f in common_f:
                b1, b2 = tr["frame_map"][f], mo["frame_map"][f]
                x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
                ious.append(inter/u if u > 0 else 0)
            if np.median(ious) > 0.35:
                matched = mo
                break
    if matched is not None:
        matched["frame_map"].update(tr["frame_map"])
        matched["confs"].extend(tr["confs"])
    else:
        merged_objects.append(tr)

next_obj_id = num_persons + 1
object_entities = {}
for mo in merged_objects:
    o_id = f"[{next_obj_id}]"
    boxes_arr = np.array(list(mo["frame_map"].values()))
    cxs = (boxes_arr[:, 0] + boxes_arr[:, 2]) / 2.0
    cys = (boxes_arr[:, 1] + boxes_arr[:, 3]) / 2.0
    disp = float(np.hypot(np.ptp(cxs), np.ptp(cys)))
    object_entities[o_id] = {
        "class": mo["class"],
        "frame_map": mo["frame_map"],
        "type": "object",
        "displacement": disp
    }
    next_obj_id += 1

all_entities = {**person_entities, **object_entities}

print(f"Registered Total Entities: {len(all_entities)} ({len(person_entities)} persons, {len(object_entities)} scene objects)")
for eid, edata in all_entities.items():
    print(f"  - Entity {eid}: class='{edata['class']}', frames_visible={len(edata['frame_map'])}")

# ------------------------------------------------------------------------------
# 5. TASK 2: SPATIAL CLUSTERING (DBSCAN / PROXIMITY CONNECTED COMPONENTS)
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 2: SPATIO-TEMPORAL INTERACTION CLUSTERING")
print("=" * 80)

active_clusters, singletons = cluster_entities_spatially(
    entities=all_entities,
    image_shape=(orig_h, orig_w),
    proximity_thresh_px=cli_args.proximity_thresh
)

print(f"Spatial Clustering Results:")
print(f"  - Total Active Interactive Clusters (|C| >= 2): {len(active_clusters)}")
print(f"  - Total Isolated Singletons (|C| == 1):         {len(singletons)} (Excluded from interaction search)")

if singletons:
    print(f"    * Isolated Non-Interacting Entities: {singletons}")

for c in active_clusters:
    cid = c["cluster_id"]
    e_list = [f"{eid} ({c['entities'][eid]['class']})" for eid in c["entity_ids"]]
    print(f"  - {cid.upper()}: Entities = {e_list}, Min Pairwise Edge Distance = {c['min_internal_distance_px']:.1f}px")

# ------------------------------------------------------------------------------
# 6. SAMPLING 8 TEMPORAL FRAMES
# ------------------------------------------------------------------------------
ideal_frame_indices = [int(start_frame + i * (end_frame - start_frame) / (NUM_VLM_FRAMES - 1)) for i in range(NUM_VLM_FRAMES)]
sample_frames_clean = {f_idx: raw_clean_frames[f_idx] for f_idx in ideal_frame_indices if f_idx in raw_clean_frames}

# Read System Prompt
prompt_txt_path = os.path.join(base_dir, "data", "prompt_system_general.txt")
if os.path.exists(prompt_txt_path):
    with open(prompt_txt_path, "r", encoding="utf-8") as f:
        vlm_system_prompt = f.read().strip()
else:
    vlm_system_prompt = ""

# ------------------------------------------------------------------------------
# 7. MULTI-SCALE DYNAMIC ROI ZOOM CROP & PAYLOAD GENERATION PER CLUSTER
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 2: DYNAMIC ROI ZOOM CROP RENDERING & PAYLOAD GENERATION")
print("=" * 80)

generated_payloads = []

for c in active_clusters:
    cid = c["cluster_id"]
    cluster_dir_name = f"{video_basename}_roi_{cid}"
    cluster_frames_dir = os.path.join(base_dir, "data", "frames", cluster_dir_name)
    
    # Dynamic Cluster Lifespan: sample 8 frames strictly during the active interaction window of this cluster
    cluster_visible_frames = sorted(list(set.union(*[set(c['entities'][eid]['frame_map'].keys()) for eid in c['entity_ids']])))
    if len(cluster_visible_frames) >= NUM_VLM_FRAMES:
        c_step = (len(cluster_visible_frames) - 1) / (NUM_VLM_FRAMES - 1)
        c_sample_indices = [cluster_visible_frames[int(round(i * c_step))] for i in range(NUM_VLM_FRAMES)]
    else:
        c_sample_indices = ideal_frame_indices

    c_sample_frames_clean = {f_idx: raw_clean_frames[f_idx] for f_idx in c_sample_indices if f_idx in raw_clean_frames}

    # Compute stabilized union crop box
    crop_boxes = compute_cluster_union_boxes(
        cluster_entity_ids=c["entity_ids"],
        entities=all_entities,
        sample_frame_indices=c_sample_indices,
        image_shape=(orig_h, orig_w),
        padding_ratio=cli_args.padding_ratio,
        stabilize_temporal_envelope=True
    )

    # Render Zoom Crop Frames
    saved_frames = render_cluster_zoom_frames(
        cluster=c,
        clean_frames_dict=c_sample_frames_clean,
        crop_boxes=crop_boxes,
        output_dir=cluster_frames_dir,
        color_palette=COLOR_PALETTE
    )

    frame_filenames = [fn for fn, _, _ in saved_frames]
    mean_zoom = float(np.mean([zf for _, _, zf in saved_frames]))

    print(f"\n✅ [{cid.upper()}] Rendered {len(saved_frames)} Zoom Crop Frames:")
    print(f"   - Storage: {cluster_frames_dir}")
    print(f"   - Mean Resolution Magnification (Zoom Factor): {mean_zoom:.2f}x")
    for fn, _, zf in saved_frames[:3]:
        print(f"     * {fn} (Zoom: {zf:.2f}x)")
    if len(saved_frames) > 3:
        print(f"     * ... and {len(saved_frames) - 3} more frames")

    # Build Cluster-Specific Prompt
    cluster_entities_str = ", ".join([f"{eid} ({c['entities'][eid]['class']})" for eid in c["entity_ids"]])
    cluster_user_prompt = (
        f"Analyze all provided sequential zoom-crop frames of this localized interaction zone ({cid}).\n"
        f"Detected entities with visual marks in this interaction cluster: {cluster_entities_str}.\n\n"
        "Examine active interactions between the marked entities across time.\n\n"
        "PREDEFINED RELATION TAXONOMY (CLOSED VOCABULARY):\n"
        f"Every predicate in the 'relation' field MUST be an exact string match selected strictly from the 26 allowed categories: {relations_list}. All out-of-vocabulary verbs are strictly prohibited.\n"
        "- Strictly evaluate interactions ONLY between marked entities [ID]. Completely ignore unmarked objects or background clutter; NEVER substitute an unmarked item with a marked person.\n"
        "- Clean ID Formatting: In the 'subject' and 'object' fields, output strictly the clean mark ID string (e.g. '[1]', '[2]') without appending class names or descriptive words.\n"
        "- Triplet Uniqueness & Predicate Exclusivity: Report each unique relation between a subject and an object AT MOST ONCE for the entire clip. Between the same subject and object, interaction predicates are strictly mutually exclusive: output ONLY the single most comprehensive predicate (e.g., casual contact during walking, greeting, or parting is categorized strictly as 'touch', never 'push'). Do NOT output duplicate or conflicting triplets for different frames.\n"
        "- Temporal Action Continuity: Sequential phases of an interaction across time (such as approaching, extending an arm, making contact, and parting) constitute ONE unified interaction event. Do NOT fragment preparatory reaching motions into a separate 'push' relation.\n"
        "- Interpersonal Actions: Categorize casual physical contact between persons (such as placing a hand on a shoulder, touching an arm or body, or tapping) strictly as 'touch'. Reserve 'push' and 'pull' strictly for visible forceful shoving where an entity is visibly propelled, knocked off balance, or dragged.\n"
        "- Predicates 'carry' and 'hold' apply strictly between a Person (subject) and a moveable Object. A person cannot 'carry' another person unless physically lifting them off the ground.\n"
        "- PHYSICAL HAND-GRASP REQUIREMENT: 'carry' and 'hold' strictly require direct physical hand contact (grasping, gripping, or lifting the object). If an object is resting on the floor or surface and a person's hands are not physically grasping it (e.g., hands are raised, swinging, or interacting with another person), the person is NOT carrying or holding it. Merely walking past, stepping near, or standing over an object on the ground is NOT an interaction; omit that pair.\n"
        "- Predicates like 'get_on', 'get_off', 'ride', 'drive' apply ONLY to vehicles or animals.\n"
        "- Categorize a person holding and transporting an object with their hands while moving as 'carry', and holding statically as 'hold'.\n"
        "- If an object remains stationary in the same location across all frames without movement, omit that pair.\n"
        "- If an active interaction occurs in any frames, report that relation even if it ends later.\n"
        "- In the 'reason' field, describe strictly the visible physical contact between this subject and this object without referencing other entities.\n\n"
        'Respond strictly with the JSON object: {"triplets": [{"subject": "[ID]", "relation": "<verb>", "object": "[ID]", "reason": "..."}]}.'
    )

    cluster_payload = {
        "task": "Video Visual Relation Detection (VidVRD) - Task 2 Dynamic ROI Zoom Crop",
        "scenario": f"Localized Interaction Zone Analysis ({cid})",
        "pipeline_variant": "yoloe_spatial_clustering_dynamic_roi_zoom",
        "cluster_info": {
            "cluster_id": cid,
            "entities": [{"mark_id": eid, "class_label": c["entities"][eid]["class"]} for eid in c["entity_ids"]],
            "zoom_magnification_factor": f"{mean_zoom:.2f}x",
            "padding_ratio": cli_args.padding_ratio,
            "min_internal_distance_px": round(c["min_internal_distance_px"], 1)
        },
        "allowed_objects_vocabulary_60": allowed_objects_60,
        "allowed_relations_vocabulary_26": relations_list,
        "visual_prompt_frames_sequence": frame_filenames,
        "vlm_system_prompt": vlm_system_prompt,
        "vlm_user_prompt": cluster_user_prompt
    }

    payload_filename = f"{video_basename}_roi_{cid}_payload.json"
    payload_file_path = os.path.join(payloads_dir, payload_filename)
    with open(payload_file_path, "w", encoding="utf-8") as pf:
        json.dump(cluster_payload, pf, indent=2, ensure_ascii=False)
    
    print(f"   - Generated Cluster Payload: {payload_file_path}")
    generated_payloads.append(payload_file_path)

print("\n" + "=" * 80)
print(f"TASK 2 PIPELINE COMPLETED SUCCESSFULLY!")
print(f"Generated {len(generated_payloads)} Cluster Payloads in: {payloads_dir}")
for p in generated_payloads:
    print(f"  * {os.path.basename(p)}")
print("=" * 80)
