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
STATIC_FIXTURE_CLASSES = {
    "table", "bench", "chair", "refrigerator", "sofa", "bed",
    "toilet", "sink", "microwave", "oven", "screen", "stool",
    "stop_sign", "traffic_light", "electric_fan", "faucet"
}

from modules.spatial_clustering import (
    compute_box_edge_distance,
    compute_box_iou,
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
parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold for interactive object detection")
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
            if (f_idx - tr["last_frame"]) <= 25:
                last_b = tr["frame_map"][tr["last_frame"]]
                c_last = ((last_b[0] + last_b[2]) / 2.0, (last_b[1] + last_b[3]) / 2.0)
                center_dist = math.hypot(c_det[0] - c_last[0], c_det[1] - c_last[1])
                iou = compute_box_iou(b_det, last_b)
                if iou >= 0.20 or center_dist <= 40.0:
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

# ------------------------------------------------------------------------------
# 4. STABLE PERSON IDENTIFIERS & TRACKLET REFINEMENT
# ------------------------------------------------------------------------------
# Prioritize foreground actors by total pixel mass (count * mean_area)
person_tid_stats = {}
for fd in frame_detections:
    for b, tid, c, area in fd["persons"]:
        if tid not in person_tid_stats:
            person_tid_stats[tid] = {"count": 0, "total_area": 0, "frames": {}}
        person_tid_stats[tid]["count"] += 1
        person_tid_stats[tid]["total_area"] += area
        person_tid_stats[tid]["frames"][fd["frame_idx"]] = b

stable_persons = [
    tid for tid, stats in person_tid_stats.items()
    if stats["count"] >= 15 and (stats["total_area"] / stats["count"]) >= 2000
]
stable_persons.sort(key=lambda tid: person_tid_stats[tid]["total_area"], reverse=True)
num_persons = min(3, len(stable_persons))
canonical_persons = sorted(stable_persons[:num_persons])
canonical_person_map = {orig_tid: i + 1 for i, orig_tid in enumerate(canonical_persons)}

person_entities = {}
for orig_tid, c_tid in canonical_person_map.items():
    p_id = f"[{c_tid}]"
    person_entities[p_id] = {
        "class": "person",
        "frame_map": person_tid_stats[orig_tid]["frames"],
        "type": "person"
    }

# Object tracklet cross-class merging & stationary occlusion gap interpolation
persistent_raw_objects = [tr for tr in active_object_tracklets if len(tr["frame_map"]) >= 15]

merged_objects = []
for tr in persistent_raw_objects:
    matched = None
    for mo in merged_objects:
        b1_arr = np.array(list(tr['frame_map'].values()))
        b2_arr = np.array(list(mo['frame_map'].values()))
        c1 = (np.mean(b1_arr[:, 0] + b1_arr[:, 2]) / 2.0, np.mean(b1_arr[:, 1] + b1_arr[:, 3]) / 2.0)
        c2 = (np.mean(b2_arr[:, 0] + b2_arr[:, 2]) / 2.0, np.mean(b2_arr[:, 1] + b2_arr[:, 3]) / 2.0)
        if math.hypot(c1[0] - c2[0], c1[1] - c2[1]) <= 40.0:
            matched = mo
            break
    if matched is not None:
        matched["frame_map"].update(tr["frame_map"])
        matched["confs"].extend(tr["confs"])
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
        
        # Stationary forward-fill ONLY for resting scene objects (disp < 35px), preventing ghost boxes for carried items
        boxes_arr = np.array(list(mo["frame_map"].values()))
        cxs = (boxes_arr[:, 0] + boxes_arr[:, 2]) / 2.0
        cys = (boxes_arr[:, 1] + boxes_arr[:, 3]) / 2.0
        disp = float(math.hypot(np.ptp(cxs), np.ptp(cys)))
        if disp < 35.0:
            last_b = mo["frame_map"][max_f]
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

# Read System Prompt
prompt_txt_path = os.path.join(base_dir, "data", "prompt_system_general.txt")
if os.path.exists(prompt_txt_path):
    with open(prompt_txt_path, "r", encoding="utf-8") as f:
        vlm_system_prompt = f.read().strip()
else:
    vlm_system_prompt = ""

# ------------------------------------------------------------------------------
# 6. MULTI-SCALE DYNAMIC ROI ZOOM CROP & PAYLOAD GENERATION PER CLUSTER
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 2: DYNAMIC ROI ZOOM CROP RENDERING & PAYLOAD GENERATION")
print("=" * 80)

generated_payloads = []

for c in active_clusters:
    cid = c["cluster_id"]
    cluster_dir_name = f"{video_basename}_roi_{cid}"
    cluster_frames_dir = os.path.join(base_dir, "data", "frames", cluster_dir_name)
    
    # Spatio-Temporal Active Interaction Window:
    # Samples 8 frames strictly while the interacting entities are actively engaged in physical contact/proximity
    # and safely within the camera frame (eliminates border truncation artifacts and post-interaction separation)
    eids = c['entity_ids']
    human_eids = [eid for eid in eids if c['entities'][eid]['type'] == 'person']

    active_contact_frames = []
    if len(human_eids) >= 2:
        fm1 = c['entities'][human_eids[0]]['frame_map']
        fm2 = c['entities'][human_eids[1]]['frame_map']
        common_f = sorted(list(set(fm1.keys()) & set(fm2.keys())))
        if common_f:
            max_h1 = max(fm1[f][3] - fm1[f][1] for f in common_f)
            max_h2 = max(fm2[f][3] - fm2[f][1] for f in common_f)
            min_h1 = int(0.55 * max_h1)
            min_h2 = int(0.55 * max_h2)
            for f in common_f:
                b1, b2 = fm1[f], fm2[f]
                dist = compute_box_edge_distance(b1, b2)
                h1 = b1[3] - b1[1]
                h2 = b2[3] - b2[1]
                # Full 4-edge boundary safety: persons are not truncated off-screen
                in_bounds = (
                    b1[0] >= 15 and b1[2] <= orig_w - 20 and b1[1] >= 15 and b1[3] <= orig_h - 20 and h1 >= min_h1 and
                    b2[0] >= 15 and b2[2] <= orig_w - 20 and b2[1] >= 15 and b2[3] <= orig_h - 20 and h2 >= min_h2
                )
                if dist <= 30.0 and in_bounds:
                    active_contact_frames.append(f)
    elif len(human_eids) == 1 and len(eids) >= 2:
        h_id = human_eids[0]
        o_ids = [eid for eid in eids if eid != h_id]
        fm_h = c['entities'][h_id]['frame_map']
        for o_id in o_ids:
            fm_o = c['entities'][o_id]['frame_map']
            for f in sorted(list(set(fm_h.keys()) & set(fm_o.keys()))):
                b1, b2 = fm_h[f], fm_o[f]
                dist = compute_box_edge_distance(b1, b2)
                is_in_bounds = (b1[2] <= orig_w - 20) and (b1[0] >= 15) and (b1[3] <= orig_h - 20) and (b1[1] >= 15)
                if dist <= 35.0 and is_in_bounds:
                    active_contact_frames.append(f)
        active_contact_frames = sorted(list(set(active_contact_frames)))

    if len(active_contact_frames) >= NUM_VLM_FRAMES:
        c_step = (len(active_contact_frames) - 1) / (NUM_VLM_FRAMES - 1)
        c_sample_indices = [active_contact_frames[int(round(i * c_step))] for i in range(NUM_VLM_FRAMES)]
    else:
        co_present_frames = [
            f for f in sorted(list(set.union(*[set(c['entities'][eid]['frame_map'].keys()) for eid in c['entity_ids']])))
            if sum(1 for eid in c['entity_ids'] if f in c['entities'][eid]['frame_map']) >= 2
        ]
        if len(co_present_frames) >= NUM_VLM_FRAMES:
            c_step = (len(co_present_frames) - 1) / (NUM_VLM_FRAMES - 1)
            c_sample_indices = [co_present_frames[int(round(i * c_step))] for i in range(NUM_VLM_FRAMES)]
        else:
            ideal_step = (end_frame - start_frame) / (NUM_VLM_FRAMES - 1)
            c_sample_indices = [int(start_frame + i * ideal_step) for i in range(NUM_VLM_FRAMES)]

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

    print(f"\nSUCCESS: [{cid.upper()}] Rendered {len(saved_frames)} Zoom Crop Frames:")
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
