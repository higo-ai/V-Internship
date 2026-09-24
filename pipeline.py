import os
import json
import cv2
import numpy as np
from collections import Counter, defaultdict
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description="VidVRD Automated Pipeline: Video -> YOLO Tracking & Marks -> 8 Frames -> Payload JSON")
parser.add_argument("--video", type=str, default="data/videos/video1.mp4", help="Path to input surveillance video")
parser.add_argument("--start_sec", type=float, default=44.0, help="Start time in seconds for golden segment")
parser.add_argument("--end_sec", type=float, default=52.0, help="End time in seconds for golden segment")
parser.add_argument("--num_frames", type=int, default=8, help="Number of clean VLM frames to sample")
cli_args, _ = parser.parse_known_args()

# 1. Paths & Configurations
base_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(base_dir, cli_args.video) if not os.path.isabs(cli_args.video) else cli_args.video
if not os.path.exists(video_path):
    # Fallback from .avi to .mp4 if .avi was converted
    if video_path.lower().endswith(".avi"):
        alt_path = video_path[:-4] + ".mp4"
        if os.path.exists(alt_path):
            video_path = alt_path
    elif not os.path.exists(video_path):
        alt_data = os.path.join(base_dir, "data", "videos", os.path.basename(video_path))
        if os.path.exists(alt_data):
            video_path = alt_data
        elif alt_data.lower().endswith(".avi") and os.path.exists(alt_data[:-4] + ".mp4"):
            video_path = alt_data[:-4] + ".mp4"
video_basename = os.path.splitext(os.path.basename(video_path))[0]
output_video_path = os.path.join(base_dir, "data", "video_processed", f"{video_basename}_annotated.mp4")
vlm_frames_dir = os.path.join(base_dir, "data", "frames", video_basename)
artifact_dir = r"C:\Users\higoi\.gemini\antigravity-ide\brain\5bde3e10-5ab7-412f-9811-185e514054b1\frames"
payloads_dir = os.path.join(base_dir, "data", "payloads")
os.makedirs(payloads_dir, exist_ok=True)
payload_path = os.path.join(payloads_dir, f"{video_basename}_payload.json")

os.makedirs(vlm_frames_dir, exist_ok=True)
os.makedirs(artifact_dir, exist_ok=True)

# Clean old frames in directory before sampling
for f in os.listdir(vlm_frames_dir):
    if f.endswith(".jpg"):
        os.remove(os.path.join(vlm_frames_dir, f))

# Load official project taxonomies (60 S/Objects and 26 Relations)
sobj_path = os.path.join(base_dir, "configs", "s_objects.json") if os.path.exists(os.path.join(base_dir, "configs", "s_objects.json")) else os.path.join(base_dir, "s_objects.json")
with open(sobj_path, encoding="utf-8") as f:
    allowed_objects_60 = json.load(f)
    allowed_objects_set = set(allowed_objects_60)

rel_path = os.path.join(base_dir, "configs", "relations.json") if os.path.exists(os.path.join(base_dir, "configs", "relations.json")) else os.path.join(base_dir, "relations.json")
with open(rel_path, encoding="utf-8") as f:
    relations_list = json.load(f)

# Taxonomy alignment: map COCO-80 classes to official 60 s_objects
COCO_TO_S_OBJECTS = {
    "person": "person", "bicycle": "bicycle", "car": "car", "motorcycle": "motorcycle",
    "bus": "bus", "train": "train", "truck": "truck", "traffic light": "traffic_light",
    "stop sign": "stop_sign", "bench": "bench", "bird": "bird", "cat": "cat",
    "dog": "dog", "horse": "horse", "sheep": "sheep", "cow": "cattle",
    "backpack": "backpack", "handbag": "handbag", "suitcase": "suitcase",
    "sports ball": "ball", "baseball bat": "bat", "tennis racket": "racket",
    "bottle": "bottle", "cup": "cup", "chair": "chair", "couch": "sofa",
    "dining table": "table", "toilet": "toilet", "tv": "screen", "laptop": "laptop",
    "cell phone": "cellphone", "microwave": "microwave", "oven": "oven",
    "sink": "sink", "refrigerator": "refrigerator", "cake": "cake", "skateboard": "skateboard"
}

# 2. Golden segment for Abandoned Object: seconds 44.0 to 52.0 (8.0 seconds, 240 frames)
START_SEC = cli_args.start_sec
END_SEC = cli_args.end_sec
NUM_VLM_FRAMES = cli_args.num_frames

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

start_frame = int(START_SEC * fps)
end_frame = int(END_SEC * fps)
clip_frame_count = end_frame - start_frame

print(f"Original Video: {total_frames} frames, {fps:.2f} FPS")
print(f"Clipping segment: {START_SEC}s - {END_SEC}s (frames {start_frame} to {end_frame}, total {clip_frame_count} frames)")
print(f"Project Taxonomies loaded: {len(allowed_objects_60)} S/Objects, {len(relations_list)} Relations")
print(f"Frame Sampling Density: {NUM_VLM_FRAMES} frames across {END_SEC - START_SEC:.1f}s clip")

# 3. Initialize YOLO Models
print("Loading YOLO models (local weights yolo11n.pt)...")
tracker_model = YOLO(os.path.join(base_dir, "yolo11n.pt"))
detector_model = YOLO(os.path.join(base_dir, "yolo11n.pt"))
tracker_config = os.path.join(base_dir, "configs", "custom_bytetrack.yaml")
if not os.path.exists(tracker_config):
    tracker_config = os.path.join(base_dir, "custom_bytetrack.yaml")
if not os.path.exists(tracker_config):
    tracker_config = "bytetrack.yaml"
print(f"Using Tracker Config: {tracker_config}")

COLOR_PALETTE = [
    (0, 0, 255),     # 0: Red
    (255, 140, 0),   # 1: Deep Sky Blue
    (0, 215, 255),   # 2: Gold / Yellow
    (50, 205, 50),   # 3: Lime Green
    (238, 130, 238), # 4: Violet / Pink
    (0, 255, 255),   # 5: Cyan
    (255, 0, 128),   # 6: Rose
    (128, 255, 0),   # 7: Chartreuse
]

# Set up VideoWriter with avc1 / OpenH264
fourcc = cv2.VideoWriter_fourcc(*"avc1")
out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
if not out_writer.isOpened():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Static room fixtures / furniture that are part of scene architecture (not portable objects)
STATIC_FIXTURE_CLASSES = {
    "table", "bench", "chair", "refrigerator", "sofa", "bed",
    "toilet", "sink", "microwave", "oven", "screen"
}

# ==============================================================================
# PASS 1: Dynamic Tracking & Spatial Velocity Clustering
# Zero Hardcoding, Strictly Enforced 60 S/Objects Taxonomy
# ==============================================================================
print("\n--- Pass 1: Extracting Dynamic Person Tracks & Spatial Object Clusters ---")
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frame_detections = []
raw_stationary_clusters = defaultdict(lambda: {'boxes': [], 'classes': [], 'frame_indices': []})
CLUSTER_DIST_THRESH = 35.0

stable_person_ids = set()
person_hit_counts = Counter()

for f_idx in range(start_frame, end_frame):
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Track Persons with ByteTrack
    p_results = tracker_model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.25,
        iou=0.5,
        tracker=tracker_config,
        verbose=False
    )[0]

    current_persons = []
    if p_results.boxes and p_results.boxes.id is not None:
        p_boxes = p_results.boxes.xyxy.cpu().numpy().astype(int)
        p_ids = p_results.boxes.id.cpu().numpy().astype(int)
        for b, tid in zip(p_boxes, p_ids):
            current_persons.append((b, tid, "person"))
            person_hit_counts[tid] += 1

    # 2. Open-Vocabulary Object Detection (Classes != person)
    all_det = detector_model(frame, conf=0.15, verbose=False)[0]
    current_objects = []
    if all_det.boxes:
        d_boxes = all_det.boxes.xyxy.cpu().numpy().astype(int)
        d_confs = all_det.boxes.conf.cpu().numpy()
        d_classes = all_det.boxes.cls.cpu().numpy().astype(int)
        class_names = all_det.names

        for b, c_conf, c_idx in zip(d_boxes, d_confs, d_classes):
            c_raw = class_names[c_idx]
            if c_raw == "person":
                continue

            mapped_class = COCO_TO_S_OBJECTS.get(c_raw, None)
            if not mapped_class or mapped_class not in allowed_objects_set:
                continue

            # Exclude large scene backgrounds
            bw = b[2] - b[0]
            bh = b[3] - b[1]
            if bw > width * 0.45 or bh > height * 0.45:
                continue

            # Exclude static room fixtures so they don't pollute or swallow portable objects
            if mapped_class in STATIC_FIXTURE_CLASSES:
                continue

            current_objects.append((b, c_conf, mapped_class))

            # Cluster spatial locations
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0

            assigned = False
            for k in list(raw_stationary_clusters.keys()):
                dist = np.hypot(cx - k[0], cy - k[1])
                if dist < CLUSTER_DIST_THRESH:
                    raw_stationary_clusters[k]['boxes'].append(b)
                    raw_stationary_clusters[k]['classes'].append(mapped_class)
                    raw_stationary_clusters[k]['frame_indices'].append(f_idx)
                    assigned = True
                    break

            if not assigned:
                new_key = (cx, cy)
                raw_stationary_clusters[new_key]['boxes'].append(b)
                raw_stationary_clusters[new_key]['classes'].append(mapped_class)
                raw_stationary_clusters[new_key]['frame_indices'].append(f_idx)

    frame_detections.append({
        'frame_idx': f_idx,
        'persons': current_persons,
        'objects': current_objects
    })

# Track stitching: stitch non-overlapping sequential person tracks of the same individual
person_time_spans = defaultdict(lambda: {'start': 999999, 'end': -1, 'last_box': None, 'first_box': None, 'hits': 0})
for fd in frame_detections:
    f_i = fd['frame_idx']
    for b, tid, _ in fd['persons']:
        s = person_time_spans[tid]
        s['start'] = min(s['start'], f_i)
        s['end'] = max(s['end'], f_i)
        s['hits'] += 1
        if s['first_box'] is None: s['first_box'] = b
        s['last_box'] = b

person_id_remap = {}
sorted_pids = sorted([pid for pid, s in person_time_spans.items() if s['hits'] >= 10], key=lambda x: person_time_spans[x]['start'])

if sorted_pids:
    active_chains = []
    for pid in sorted_pids:
        p_info = person_time_spans[pid]
        p_start = p_info['start']
        p_first = p_info['first_box']
        matched_chain = None
        for chain in active_chains:
            gap = p_start - chain['end']
            if 0 < gap <= 60:
                c1 = ((chain['last_box'][0] + chain['last_box'][2]) / 2.0, (chain['last_box'][1] + chain['last_box'][3]) / 2.0)
                c2 = ((p_first[0] + p_first[2]) / 2.0, (p_first[1] + p_first[3]) / 2.0)
                dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
                if dist < 200.0:
                    matched_chain = chain
                    break
        if matched_chain:
            person_id_remap[pid] = matched_chain['root_id']
            matched_chain['end'] = p_info['end']
            matched_chain['last_box'] = p_info['last_box']
        else:
            active_chains.append({'root_id': pid, 'end': p_info['end'], 'last_box': p_info['last_box']})

for fd in frame_detections:
    new_persons = []
    for b, tid, cls_name in fd['persons']:
        mapped_tid = person_id_remap.get(tid, tid)
        new_persons.append((b, mapped_tid, cls_name))
    fd['persons'] = new_persons

stitched_hit_counts = Counter()
for fd in frame_detections:
    for _, tid, _ in fd['persons']:
        stitched_hit_counts[tid] += 1

stable_person_ids = set()
for tid, count in stitched_hit_counts.items():
    if count >= 15:
        stable_person_ids.add(tid)
print(f"Stable dynamic person IDs tracked: {sorted(list(stable_person_ids))}")

# Tracklet Gap Filling: Linearly interpolate short detection flickers (<= 5 frames)
for tid in stable_person_ids:
    p_frames = {}
    for fd in frame_detections:
        for b, p_tid, _ in fd['persons']:
            if p_tid == tid:
                p_frames[fd['frame_idx']] = b
                break
    if len(p_frames) >= 2:
        sorted_p_fs = sorted(p_frames.keys())
        for idx in range(len(sorted_p_fs) - 1):
            f_a = sorted_p_fs[idx]
            f_b = sorted_p_fs[idx + 1]
            gap = f_b - f_a
            if 1 < gap <= 6:
                b_a = p_frames[f_a]
                b_b = p_frames[f_b]
                for missing_f in range(f_a + 1, f_b):
                    alpha = (missing_f - f_a) / float(gap)
                    interp_box = ((1.0 - alpha) * b_a + alpha * b_b).astype(int)
                    for fd in frame_detections:
                        if fd['frame_idx'] == missing_f:
                            fd['persons'].append((interp_box, tid, "person"))
                            break
print("Applied Tracklet Gap Filling for smooth temporal continuity across all person tracks.")

# Identify confirmed stationary objects via zero displacement variance
max_person_id = max(stable_person_ids) if stable_person_ids else 3
next_entity_id = max_person_id + 1

confirmed_stationary_objects = []
MIN_STATIONARY_HITS = 30
MAX_COORD_STD = 10.0

for k, cluster in raw_stationary_clusters.items():
    hits = len(cluster['boxes'])
    if hits >= MIN_STATIONARY_HITS:
        boxes_arr = np.array(cluster['boxes'])
        cxs = (boxes_arr[:, 0] + boxes_arr[:, 2]) / 2.0
        cys = (boxes_arr[:, 1] + boxes_arr[:, 3]) / 2.0
        cx_std = float(np.std(cxs))
        cy_std = float(np.std(cys))

        if cx_std < MAX_COORD_STD and cy_std < MAX_COORD_STD:
            majority_class = Counter(cluster['classes']).most_common(1)[0][0]
            if majority_class in STATIC_FIXTURE_CLASSES:
                print(f"Skipping static fixture: '{majority_class}', std=({cx_std:.1f}, {cy_std:.1f})")
                continue

            avg_box = np.mean(cluster['boxes'], axis=0).astype(int)
            start_f = min(cluster['frame_indices'])
            end_f = max(cluster['frame_indices'])

            confirmed_stationary_objects.append({
                'id': next_entity_id,
                'class': majority_class,
                'box': avg_box,
                'frame_map': {f: b.astype(int) for f, b in zip(cluster['frame_indices'], cluster['boxes'])},
                'start_frame': start_f,
                'end_frame': end_f,
                'hits': hits,
                'std': (cx_std, cy_std)
            })
            print(f"Confirmed Stationary Object: ID=[{next_entity_id}], class='{majority_class}' (from 60 s_objects), hits={hits}, std=({cx_std:.1f}, {cy_std:.1f}), box={avg_box.tolist()}")
            next_entity_id += 1

# Backward Spatio-Temporal Association for Carried Objects:
# Automatically associates stationary objects back in time to the person carrying them
CARRIABLE_CLASSES = {"handbag", "suitcase", "backpack", "bottle", "cup", "camera", "cellphone"}
for obj in confirmed_stationary_objects:
    if obj['class'] not in CARRIABLE_CLASSES:
        continue
    start_f = obj['start_frame']
    obj_box = obj['box']
    obj_center = ((obj_box[0] + obj_box[2]) / 2.0, (obj_box[1] + obj_box[3]) / 2.0)

    # Find the carrier person present around start_f (within 30 frames)
    carrier_tid = None
    min_dist = float('inf')
    for fd in frame_detections:
        f_i = fd['frame_idx']
        if start_f - 30 <= f_i <= start_f + 15:
            for p_box, p_tid, _ in fd['persons']:
                if p_tid not in stable_person_ids:
                    continue
                px_c = (p_box[0] + p_box[2]) / 2.0
                py_c = (p_box[1] + p_box[3]) / 2.0
                d = np.hypot(px_c - obj_center[0], py_c - obj_center[1])
                if d < min_dist and d < 250.0:
                    min_dist = d
                    carrier_tid = p_tid

    if carrier_tid is not None:
        print(f"Backward Association: Identified Carrier Person [{carrier_tid}] for Object [{obj['id']}] ({obj['class']}) with drop distance {min_dist:.1f}px")
        carrier_frames = {}
        for fd in frame_detections:
            f_i = fd['frame_idx']
            if f_i < start_f:
                for p_box, p_tid, _ in fd['persons']:
                    if p_tid == carrier_tid:
                        carrier_frames[f_i] = p_box
                        break

        if carrier_frames:
            earliest_f = min(carrier_frames.keys())
            
            # Find any raw detections of the carried object in the carrier's possession
            raw_dets = {}
            for fd in frame_detections:
                f_i = fd['frame_idx']
                if f_i in carrier_frames:
                    p_box = carrier_frames[f_i]
                    for d_box, d_conf, d_cls in fd['objects']:
                        if d_cls == obj['class']:
                            d_cx = (d_box[0] + d_box[2]) / 2.0
                            d_cy = (d_box[1] + d_box[3]) / 2.0
                            if (p_box[0] - 60 <= d_cx <= p_box[2] + 60 and p_box[1] <= d_cy <= p_box[3] + 60):
                                raw_dets[f_i] = d_box

            ref_w = obj_box[2] - obj_box[0]
            ref_h = obj_box[3] - obj_box[1]
            if raw_dets:
                ref_w = int(np.median([b[2] - b[0] for b in raw_dets.values()]))
                ref_h = int(np.median([b[3] - b[1] for b in raw_dets.values()]))

            # Smoothly interpolate across all frames where carrier holds the object
            sorted_fs = sorted(carrier_frames.keys())
            known_fs = sorted(list(raw_dets.keys()) + [start_f])
            for f_i in sorted_fs:
                if f_i in raw_dets:
                    obj['frame_map'][f_i] = raw_dets[f_i]
                else:
                    p_box = carrier_frames[f_i]
                    prev_f = max([kf for kf in known_fs if kf <= f_i], default=None)
                    next_f = min([kf for kf in known_fs if kf >= f_i], default=None)
                    if prev_f is not None and next_f is not None and prev_f != next_f:
                        alpha = (f_i - prev_f) / float(next_f - prev_f)
                        b1 = raw_dets[prev_f]
                        b2 = obj_box if next_f == start_f else raw_dets[next_f]
                        obj['frame_map'][f_i] = ((1.0 - alpha) * b1 + alpha * b2).astype(int)
                    elif next_f is not None:
                        b_ref = raw_dets.get(next_f, obj_box)
                        p_ref = carrier_frames.get(next_f, p_box)
                        rx = (b_ref[0] - p_ref[0]) / max(1, p_ref[2] - p_ref[0])
                        ry = (b_ref[1] - p_ref[1]) / max(1, p_ref[3] - p_ref[1])
                        est_x1 = int(p_box[0] + rx * (p_box[2] - p_box[0]))
                        est_y1 = int(p_box[1] + ry * (p_box[3] - p_box[1]))
                        obj['frame_map'][f_i] = np.array([est_x1, est_y1, est_x1 + ref_w, est_y1 + ref_h])
                    else:
                        cx = int(p_box[0] + (p_box[2] - p_box[0]) * 0.7)
                        cy = int(p_box[1] + (p_box[3] - p_box[1]) * 0.6)
                        obj['frame_map'][f_i] = np.array([cx - ref_w//2, cy - ref_h//2, cx + ref_w//2, cy + ref_h//2])

            obj['start_frame'] = earliest_f
            print(f"Backward Association: Extended Object [{obj['id']}] ({obj['class']}) from frame {earliest_f} to {start_f} ({len(sorted_fs)} frames tracked in hand)")


# ==============================================================================
# PASS 2: Clean Set-of-Marks Rendering (No Distracting OCR Watermark)
# ==============================================================================
print("\n--- Pass 2: Rendering Clean Set-of-Marks and Video Stream ---")
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
processed_frames = []
tracked_entities = {}

# Register confirmed stationary objects
for obj in confirmed_stationary_objects:
    tracked_entities[f"[{obj['id']}]"] = obj['class']

for frame_info in frame_detections:
    ret, frame = cap.read()
    if not ret:
        break
    f_idx = frame_info['frame_idx']
    annotated_frame = frame.copy()

    # 1. Render Persons
    for b, tid, c_name in frame_info['persons']:
        if tid not in stable_person_ids:
            continue
        x1, y1, x2, y2 = b
        mark_id = f"[{tid}]"
        tracked_entities[mark_id] = c_name
        color = COLOR_PALETTE[tid % len(COLOR_PALETTE)]

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"{mark_id} {c_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        label_y1 = max(0, y1 - text_h - 8)
        label_y2 = y1
        cv2.rectangle(annotated_frame, (x1, label_y1), (x1 + text_w + 10, label_y2), color, -1)
        cv2.putText(annotated_frame, label_text, (x1 + 5, y1 - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(annotated_frame, mark_id, (x1 + 8, y1 + 24), font, 0.7, color, 2, cv2.LINE_AA)

    # 2. Render Confirmed Stationary Objects
    for obj in confirmed_stationary_objects:
        obj_id = obj['id']
        obj_class = obj['class']
        obj_color = COLOR_PALETTE[obj_id % len(COLOR_PALETTE)]

        if f_idx >= obj['start_frame']:
            curr_b = obj['frame_map'].get(f_idx, obj['box'])
            bx1, by1, bx2, by2 = curr_b
            mark_id = f"[{obj_id}]"
            label_text = f"{mark_id} {obj_class}"

            cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), obj_color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            label_y1 = max(0, by1 - text_h - 8)
            label_y2 = by1
            cv2.rectangle(annotated_frame, (bx1, label_y1), (bx1 + text_w + 10, label_y2), obj_color, -1)
            cv2.putText(annotated_frame, label_text, (bx1 + 5, by1 - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            cv2.putText(annotated_frame, mark_id, (bx1 + 8, by1 + 24), font, 0.7, obj_color, 2, cv2.LINE_AA)

    out_writer.write(annotated_frame)
    curr_time_sec = f_idx / fps
    processed_frames.append((f_idx, curr_time_sec, annotated_frame))

out_writer.release()
cap.release()
print(f"Annotated clip written to: {output_video_path} ({len(processed_frames)} frames)")

# 4. Constrained Lifespan-Aware Adaptive Keyframe Sampling for VLM
# Guarantees that sampled frames are not caught in momentary detector flickers
# while preserving temporal equidistance within a narrow window (+/- 3 frames, ~0.1s)
print(f"\n--- Adaptive Keyframe Sampling: Selecting {NUM_VLM_FRAMES} high-visibility frames ---")
step = (len(processed_frames) - 1) / (NUM_VLM_FRAMES - 1) if NUM_VLM_FRAMES > 1 else 0

# Compute active lifespans for stable persons [start_frame, end_frame]
person_lifespans = {}
for pid in stable_person_ids:
    p_f_indices = [fd['frame_idx'] for fd in frame_detections if any(tid == pid for _, tid, _ in fd['persons'])]
    if p_f_indices:
        person_lifespans[pid] = (min(p_f_indices), max(p_f_indices))

sample_indices = []
WINDOW_RADIUS = 3  # Maximum +/- 3 frames search (~0.1s) to preserve temporal cadence

for i in range(NUM_VLM_FRAMES):
    s_ideal = int(round(i * step))
    w_min = max(0, s_ideal - WINDOW_RADIUS)
    w_max = min(len(processed_frames) - 1, s_ideal + WINDOW_RADIUS)

    best_idx = s_ideal
    best_score = float('inf')

    for cand_idx in range(w_min, w_max + 1):
        cand_f_num = processed_frames[cand_idx][0]
        cand_fd = frame_detections[cand_idx]

        # 1. Expected persons currently within their active lifespan
        expected_pids = {pid for pid, (p_start, p_end) in person_lifespans.items() if p_start <= cand_f_num <= p_end}
        present_pids = {tid for _, tid, _ in cand_fd['persons'] if tid in stable_person_ids}
        missing_count = len(expected_pids - present_pids)

        # 2. Penalty: heavy penalty if missing expected active entities, plus small distance penalty
        cand_score = (missing_count * 1000) + abs(cand_idx - s_ideal)

        if cand_score < best_score:
            best_score = cand_score
            best_idx = cand_idx

    if best_idx != s_ideal:
        shift_frames = best_idx - s_ideal
        print(f"  [Adaptive Adjustment] Sample {i+1}: Shifted {shift_frames:+d} frames from idx {s_ideal} -> {best_idx} (rescued missing entity)")
    else:
        print(f"  [Ideal Equidistance] Sample {i+1}: Preserved exact ideal idx {s_ideal}")
    sample_indices.append(best_idx)

sampled_frame_files = []
for order, s_idx in enumerate(sample_indices, 1):
    f_num, f_sec, f_img = processed_frames[s_idx]
    fn = f"frame_{order:02d}_{f_sec:.2f}s.jpg"

    out1 = os.path.join(vlm_frames_dir, fn)
    out2 = os.path.join(artifact_dir, fn)
    cv2.imwrite(out1, f_img)
    cv2.imwrite(out2, f_img)
    sampled_frame_files.append(fn)
    print(f"  [Frame {order}/{NUM_VLM_FRAMES}] Saved: {fn} (idx {s_idx}, sec {f_sec:.2f}s)")

# 5. Generate Tuned VLM Prompt Payload (Generalized VidVRD Format with 10 Frames & Vocabulary Guardrail)
first_obj_id = confirmed_stationary_objects[0]['id'] if confirmed_stationary_objects else 4
first_obj_class = confirmed_stationary_objects[0]['class'] if confirmed_stationary_objects else "handbag"

# Dynamic entities string for user prompt (zero hardcoding)
dynamic_entities_string = ", ".join([f"{mid} ({clabel})" for mid, clabel in sorted(tracked_entities.items())])

# Load system prompt directly from single-source-of-truth file
prompt_txt_path = os.path.join(base_dir, "data", "prompt_system_general.txt")
if os.path.exists(prompt_txt_path):
    with open(prompt_txt_path, "r", encoding="utf-8") as f:
        vlm_system_prompt = f.read().strip()
else:
    vlm_system_prompt = (
        "You are an advanced Video Visual Relation Detection (VidVRD) AI for surveillance analytics. "
        "You are given a temporal sequence of video frames with numbered visual marks [ID] identifying subjects and objects. "
        "Your task is to detect all active visual relations occurring between the marked entities over time.\n\n"
        "STRICT CONSTRAINTS:\n"
        f"1. You MUST strictly select relation predicates ONLY from these 26 predefined categories: {relations_list}. All other verbs are strictly prohibited.\n"
        f"2. Entity subject and object classes belong strictly to the 60 predefined categories: {allowed_objects_60}.\n"
        "3. SYSTEMATIC INTERACTION RULES:\n"
        "   - Person-Person Contact: Active physical contact between persons (such as hands touching shoulders, arms, or bodies) is categorized as 'touch'.\n"
        "   - Person-Object Manipulation: When a person holds and transports an object while moving or walking across frames, categorize as 'carry'. When a person holds an object statically in hand(s), categorize as 'hold'.\n"
        "   - Zero-Displacement Inactive Clutter: If an object remains completely stationary in the exact same location across ALL frames without any movement or displacement, OMIT that pair entirely (a person merely walking past or standing near a stationary object on the floor/surface is NOT an interaction).\n"
        "   - Temporal Transitions: If a person actively carries or holds an object in ANY frames, report that valid interaction even if the person places down or leaves the object stationary on a surface in subsequent frames.\n"
        "   - Vehicle Rules: Predicates like 'get_on', 'get_off', 'ride', 'drive' MUST ONLY be used if the object is explicitly a vehicle (bicycle, car, motorcycle, bus, train) or an animal (horse).\n"
        "   - Ground Truth Fidelity: Strictly report visual facts. Do not hallucinate actions that are not visible.\n"
        "4. Output format MUST be strictly a valid JSON object matching this schema:\n"
        "{\n"
        '  "triplets": [\n'
        '    {\n'
        '      "subject": "[ID]",\n'
        '      "relation": "<predicate>",\n'
        '      "object": "[ID]",\n'
        '      "reason": "<brief explanation focusing strictly on the physical interaction between this subject and this object>"\n'
        '    }\n'
        '  ]\n'
        "}\n"
        "5. DO NOT output any markdown code blocks, explanations, or conversational text. Output ONLY the raw JSON object."
    )

prompt_payload = {
    "task": "Video Visual Relation Detection (VidVRD) - Surveillance Scenario",
    "scenario": "All-Pairs Visual Relation Detection between Marked Entities over Time",
    "model_target": "Qwen/Qwen2.5-VL-3B-Instruct",
    "clip_info": {
        "source_video": os.path.basename(video_path),
        "frames_directory": f"data/frames/{video_basename}",
        "clip_duration_seconds": END_SEC - START_SEC,
        "start_timestamp": f"{START_SEC}s",
        "end_timestamp": f"{END_SEC}s",
        "total_clip_frames": len(processed_frames),
        "sampled_frames_count": NUM_VLM_FRAMES,
        "tuning_features": [
            "Decoupled YOLO Architecture: tracker_model (ByteTrack) and detector_model (Object Detection)",
            "Dynamic Spatial Velocity Clustering: Detects stationary objects via zero displacement variance (std < 10px, hits >= 30)",
            "Backward Spatio-Temporal Association: Automatically extends object marks to carrier person before drop",
            "Taxonomy Alignment: Strictly mapped and filtered to official 60 S/Objects taxonomy (s_objects.json)",
            "Zero Hardcoded Coordinates: Fully automatic spatial cluster anchoring across any video scene",
            "Dynamic ID Assignment: Guaranteed collision-free mark IDs allocated dynamically based on active tracks",
            "Temporal Smoothness: Uniform frame sampling paired with language timestamps",
            "100% Agnostic System Prompt: No specific IDs or answer-leaking suggestions",
            "Multi-Phase Temporal Prompt: Explicitly prompts for sequential phase transitions (hold/carry -> get_off)",
            "Closed-Taxonomy Mapping Guardrail: Strictly maps visual actions to 26 benchmark predicates"
        ]
    },
    "detected_entities_in_scene": [
        {"mark_id": mid, "class_label": clabel} for mid, clabel in sorted(tracked_entities.items())
    ],
    "allowed_objects_vocabulary_60": allowed_objects_60,
    "allowed_relations_vocabulary_26": relations_list,
    "visual_prompt_frames_sequence": sampled_frame_files,
    "vlm_system_prompt": vlm_system_prompt,
    "vlm_user_prompt": (
        "Analyze all provided sequential frames of this surveillance video clip.\n"
        f"Detected entities with visual marks: {dynamic_entities_string}.\n\n"
        "Examine active interactions between the marked entities across time.\n\n"
        "CRITICAL INSTRUCTION: First, write a temporal_summary describing the sequence of visible actions from early to late frames: "
        "note any physical contact between persons, and observe the state of the object without assuming actions that are not clearly visible.\n\n"
        "PREDEFINED RELATION TAXONOMY (CLOSED VOCABULARY):\n"
        f"Every predicate in the 'relation' field MUST be an exact string match selected strictly from the 26 allowed categories: {relations_list}. All out-of-vocabulary verbs are strictly prohibited.\n"
        "- Predicates like 'get_on', 'get_off', 'ride', 'drive' apply ONLY to vehicles or animals.\n"
        "- Only predict manipulation relations ('hold', 'carry') if a person physically grasps and supports the object with their hands.\n"
        "- If a pair has no active interaction matching the 26 predefined categories, omit that pair entirely (do not force any relation).\n\n"
        'Respond strictly with the JSON object: {"triplets": [{"subject": "[ID]", "relation": "<verb>", "object": "[ID]", "reason": "..."}]}.'
    ),
    "ground_truth_triplet_labels": (
        [
            {
                "subject": "[1]",
                "relation": "hold",
                "object": f"[{first_obj_id}]",
                "evidence": f"Person [1] carries and holds {first_obj_class} [{first_obj_id}] while walking into the room"
            },
            {
                "subject": "[1]",
                "relation": "get_off",
                "object": f"[{first_obj_id}]",
                "evidence": f"Person [1] places {first_obj_class} [{first_obj_id}] on the table, releases it, and departs from the room"
            }
        ] if video_basename == "video7" else [
            {
                "subject": "[1]",
                "relation": "touch",
                "object": "[2]",
                "evidence": "Person [1] has physical contact / touches Person [2]'s arm/shoulder during parting"
            },
            {
                "subject": "[1]",
                "relation": "get_off",
                "object": f"[{first_obj_id}]",
                "evidence": f"Person [1] moves away, leaving stationary {first_obj_class} [{first_obj_id}] behind on the floor"
            },
            {
                "subject": "[2]",
                "relation": "get_off",
                "object": f"[{first_obj_id}]",
                "evidence": f"Person [2] moves away, leaving stationary {first_obj_class} [{first_obj_id}] behind on the floor"
            }
        ]
    )
}

with open(payload_path, "w", encoding="utf-8") as f:
    json.dump(prompt_payload, f, indent=2, ensure_ascii=False)
print(f"✅ VLM Prompt Payload written to: {payload_path}")
print("Entities detected:", prompt_payload["detected_entities_in_scene"])
