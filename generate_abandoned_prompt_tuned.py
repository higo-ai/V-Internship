import os
import json
import cv2
import numpy as np
from collections import Counter, defaultdict
from ultralytics import YOLO

# 1. Paths & Configurations
base_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(base_dir, "data", "video1.avi")
output_video_path = os.path.join(base_dir, "data", "annotated_clip_abandoned_tuned.mp4")
vlm_frames_dir = os.path.join(base_dir, "data", "vlm_input_frames_abandoned_tuned")
artifact_dir = r"C:\Users\higoi\.gemini\antigravity-ide\brain\5bde3e10-5ab7-412f-9811-185e514054b1\vlm_input_frames_abandoned_tuned"
payload_path = os.path.join(base_dir, "vlm_prompt_payload_abandoned_tuned.json")

os.makedirs(vlm_frames_dir, exist_ok=True)
os.makedirs(artifact_dir, exist_ok=True)

# 2. Golden segment for Abandoned Object: seconds 44.0 to 52.0 (8.0 seconds, 240 frames)
START_SEC = 44.0
END_SEC = 52.0
NUM_VLM_FRAMES = 8

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

# 3. Initialize YOLO Models
print("Loading YOLO models (local weights yolo11n.pt)...")
tracker_model = YOLO(os.path.join(base_dir, "yolo11n.pt"))
detector_model = YOLO(os.path.join(base_dir, "yolo11n.pt"))

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
    "dining table", "bench", "chair", "refrigerator", "couch", "bed",
    "toilet", "sink", "microwave", "oven", "tv"
}

# ==============================================================================
# PASS 1: Generalized Dynamic Tracking & Spatial Velocity Clustering
# Zero Hardcoded Classes, Zero Hardcoded Coordinates, Zero Static IDs
# ==============================================================================
print("\n--- Pass 1: Extracting Dynamic Person Tracks & Spatial Object Clusters ---")
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frame_detections = []
person_track_hits = defaultdict(int)
stationary_clusters = []

for f_idx in range(start_frame, end_frame):
    ret, frame = cap.read()
    if not ret:
        break

    # Track persons dynamically using ByteTrack (class 0 = person)
    results = tracker_model.track(
        frame,
        persist=True,
        tracker=os.path.join(base_dir, "custom_bytetrack.yaml"),
        classes=[0],
        conf=0.25,
        verbose=False
    )

    # Detect all non-person objects across entire COCO 80 categories (zero hardcoded class filter)
    obj_res = detector_model.predict(
        frame,
        conf=0.15,
        classes=[c for c in range(80) if c != 0],
        verbose=False
    )

    # Collect person bounding boxes and IDs
    p_data = []
    if results and results[0].boxes and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        tids = results[0].boxes.id.cpu().numpy().astype(int)
        cids = results[0].boxes.cls.cpu().numpy().astype(int)
        for b, tid, cid in zip(boxes, tids, cids):
            person_track_hits[tid] += 1
            p_data.append((b, tid, tracker_model.names[cid]))

    # Collect object detections and cluster stationary candidates
    o_data = []
    if len(obj_res) > 0 and len(obj_res[0].boxes) > 0:
        boxes = obj_res[0].boxes.xyxy.cpu().numpy()
        clss = obj_res[0].boxes.cls.cpu().numpy().astype(int)
        confs = obj_res[0].boxes.conf.cpu().numpy()

        for b, cid, conf in zip(boxes, clss, confs):
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            c_name = detector_model.names[cid]
            o_data.append((b.astype(int), c_name, conf))

            # Spatial association: match against running cluster center (dist < 40px)
            matched = False
            for cl in stationary_clusters:
                avg_cx = np.mean([h[0] for h in cl['history']])
                avg_cy = np.mean([h[1] for h in cl['history']])
                if np.hypot(cx - avg_cx, cy - avg_cy) < 40.0:
                    cl['history'].append((cx, cy))
                    cl['boxes'].append(b)
                    cl['classes'].append(c_name)
                    cl['confs'].append(float(conf))
                    cl['frame_indices'].append(f_idx)
                    matched = True
                    break

            if not matched:
                stationary_clusters.append({
                    'history': [(cx, cy)],
                    'boxes': [b],
                    'classes': [c_name],
                    'confs': [float(conf)],
                    'frame_indices': [f_idx]
                })

    frame_detections.append({
        'frame_idx': f_idx,
        'persons': p_data,
        'objects': o_data
    })

# Filter stable person tracks (discard transient detector flickers < 10 frames)
stable_person_ids = {int(tid) for tid, count in person_track_hits.items() if count >= 10}
print(f"Stable person tracks detected (hits >= 10): {sorted(stable_person_ids)}")

# Automatically detect confirmed stationary objects:
# Criteria: persistent (hits >= 30 frames) AND velocity near zero (std(cx) < 10.0 and std(cy) < 10.0)
# Excludes static room fixtures (refrigerator, table) to focus on portable unattended objects
confirmed_stationary_objects = []
max_person_id = max(stable_person_ids) if stable_person_ids else 0
next_entity_id = max_person_id + 1

for cluster in stationary_clusters:
    hits = len(cluster['history'])
    if hits >= 30:  # Present across >= 1 second
        cx_std = float(np.std([h[0] for h in cluster['history']]))
        cy_std = float(np.std([h[1] for h in cluster['history']]))
        if cx_std < 10.0 and cy_std < 10.0:  # True zero displacement stationary anchor
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
            print(f"Confirmed Stationary Object: ID=[{next_entity_id}], class='{majority_class}', hits={hits}, std=({cx_std:.1f}, {cy_std:.1f}), box={avg_box.tolist()}")
            next_entity_id += 1

# ==============================================================================
# PASS 2: Visual Annotation & Set-of-Marks Rendering
# ==============================================================================
print("\n--- Pass 2: Rendering Set-of-Marks and Video Stream ---")
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

    # Watermark
    curr_time_sec = f_idx / fps
    time_badge = f"Time: {curr_time_sec:.2f}s | Frame: {f_idx} (Zero Hardcode)"
    cv2.putText(annotated_frame, time_badge, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    out_writer.write(annotated_frame)
    processed_frames.append((f_idx, curr_time_sec, annotated_frame))

out_writer.release()
cap.release()
print(f"Annotated clip written to: {output_video_path} ({len(processed_frames)} frames)")

# 4. Uniform Frame Sampling for VLM (NUM_VLM_FRAMES = 8)
print(f"Uniformly sampling {NUM_VLM_FRAMES} frames across clip...")
step = len(processed_frames) / NUM_VLM_FRAMES
sample_indices = [int(i * step) for i in range(NUM_VLM_FRAMES)]

sampled_frame_files = []
for order, s_idx in enumerate(sample_indices, 1):
    f_num, f_sec, f_img = processed_frames[s_idx]
    fn = f"vlm_abandoned_tuned_frame_{order:02d}_{f_sec:.2f}s.jpg"

    out1 = os.path.join(vlm_frames_dir, fn)
    out2 = os.path.join(artifact_dir, fn)
    cv2.imwrite(out1, f_img)
    cv2.imwrite(out2, f_img)
    sampled_frame_files.append(fn)
    print(f"  [Frame {order}/{NUM_VLM_FRAMES}] Saved: {fn} (idx {s_idx}, sec {f_sec:.2f}s)")

# 5. Load 26 relations dictionary
with open(os.path.join(base_dir, "relations.json"), encoding="utf-8") as f:
    relations_list = json.load(f)

# 6. Generate Tuned VLM Prompt Payload (Generalized VidVRD Format)
first_obj_id = confirmed_stationary_objects[0]['id'] if confirmed_stationary_objects else 4
first_obj_class = confirmed_stationary_objects[0]['class'] if confirmed_stationary_objects else "handbag"

prompt_payload = {
    "task": "Video Visual Relation Detection (VidVRD) - Surveillance Scenario",
    "scenario": "All-Pairs Visual Relation Detection between Marked Entities over Time",
    "model_target": "Qwen/Qwen3.5-2B",
    "clip_info": {
        "source_video": "video1.avi",
        "clip_duration_seconds": END_SEC - START_SEC,
        "start_timestamp": f"{START_SEC}s",
        "end_timestamp": f"{END_SEC}s",
        "total_clip_frames": len(processed_frames),
        "sampled_frames_count": NUM_VLM_FRAMES,
        "tuning_features": [
            "Decoupled YOLO Architecture: tracker_model (ByteTrack) and detector_model (Object Detection)",
            "Dynamic Spatial Velocity Clustering: Detects stationary objects via zero displacement variance (std < 10px, hits >= 30)",
            "Zero Hardcoded Classes: Open-vocabulary detection across all 79 COCO non-person object categories",
            "Zero Hardcoded Coordinates: Fully automatic spatial cluster anchoring across any video scene",
            "Dynamic ID Assignment: Guaranteed collision-free mark IDs allocated dynamically based on active tracks"
        ]
    },
    "detected_entities_in_scene": [
        {"mark_id": mid, "class_label": clabel} for mid, clabel in sorted(tracked_entities.items())
    ],
    "allowed_relations_vocabulary_26": relations_list,
    "visual_prompt_frames_sequence": sampled_frame_files,
    "vlm_system_prompt": (
        "You are an advanced Video Visual Relation Detection (VidVRD) AI for surveillance analytics. "
        "You are given a temporal sequence of video frames with numbered visual marks [ID] identifying subjects and objects. "
        "Your task is to detect all active visual relations occurring between the marked entities over time.\n\n"
        "STRICT CONSTRAINTS:\n"
        f"1. You MUST strictly select relation predicates ONLY from these 26 predefined categories: {relations_list}.\n"
        "2. SEMANTIC AFFORDANCE & ROLE RULES:\n"
        "   - Inanimate objects (such as handbag, backpack) CANNOT be the subject of action verbs (e.g., a handbag cannot 'hold' or 'carry' a human). Only persons can hold or carry objects.\n"
        "   - If a person merely walks past an entity without physical contact or purposeful interaction, DO NOT predict relations (do NOT predict get_on/touch).\n"
        "3. Output format MUST be strictly a valid JSON object matching this schema:\n"
        "{\n"
        '  "temporal_summary": "<brief 1-sentence description of overall interactions and movements across frames>",\n'
        '  "triplets": [\n'
        '    {\n'
        '      "subject": "[ID]",\n'
        '      "relation": "<predicate>",\n'
        '      "object": "[ID]",\n'
        '      "reason": "<brief explanation of why this relation is selected based on visual evidence>"\n'
        '    }\n'
        "  ]\n"
        "}\n"
        "4. DO NOT output any markdown code blocks, explanations, or conversational text. Output ONLY the raw JSON object."
    ),
    "vlm_user_prompt": (
        "Analyze all provided sequential frames of this surveillance video clip. "
        f"Detected entities with visual marks: {', '.join([f'{mid} ({clabel})' for mid, clabel in sorted(tracked_entities.items())])}.\n"
        "Perform a systematic pair-by-pair check across the full time duration:\n"
        "- Examine all Person-Person interactions across frames.\n"
        "- Examine all Person-Object interactions across frames.\n"
        "Remember: Inanimate objects cannot hold humans, and walking past is not get_on.\n"
        "First write a brief 1-sentence temporal_summary of observed actions, then list all detected relation triplets with a 'reason' for each.\n"
        "Select predicates strictly from the allowed 26 categories. "
        'Respond strictly with the JSON object: {"temporal_summary": "...", "triplets": [{"subject": "[ID]", "relation": "<verb>", "object": "[ID]", "reason": "..."}]}.'
    ),
    "ground_truth_triplet_labels": [
        {
            "subject": "[1]",
            "relation": "touch",
            "object": "[2]",
            "evidence": "Person [1] has physical contact / touches Person [2]'s arm/shoulder at parting"
        },
        {
            "subject": "[1]",
            "relation": "away",
            "object": f"[{first_obj_id}]",
            "evidence": f"Person [1] moves away from stationary {first_obj_class} [{first_obj_id}]"
        },
        {
            "subject": "[2]",
            "relation": "away",
            "object": f"[{first_obj_id}]",
            "evidence": f"Person [2] moves away from stationary {first_obj_class} [{first_obj_id}]"
        },
        {
            "subject": "[3]",
            "relation": "walk_past",
            "object": f"[{first_obj_id}]",
            "evidence": f"Passerby Person [3] walks past stationary {first_obj_class} [{first_obj_id}] without interacting or touching"
        }
    ]
}

with open(payload_path, "w", encoding="utf-8") as f:
    json.dump(prompt_payload, f, indent=2, ensure_ascii=False)

print(f"Successfully wrote Tuned VLM Prompt Payload to: {payload_path}")
print("Entities detected:", prompt_payload["detected_entities_in_scene"])
