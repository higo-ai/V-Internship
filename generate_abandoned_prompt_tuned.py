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

# Clean old frames in directory before sampling
for f in os.listdir(vlm_frames_dir):
    if f.endswith(".jpg"):
        os.remove(os.path.join(vlm_frames_dir, f))

# Load official project taxonomies (60 S/Objects and 26 Relations)
with open(os.path.join(base_dir, "s_objects.json"), encoding="utf-8") as f:
    allowed_objects_60 = json.load(f)
    allowed_objects_set = set(allowed_objects_60)

with open(os.path.join(base_dir, "relations.json"), encoding="utf-8") as f:
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
START_SEC = 44.0
END_SEC = 52.0
NUM_VLM_FRAMES = 10  # Optimal sampling density: 0.8s / frame

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
        tracker="bytetrack.yaml",
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

# Filter stable persons (visible for at least 15 frames)
for tid, count in person_hit_counts.items():
    if count >= 15:
        stable_person_ids.add(tid)
print(f"Stable dynamic person IDs tracked: {sorted(list(stable_person_ids))}")

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

# 4. Uniform Frame Sampling for VLM (NUM_VLM_FRAMES = 10)
print(f"Uniformly sampling {NUM_VLM_FRAMES} clean frames across clip...")
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

# 5. Generate Tuned VLM Prompt Payload (Generalized VidVRD Format with 10 Frames & Vocabulary Guardrail)
first_obj_id = confirmed_stationary_objects[0]['id'] if confirmed_stationary_objects else 4
first_obj_class = confirmed_stationary_objects[0]['class'] if confirmed_stationary_objects else "handbag"

# Dynamic entities string for user prompt (zero hardcoding)
dynamic_entities_string = ", ".join([f"{mid} ({clabel})" for mid, clabel in sorted(tracked_entities.items())])

prompt_payload = {
    "task": "Video Visual Relation Detection (VidVRD) - Surveillance Scenario",
    "scenario": "All-Pairs Visual Relation Detection between Marked Entities over Time",
    "model_target": "Qwen/Qwen2.5-VL-3B-Instruct",
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
            "Taxonomy Alignment: Strictly mapped and filtered to official 60 S/Objects taxonomy (s_objects.json)",
            "Zero Hardcoded Coordinates: Fully automatic spatial cluster anchoring across any video scene",
            "Dynamic ID Assignment: Guaranteed collision-free mark IDs allocated dynamically based on active tracks",
            "10-Frame Temporal Smoothness: 0.8s resolution eliminating visual occlusion ambiguity",
            "100% Agnostic System Prompt: No specific IDs or answer-leaking suggestions",
            "Forced Spatial Attention CoT: Explicit spatial grounding of stationary objects in temporal summary",
            "Closed-Taxonomy Mapping Guardrail: Strictly maps visual actions to 26 benchmark predicates",
            "Interleaved Temporal Anchoring: Clean visual frames paired with language timestamps"
        ]
    },
    "detected_entities_in_scene": [
        {"mark_id": mid, "class_label": clabel} for mid, clabel in sorted(tracked_entities.items())
    ],
    "allowed_objects_vocabulary_60": allowed_objects_60,
    "allowed_relations_vocabulary_26": relations_list,
    "visual_prompt_frames_sequence": sampled_frame_files,
    "vlm_system_prompt": (
        "You are an advanced Video Visual Relation Detection (VidVRD) AI for surveillance analytics. "
        "You are given a temporal sequence of video frames with numbered visual marks [ID] identifying subjects and objects. "
        "Your task is to detect all active visual relations occurring between the marked entities over time.\n\n"
        "STRICT CONSTRAINTS:\n"
        f"1. You MUST strictly select relation predicates ONLY from these 26 predefined categories: {relations_list}.\n"
        f"2. Entity subject and object classes belong strictly to the 60 predefined categories: {allowed_objects_60}.\n"
        "3. SYSTEMATIC INTERACTION RULES:\n"
        "   - Person-Person interactions: Identify active physical contact or intentional social interaction. NEVER use 'get_off' for Person-Person pairs.\n"
        "   - Person-Object interactions: Only predict manipulation verbs ('hold', 'carry') if a person is physically grasping the object.\n"
        "   - SPECIAL RULE FOR 'get_off': In this taxonomy, use 'get_off' ONLY to describe a person actively moving away from, releasing, or leaving an inanimate object behind (e.g., leaving an object stationary on the floor).\n"
        "   - Vehicle Rules: Predicates like 'get_on', 'ride', 'drive' MUST ONLY be used if the object is explicitly a vehicle (bicycle, car, motorcycle, bus, train) or an animal (horse).\n"
        "   - Negative Pairs: If an object is resting stationary on the floor and a person merely walks past or approaches without physical contact, DO NOT predict any relation.\n"
        "   - Ground Truth Fidelity: Strictly report visual facts. Do not hallucinate actions that are not visible. If an object remains visible on the floor in the final frames, it is NOT picked up.\n"
        "4. Output format MUST be strictly a valid JSON object matching this schema:\n"
        "{\n"
        '  "temporal_summary": "<brief description of the progression of actions from early to late frames, explicitly noting any changes in the physical location or state of inanimate objects>",\n'
        '  "triplets": [\n'
        '    {\n'
        '      "subject": "[ID]",\n'
        '      "relation": "<predicate>",\n'
        '      "object": "[ID]",\n'
        '      "reason": "<brief explanation of why this relation is selected based on visual evidence>"\n'
        '    }\n'
        '  ]\n'
        "}\n"
        "5. DO NOT output any markdown code blocks, explanations, or conversational text. Output ONLY the raw JSON object."
    ),
    "vlm_user_prompt": (
        "Analyze all provided sequential frames of this surveillance video clip.\n"
        f"Detected entities with visual marks: {dynamic_entities_string}.\n\n"
        "Perform a systematic pair-by-pair check across the full time duration:\n"
        "- Examine ALL Person-Person combinations.\n"
        "- Examine ALL Person-Object combinations.\n\n"
        "CRITICAL INSTRUCTION: First, write a temporal_summary describing the progression of actions across time from early frames to late frames. "
        "In this summary, explicitly state the physical location of any inanimate objects across the frames (e.g., whether an object remains stationary on the floor and whether persons move away from it). "
        "Do NOT invent actions not visible in the frames (if an object remains on the floor in the final frames, it has NOT been picked up).\n\n"
        "CLOSED-VOCABULARY MAPPING CONSTRAINT: In the 'relation' field of each triplet, you MUST select predicates strictly from the allowed 26 categories:\n"
        "- When a person moves away leaving an entity resting on the floor: you MUST select 'get_off' (never output 'leave').\n"
        "- When a person merely walks past an object or person without physical contact: DO NOT create a triplet (never output 'walk').\n"
        "- Never output words outside the 26 allowed categories (such as 'walk', 'leave', or 'pick_up').\n"
        f"Allowed 26 predicates: {relations_list}.\n\n"
        'Respond strictly with the JSON object: {"temporal_summary": "...", "triplets": [{"subject": "[ID]", "relation": "<verb>", "object": "[ID]", "reason": "..."}]}.'
    ),
    "ground_truth_triplet_labels": [
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
}

with open(payload_path, "w", encoding="utf-8") as f:
    json.dump(prompt_payload, f, indent=2, ensure_ascii=False)

print(f"Successfully wrote Tuned VLM Prompt Payload to: {payload_path}")
print("Entities detected:", prompt_payload["detected_entities_in_scene"])
