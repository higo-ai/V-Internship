import os
import json
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Paths & Configurations
base_dir = r'C:\AIThucChien\VinFast Internship'
video_path = os.path.join(base_dir, 'data', 'video1.avi')
output_video_path = os.path.join(base_dir, 'data', 'annotated_clip_5s.mp4')
vlm_frames_dir = os.path.join(base_dir, 'data', 'vlm_input_frames')
artifact_dir = r'C:\Users\higoi\.gemini\antigravity-ide\brain\5bde3e10-5ab7-412f-9811-185e514054b1\vlm_input_frames'
payload_path = os.path.join(base_dir, 'vlm_prompt_payload.json')

os.makedirs(vlm_frames_dir, exist_ok=True)
os.makedirs(artifact_dir, exist_ok=True)

# 2. Golden segment: seconds 36.0 to 41.0 (5 seconds)
START_SEC = 36.0
END_SEC = 41.0
NUM_VLM_FRAMES = 8  # Uniform sampling target (flexible: 8, 10, 12, 16)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

start_frame = int(START_SEC * fps)
end_frame = int(END_SEC * fps)
clip_frame_count = end_frame - start_frame

print(f'Original Video: {total_frames} frames, {fps:.2f} FPS')
print(f'Clipping segment: {START_SEC}s - {END_SEC}s (frames {start_frame} to {end_frame}, total {clip_frame_count} frames)')

# 3. Initialize YOLO Tracking Model (Pre-trained lightweight model)
print('Loading YOLO Tracking model...')
model = YOLO('yolo11n.pt')

# Distinct high-contrast colors for Set-of-Marks (BGR format)
COLOR_PALETTE = [
    (0, 0, 255),    # Red
    (255, 140, 0),  # Deep Sky Blue
    (0, 215, 255),  # Gold / Yellow
    (50, 205, 50),  # Lime Green
    (238, 130, 238),# Violet
    (0, 255, 255),  # Cyan
]

# Set up VideoWriter for MP4 output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
processed_frames = []
tracked_entities = {}

frame_idx = start_frame
while frame_idx < end_frame:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO tracking with ByteTrack
    # classes: 0=person, 24=backpack, 26=handbag, 28=suitcase
    results = model.track(
        frame,
        persist=True,
        tracker='bytetrack.yaml',
        classes=[0, 24, 26, 28],
        conf=0.25,
        verbose=False
    )
    
    annotated_frame = frame.copy()
    
    if results and results[0].boxes and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        
        for box, tid, cid in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = box
            class_name = model.names[cid]
            mark_id = f'[{tid}]'
            
            tracked_entities[mark_id] = class_name
            
            color = COLOR_PALETTE[tid % len(COLOR_PALETTE)]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label banner with Set-of-Marks format
            label_text = f'{mark_id} {class_name}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            label_y1 = max(0, y1 - text_h - 8)
            label_y2 = y1
            cv2.rectangle(annotated_frame, (x1, label_y1), (x1 + text_w + 10, label_y2), color, -1)
            cv2.putText(annotated_frame, label_text, (x1 + 5, y1 - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            # Put mark ID clearly inside the box corner
            id_badge = f'{mark_id}'
            cv2.putText(annotated_frame, id_badge, (x1 + 8, y1 + 24), font, 0.7, color, 2, cv2.LINE_AA)
    
    # Add timestamp watermark
    curr_time_sec = frame_idx / fps
    time_badge = f'Time: {curr_time_sec:.2f}s | Frame: {frame_idx}'
    cv2.putText(annotated_frame, time_badge, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    
    out_writer.write(annotated_frame)
    processed_frames.append((frame_idx, curr_time_sec, annotated_frame))
    frame_idx += 1

out_writer.release()
cap.release()
print(f'Annotated clip written to: {output_video_path} ({len(processed_frames)} frames)')

# 4. Uniform Frame Sampling for VLM (NUM_VLM_FRAMES = 8)
print(f'Uniformly sampling {NUM_VLM_FRAMES} frames across clip...')
step = len(processed_frames) / NUM_VLM_FRAMES
sample_indices = [int(i * step) for i in range(NUM_VLM_FRAMES)]

sampled_frame_files = []
for order, s_idx in enumerate(sample_indices, 1):
    f_num, f_sec, f_img = processed_frames[s_idx]
    fn = f'vlm_frame_{order:02d}_{f_sec:.2f}s.jpg'
    
    out1 = os.path.join(vlm_frames_dir, fn)
    out2 = os.path.join(artifact_dir, fn)
    cv2.imwrite(out1, f_img)
    cv2.imwrite(out2, f_img)
    sampled_frame_files.append(fn)
    print(f'  [Frame {order}/{NUM_VLM_FRAMES}] Saved: {fn} (idx {s_idx}, sec {f_sec:.2f}s)')

# 5. Load 26 relations dictionary
with open(os.path.join(base_dir, 'relations.json'), encoding='utf-8') as f:
    relations_list = json.load(f)

# 6. Generate VLM Prompt Payload (JSON schema for Qwen2-VL-2B)
prompt_payload = {
    'task': 'Video Visual Relation Detection (VidVRD) with Set-of-Marks Visual Prompting',
    'model_target': 'Qwen2-VL-2B-Instruct',
    'clip_info': {
        'source_video': 'video1.avi',
        'clip_duration_seconds': END_SEC - START_SEC,
        'start_timestamp': f'{START_SEC}s',
        'end_timestamp': f'{END_SEC}s',
        'total_clip_frames': len(processed_frames),
        'sampled_frames_count': NUM_VLM_FRAMES
    },
    'detected_entities_in_scene': [
        {'mark_id': mid, 'class_label': clabel} for mid, clabel in sorted(tracked_entities.items())
    ],
    'allowed_relations_vocabulary_26': relations_list,
    'visual_prompt_frames_sequence': sampled_frame_files,
    'vlm_system_prompt': (
        'You are an advanced Video Visual Relation Detection AI. You are given a sequential series of video frames '
        'with numbered visual marks [ID] identifying subjects and objects. Your task is to detect all active relations '
        'between the marked entities over time. You MUST strictly choose relations from the provided 26 relation categories. '
        'Output strictly valid JSON list of triplets: [{" subject\: \[ID]\, \relation\: \<verb>\, \object\: \[ID]\}].'
 ),
 'vlm_user_prompt': (
 f'Analyze the {NUM_VLM_FRAMES} sequential frames of this surveillance clip. '
 f'Entities detected with visual marks: {list(tracked_entities.keys())}. '
 f'Which relations from the 26 categories are actively occurring between these marked entities across these frames? '
 f'Respond ONLY with a JSON array of triplets.'
 ),
 'ground_truth_triplet_labels': [
 {'subject': '[1]', 'relation': 'hold', 'object': '[16]', 'description': 'Person [1] holds handbag [16] while taking it off'},
 {'subject': '[1]', 'relation': 'carry', 'object': '[16]', 'description': 'Person [1] carries handbag [16] on shoulder at start of clip'}
 ]
}

with open(payload_path, 'w', encoding='utf-8') as f:
 json.dump(prompt_payload, f, indent=4, ensure_ascii=False)

print(f'VLM Prompt Payload written to: {payload_path}')
print('SUCCESS! ALL 3 STEPS OF PHASE 3 COMPLETED!')