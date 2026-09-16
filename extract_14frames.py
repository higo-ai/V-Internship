import os
import shutil
import cv2

video_path = r'C:\AIThucChien\VinFast Internship\data\video1.avi'
output_dir = r'C:\AIThucChien\VinFast Internship\data\preview_frames'
artifact_dir = r'C:\Users\higoi\.gemini\antigravity-ide\brain\5bde3e10-5ab7-412f-9811-185e514054b1\preview_14frames'

# Clean old frames
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

if os.path.exists(artifact_dir):
    shutil.rmtree(artifact_dir)
os.makedirs(artifact_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

timestamps = [
    (3, '01_03s_walk_in'),
    (8, '02_08s_wave1'),
    (18, '03_18s_wave2'),
    (27, '04_27s_look_around'),
    (33, '05_33s_backpack_enter'),
    (35, '06_35s_wave_each_other'),
    (38, '07_38s_talk'),
    (41, '08_41s_take_off_backpack'),
    (44, '09_44s_backpack_on_floor'),
    (46, '10_46s_arms_on_shoulders'),
    (48, '11_48s_leave_scene'),
    (50, '12_50s_person_walk_past'),
    (60, '13_60s_person_running'),
    (68, '14_68s_two_persons_walk_past')
]

extracted = []
for sec, name in timestamps:
    frame_idx = int(sec * fps)
    if frame_idx >= total_frames:
        frame_idx = total_frames - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        fn = f'{name}.jpg'
        out1 = os.path.join(output_dir, fn)
        out2 = os.path.join(artifact_dir, fn)
        cv2.imwrite(out1, frame)
        cv2.imwrite(out2, frame)
        extracted.append(fn)
        print(f'Saved: {fn} (sec {sec}, frame {frame_idx})')

cap.release()
print(f'TOTAL EXTRACTED: {len(extracted)} frames')