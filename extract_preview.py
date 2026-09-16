import os
import cv2

video_path = r'C:\AIThucChien\VinFast Internship\data\video1.avi'
output_dir = r'C:\AIThucChien\VinFast Internship\data\preview_frames'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print('ERROR: Cannot open video')
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0

print(f'Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s')

# Sample timestamps across the video
timestamps_sec = [5, 12, 20, 28, 35, 45, 55, 65]
extracted = []

for sec in timestamps_sec:
    frame_idx = int(sec * fps)
    if frame_idx >= total_frames:
        frame_idx = total_frames - 1
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        filename = f'frame_{sec:02d}s_f{frame_idx:04d}.jpg'
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, frame)
        extracted.append(filename)
        print(f'Extracted: {filename} at {sec}s (frame {frame_idx})')

cap.release()
print(f'Successfully extracted {len(extracted)} preview frames to {output_dir}')