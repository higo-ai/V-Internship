# -*- coding: utf-8 -*-
"""
PREVIEW & CONVERT VIDEO TOOL
Hỗ trợ 3 tính năng:
- Chế độ 1 (Convert MP4): Chuyển đổi video .avi sang .mp4 chuẩn H.264 (avc1) để xem trực tiếp trong Antigravity IDE.
- Chế độ 2 (Toàn cảnh): Băm đều 12-15 ảnh trên toàn bộ video để nắm cốt truyện.
- Chế độ 3 (Khoảnh khắc vàng): Zoom sâu vào khoảng thời gian cụ thể (--start_sec và --end_sec) để soi rõ hành động.
"""
import os
import sys
import time
import cv2
import argparse

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="Trích xuất ảnh mốc thời gian và chuyển đổi video để xem trong Antigravity")
    parser.add_argument("--video", type=str, default="data/videos/video1.mp4", help="Đường dẫn tới file video (.mp4 hoặc .avi)")
    parser.add_argument("--convert", action="store_true", help="Chuyển đổi video sang .mp4 chuẩn H.264 (avc1) để xem trực tiếp trên Antigravity")
    parser.add_argument("--output_mp4", type=str, default=None, help="Đường dẫn file .mp4 đầu ra khi convert (mặc định cùng tên cùng thư mục)")
    parser.add_argument("--num_frames", type=int, default=12, help="Số lượng ảnh muốn băm đều (mặc định 12 ảnh)")
    parser.add_argument("--interval", type=float, default=None, help="Hoặc chỉ định chu kỳ băm cứ mỗi N giây")
    parser.add_argument("--start_sec", type=float, default=None, help="Mốc giây bắt đầu zoom (tùy chọn)")
    parser.add_argument("--end_sec", type=float, default=None, help="Mốc giây kết thúc zoom (tùy chọn)")
    parser.add_argument("--output_dir", type=str, default=None, help="Thư mục lưu ảnh preview")
    return parser.parse_args()

def convert_to_mp4(video_path, output_mp4=None):
    """
    Chuyển đổi video bất kỳ (như .avi) sang .mp4 chuẩn H.264 (avc1)
    sử dụng thư viện OpenH264 (Cisco) để Antigravity IDE xem trực tiếp được.
    """
    base_name, ext = os.path.splitext(video_path)
    out_file = output_mp4 if output_mp4 else f"{base_name}.mp4"

    # Nếu file nguồn đã là MP4 và trùng file đích thì không cần convert
    if os.path.abspath(video_path).lower() == os.path.abspath(out_file).lower():
        print(f"ℹ️ [INFO] Video nguồn đã ở định dạng .mp4 ({video_path}). Không cần chuyển đổi!")
        return out_file

    # Nếu file đích đã tồn tại và đủ dung lượng thì bỏ qua không convert lại
    if os.path.exists(out_file) and os.path.getsize(out_file) > 100000:
        print("=" * 60)
        print(f"ℹ️ [INFO] File MP4 đã tồn tại sẵn: {out_file}")
        print("👉 Không cần convert lại, bạn có thể click mở xem trực tiếp ngay!")
        print("=" * 60)
        return out_file

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ [ERROR] Không thể mở video nguồn: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    print("=" * 60)
    print("🎥 CHUYỂN ĐỔI VIDEO SANG MP4 (CHUẨN H.264 / AVC1)")
    print("=" * 60)
    print(f"• Nguồn: {video_path}")
    print(f"• Thông số: {width}x{height}, {fps:.2f} FPS, {total_frames} frames ({duration:.2f}s)")
    print(f"• Đích: {out_file}")
    print("-" * 60)

    # Ưu tiên codec avc1 (H.264) với openh264.dll để Electron/Chromium đọc được
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
    if not writer.isOpened():
        print("⚠️ Không mở được codec 'avc1', thử chuyển sang 'mp4v'...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

    t_start = time.time()
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        writer.write(frame)
        count += 1
        if count % 200 == 0 or count == total_frames:
            percent = (count / total_frames) * 100 if total_frames > 0 else 0
            print(f"  ⏳ Đang chuyển đổi: {count}/{total_frames} frames ({percent:.1f}%)...")

    writer.release()
    cap.release()

    dt = time.time() - t_start
    speed_fps = count / dt if dt > 0 else 0
    print("-" * 60)
    print(f"✅ Hoàn tất chuyển đổi trong {dt:.2f}s (Tốc độ: {speed_fps:.1f} FPS)!")
    print(f"📂 File MP4 đã sẵn sàng: {out_file}")
    print("👉 Bây giờ bạn có thể CLICK VÀO FILE .mp4 trên cây thư mục Antigravity để xem trực tiếp!")
    print("=" * 60)
    return out_file

def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, args.video) if not os.path.isabs(args.video) else args.video

    if not os.path.exists(video_path):
        fallback_path = os.path.join(base_dir, "data", os.path.basename(video_path))
        if os.path.exists(fallback_path):
            video_path = fallback_path
        else:
            fallback_video_dir = os.path.join(base_dir, "data", "videos", os.path.basename(video_path))
            if os.path.exists(fallback_video_dir):
                video_path = fallback_video_dir
            elif video_path.lower().endswith(".avi"):
                mp4_name = os.path.splitext(os.path.basename(video_path))[0] + ".mp4"
                mp4_candidate = os.path.join(base_dir, "data", "videos", mp4_name)
                if os.path.exists(mp4_candidate):
                    video_path = mp4_candidate
                else:
                    print(f"❌ [ERROR] Không tìm thấy file video: {video_path}")
                    return
            else:
                print(f"❌ [ERROR] Không tìm thấy file video: {video_path}")
                return

    # Nếu truyền cờ --convert thì thực hiện chuyển đổi sang MP4
    if args.convert:
        out_mp4 = convert_to_mp4(video_path, args.output_mp4)
        return

    # Nếu không chuyển đổi thì thực hiện chế độ trích xuất ảnh Preview
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ [ERROR] Không thể mở video bằng OpenCV: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Xác định khoảng start/end
    start_s = max(0.0, args.start_sec) if args.start_sec is not None else 0.0
    end_s = min(duration_sec, args.end_sec) if args.end_sec is not None else duration_sec

    start_frame = int(start_s * fps)
    end_frame = min(total_frames - 1, int(end_s * fps))

    if args.output_dir:
        out_dir = os.path.join(base_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    else:
        suffix = f"_{start_s:.0f}s_{end_s:.0f}s" if (args.start_sec is not None or args.end_sec is not None) else ""
        out_dir = os.path.join(base_dir, "data", "preview", f"{video_name}{suffix}")

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"🎬 PREVIEW VIDEO: {os.path.basename(video_path)}")
    print("=" * 60)
    print(f"• Tổng thời lượng: {duration_sec:.2f}s ({total_frames} frames, {fps:.2f} FPS)")
    print(f"• Đoạn trích xuất: {start_s:.2f}s ➔ {end_s:.2f}s (tổng {end_s - start_s:.2f}s)")
    print(f"• Thư mục lưu ảnh: {out_dir}")
    print("-" * 60)

    # Xác định các mốc frame cần lấy
    if args.interval and args.interval > 0:
        step_frames = int(args.interval * fps)
        target_indices = list(range(start_frame, end_frame + 1, max(1, step_frames)))
    else:
        n = max(2, args.num_frames)
        step = (end_frame - start_frame) / (n - 1) if n > 1 else (end_frame - start_frame)
        target_indices = [int(start_frame + i * step) for i in range(n)]

    saved_files = []
    for idx, f_idx in enumerate(target_indices, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        
        timestamp_sec = f_idx / fps
        fn = f"preview_{idx:02d}_{timestamp_sec:.2f}s.jpg"
        out_path = os.path.join(out_dir, fn)
        cv2.imwrite(out_path, frame)
        saved_files.append((fn, timestamp_sec))
        print(f"  [{idx:02d}/{len(target_indices):02d}] Đã lưu: {fn} (mốc {timestamp_sec:.2f}s)")

    cap.release()
    print("=" * 60)
    print(f"✅ Hoàn tất trích xuất {len(saved_files)} ảnh preview!")
    print(f"📂 Thư mục ảnh: {out_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
