# Danh Sách Nhiệm Vụ (Task Checklist)

Tài liệu theo dõi tiến độ các đầu việc hàng ngày theo chỉ đạo và định hướng của Mentor.

---

## Ngày: 2026-09-24

- [x] **Task 1: Làm sạch Prompt hệ thống & Trả nghĩa chuẩn cho nhãn `get_off`**
  - Xóa bỏ hoàn toàn việc ép nghĩa nhãn `get_off` cho việc buông bỏ/rời xa đồ vật; quy định `get_off` chỉ áp dụng chuẩn ngữ nghĩa cho phương tiện giao thông hoặc thú cưỡi.
  - Cập nhật câu prompt trung tính, đồng bộ lại các tệp cấu hình payload (`video1_payload.json`, `video7_payload.json`).

- [x] **Task 2: Tích hợp bộ đo lường tài nguyên Token trên Colab**
  - Bổ sung đoạn mã đếm định lượng Input Tokens (Text Prompt + 8 Visual Frames) và Output Tokens (JSON sinh ra) vào file notebook Colab.
  - Hiển thị trực quan thống kê tài nguyên tính toán sau mỗi lượt chạy suy luận.

- [x] **Task 3: Chạy Benchmark mô hình Qwen2.5-VL-3B trên Video 1 & Video 7 với Prompt sạch**
  - Chạy suy luận trên Google Colab với mô hình chủ lực `Qwen2.5-VL-3B-Instruct`.
  - Kiểm tra độ chính xác quan hệ và hoàn thành đối chứng Benchmark Baseline đạt F1 = 1.0 trên cả 2 video:
    - **Video 1**: Input: 4,384 tokens, Output: 95 tokens, Total: 4,479 tokens. Dự đoán chuẩn: `[1] touch [2]`, `[2] touch [1]`. Loại bỏ 100% nhãn rác túi xách ở sàn.
    - **Video 7**: Input: 4,788 tokens, Output: 58 tokens, Total: 4,846 tokens. Dự đoán chuẩn: `[1] carry [2]`. Bắt trọn hành vi mang xách qua thời gian.
  - Xác lập kết quả baseline đối chứng khoa học, sẵn sàng chuyển tiếp sang nâng cấp pipeline detector mới.

- [x] **Task 4: Xây dựng pipeline detector mới độc lập (YOLOE-26m) & Chạy luồng xuôi thuần túy** [Completed]
  - Bảo toàn nguyên vẹn file pipeline.py cũ làm mốc đối chứng; tạo file pipeline mới độc lập (pipeline_yoloe.py).
  - Tích hợp mô hình YOLOE-26m với danh sách 60 class mục tiêu của đề tài (configs/s_objects.json).
  - Loại bỏ hoàn toàn thuật toán dò ngược (Backward Spatio-Temporal Association), chỉ chạy tracking xuôi tự nhiên một chiều kết hợp ByteTrack theo hướng dẫn của Mentor.
  - Tự động loại bỏ Inactive Background Clutter (vật thể tĩnh nền không dịch chuyển <20px) và vật kiến trúc phòng (STATIC_FIXTURE_CLASSES).
  - Thực thi kiểm thử thành công trên Video 7 (114s - 130s): bám bắt chính xác [1] person và [2] handbag khi người mang túi vào phòng và đặt lên bàn (hits=416, mean_conf=0.79, displacement=417.0px).
  - Kết xuất tách biệt hoàn toàn: data/video_processed/video7_yoloe_annotated.mp4, data/frames/video7_yoloe/, data/payloads/video7_yoloe_payload.json.
