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

---

## Ngày: 2026-09-25

- [x] **Task 1: Benchmark VLM trên Google Colab với Payload chuẩn từ Pipeline YOLOE (Video 1 & Video 7)**
  - Tải và đồng bộ bộ 8 frames sạch cùng payload `video1_yoloe_payload.json` và `video7_yoloe_payload.json` lên Colab.
  - Chạy suy luận với `Qwen2.5-VL-3B-Instruct` đối chứng tính nhất quán và độ chính xác.
  - Phát hiện và giải quyết triệt để hiện tượng Ảo giác thế chỗ thực thể (Entity Substitution Hallucination) ở Video 1 bằng Double-Lock Prompt Guardrails (Unmarked Entity Exclusion & Domain Constraints).
  - Kiểm chứng thành công: Video 1 đạt chuẩn `touch` đối xứng 100%, Video 7 đạt chuẩn tương tác tay `hold` 100% (F1 = 1.0 trên cả 2 video).
  - Đo đạc chi tiết tài nguyên token: Video 1 (4,667 tokens), Video 7 (5,040 tokens).

- [ ] **Task 2: Mở rộng kiểm thử Pipeline YOLOE sang Video mới (Kiểm chứng tính tổng quát - Generalization)**
  - Lựa chọn thêm một video mới trong tập dữ liệu (ví dụ `video10.mp4` hoặc video có tương tác người - người / người - vật).
  - Chạy thực nghiệm `pipeline_yoloe.py` với cấu hình chuẩn (60 class, luồng xuôi, bộ lọc rác nền tĩnh `displacement >= 20px`).
  - Đánh giá khả năng tổng quát hóa của pipeline mới trên video chưa từng qua tinh chỉnh (zero manual tuning).

- [ ] **Task 3: Nghiên cứu & Thiết kế giải pháp Gom cụm tương tác (Spatial Clustering & ROI Zoom Crop)**
  - Nghiên cứu ý tưởng định hướng của Mentor: Gom nhóm các bounding box gần nhau trên frame bằng thuật toán không gian (DBSCAN / Scikit-learn hoặc khoảng cách Euclide).
  - Xây dựng thuật toán tính toán hộp bao quanh cụm (Union Bounding Box) và cơ chế cắt ảnh phóng to vùng tương tác (ROI Crop).
  - Đánh giá ưu/nhược điểm của ROI Crop: Giúp VLM nhìn rõ vật thể nhỏ, khử nhiễu người ở xa; đồng thời ghi nhận thách thức duy trì cụm xuyên suốt nhiều frame theo thời gian.
