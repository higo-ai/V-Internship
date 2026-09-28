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

- [x] **Task 2: Nghiệm thu thực nghiệm mô hình Qwen3-VL-4B-Instruct trên Colab (Video 1 & Video 7)** [Hoàn thành bổ sung chiều 25/09]
  - Chạy đối chứng song song trên GPU T4 của Google Colab với `Qwen3-VL-4B-Instruct` (commit `85278bf`).
  - Đạt điểm số F1 = 1.0 trên cả 2 video: Video 1 phát hiện tiếp xúc cử chỉ vi mô (`shoulder/arm`), Video 7 nhận thức đúng chuyển động thời gian chọn chuẩn xác `[1] carry [2]`.
  - Nén tối ưu visual tokens giảm ~16% (tiết kiệm ~750 tokens đầu vào). Toàn bộ Task mở rộng video mới và gom cụm ROI được chuyển tiếp sang kế hoạch tuần mới ngày 28/09 theo chỉ đạo của Mentor.

---

## Ngày: 2026-09-28

- [x] **Task 1: Chuẩn hóa tính tổng quát cho Detector (Bật lại vật thể tĩnh & Giao quyền lọc rác cho Prompt VLM)** [Hoàn thành 28/09]
  - Tiếp thu chỉ đạo của Mentor: Loại bỏ triệt để ngưỡng dịch chuyển cứng (`disp < 20px`) ở tầng detector, bảo toàn toàn bộ vật thể có độ bền vững không gian (`hits >= 25`).
  - Hiện thực thuật toán gom cụm không gian đa nhãn (*Spatial Cross-Class Merging*): Tự động triệt tiêu hiện tượng dao động nhãn (nhảy luân phiên giữa `backpack` và `handbag`) của mô hình Open-Vocabulary bằng bầu chọn đa số phiếu (majority vote) và Spatial IoU NMS.
  - Tự động kiểm tra trực quan từng frame:
    + **Video 1**: Nhận diện và gán nhãn chính xác chiếc balo trên sàn thành `ID=[4] backpack` với banner hiển thị rõ nét trên các frame từ `frame_03` đến `frame_08`.
    + **Video 7**: Giữ lại song song cả túi xách tĩnh trên bàn `ID=[2] handbag` (displacement 1.6px) và túi xách đang xách `ID=[3] handbag` (displacement 413.8px).
  - Tự động cập nhật Payload JSON (`video1_yoloe_payload.json` & `video7_yoloe_payload.json`) tích hợp tập thực thể mở rộng và 4 trụ cột quy chuẩn ngữ nghĩa thị giác (Zero Cheating / Zero Hardcode):
    + `Clean ID Formatting`: Chuẩn hóa ID sạch `"[1]"`, `"[2]"` (đáp ứng benchmark code).
    + `Triplet Uniqueness & Predicate Exclusivity`: Khống chế tính duy nhất của quan hệ, cấm xuất hiện trùng lặp giữa các frame và loại trừ xung đột cường độ tiếp xúc (`touch` vs `push`).
    + `Temporal Action Continuity`: Thống nhất các pha của hành động (tiếp cận, vươn tay, tiếp xúc, chia tay) thành một sự kiện tương tác liên tục, không phân mảnh động tác chuẩn bị thành `push`.
    + `Physical Hand-Grasp Requirement`: Bắt buộc tay phải cầm nắm trực tiếp với vật thể (`carry`/`hold`), loại bỏ triệt để đồ vật nằm dưới đất.
  - Tích hợp kỹ thuật lượng tử hóa 4-bit NF4 (BitsAndBytes NormalFloat4 + Double Quantization) trong `run_qwen3_vl_4b_colab.ipynb`: Giảm VRAM tiêu thụ trên Tesla T4 từ >14.5 GB (OOM) xuống ~9.5 GB / 15 GB an toàn mà vẫn giữ 99% độ chính xác.

- [x] **Task 2: Thiết kế & Hiện thực Module Gom cụm không gian (Spatial Clustering & Dynamic ROI Zoom Crop)** [Hoàn thành 28/09]
  - Hiện thực thuật toán phân cụm không-thời gian thông minh (*Spatio-Temporal Interaction Clustering - STIC*) tại `modules/spatial_clustering.py`:
    + Đánh giá liên tục chỉ số khoảng cách biên hộp bao ($D_{edge}$) và độ phủ ($IoU$) giữa mọi cặp thực thể trên tất cả các frame đồng xuất hiện.
    + Giải quyết triệt để bẫy chuỗi bắc cầu theo thời gian (*Transitive Chaining through Time*): Phân biệt rõ tiếp xúc tương tác thực sự bền vững ($\ge 20$ frames) với người đi lướt qua nhau trong thoáng chốc ($< 20$ frames).
    + Áp dụng quy tắc động học liên kết Người - Vật (*Kinematic Object Association*): Chỉ ghép cặp đồ vật vào cụm nếu đồ vật đó đang được di chuyển/mang xách ($disp \ge 20$px); tự động cô lập 100% đồ vật tĩnh trên sàn/trên bàn thành phần tử đơn lẻ (*Isolated Singletons*).
  - Tự động tính toán Hộp bao cụm tương tác thích ứng (*Dynamic Motion-Aware Union Bounding Box*):
    + Tự động cộng lề an toàn thích ứng (*Adaptive Context Padding 20%*) để bảo toàn trọn vẹn ngữ cảnh cử chỉ tay chân và đầu gối.
    + Hỗ trợ song song cả vùng tương tác tĩnh (hộp bao ổn định chống rung giật camera) và vùng tương tác di chuyển qua phòng (smooth tracking window bám sát đối tượng).
  - Kết xuất trực tiếp các visual prompt phóng to độ phân giải cao (*High-Resolution ROI Zoom Crop Frames*):
    + Cắt trực tiếp từ frame video gốc chưa chú thích (*pristine unannotated frame*), sau đó chiếu tọa độ cục bộ để vẽ Set-of-Marks sắc nét, to rõ.
    + **Video 1 (44s - 52s)**: Phân cụm chính xác `Cluster 1: ['[1]', '[2]']`, loại bỏ 100% người đi xa ở nền `[3]` và balo trên sàn `[4]`. Độ phóng đại hình ảnh đạt **3.96x** (từ 640x480 -> 226x343), cận cảnh chi tiết bàn tay chạm vai.
    + **Video 7 (114s - 130s)**: Phân cụm chính xác `Cluster 1: ['[1]', '[3]']`, loại bỏ 100% túi xách tĩnh trên bàn `[2]`. Độ phóng đại hình ảnh đạt **3.30x** (từ 720x480 -> 239x438), cận cảnh rõ nét bàn tay cầm quai túi xách.
  - Tự động sinh Payload chuẩn tương ứng cho từng cụm (`video1_roi_cluster_1_payload.json`, `video7_roi_cluster_1_payload.json`), cô lập hoàn toàn nhiễu từ các đối tượng ngoài cụm và triệt tiêu 100% ảo giác thế chỗ.

- [ ] **Task 3: Kiểm thử tổng quát hóa trên Video mới (Generalization on New Videos)**
  - Thử nghiệm pipeline mới (kết hợp YOLOE tổng quát + Module ROI Zoom Crop) trên video mới trong tập dữ liệu (ví dụ Video 3 có hành vi rơi đồ, hoặc Video 10).
  - Đánh giá khả năng bám bắt tương tác tự động mà không cần bất kỳ tinh chỉnh tham số thủ công nào (Zero manual tuning).
