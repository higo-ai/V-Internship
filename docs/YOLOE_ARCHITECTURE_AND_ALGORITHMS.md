# KIẾN TRÚC VÀ THUẬT TOÁN HỆ THỐNG YOLOE (VIDVRD PIPELINE)

- **Mã nguồn thực thi chính:** [pipeline_yoloe.py](../pipeline_yoloe.py)


---

## 1. SƠ ĐỒ KIẾN TRÚC HỆ THỐNG (SYSTEM ARCHITECTURE)

```
[Video Giám sát Đầu vào]
          │
          ▼
[Mô hình YOLOE-26m Đơn nhất: 60 Classes Taxonomy, Single Forward Pass trên CUDA/CPU]
(pipeline_yoloe.py: L241-L251)
          │
          ├───► Stream A: Chủ thể Người (Human Subjects)
          │         │   (pipeline_yoloe.py: L263-L272)
          │         ├─► Lọc đốm nhiễu vi mô (Physical Plausibility Filter: L268-L272)
          │         ├─► Khử trùng lặp thân người (Anatomical Torso Safeguard: L167-L227)
          │         └─► Khâu nối ID không trùng lặp (Smart Zero-Overlap Stitching: L303-L345)
          │
          └───► Stream B: Vật thể Tương tác Di động (Portable Interactive Objects)
                    │   (pipeline_yoloe.py: L274-L290)
                    ├─► Lọc vật thể tĩnh gắn tường/sàn (Static Fixtures Filter: L274)
                    ├─► Ghép nối chuỗi vật thể (Object Tracklet Association & Stitching: L395-L475)
                    ├─► Lọc vật thể bất động (Active Entity Spatio-Temporal Filter: L486-L513)
                    └─► Duy trì vị trí đồ đặt xuống (Stationary Forward-Fill: L515-L538)
          │
          ▼
[Lấy mẫu Khung hình Thích ứng Độ rõ Thị giác: Chọn 8 Frames cho VLM]
(pipeline_yoloe.py: L622-L680)
          │
          ▼
[Sinh Payload Prompt VLM: Từ điển đóng 26 Predicates, Không rò rỉ đáp án]
(pipeline_yoloe.py: L704-L790)
          │
          ▼
[Suy luận VLM: Qwen2.5-VL-3B-Instruct / Qwen3-VL-4B -> Xuất Quan hệ Triplets JSON]
```

---

## 2. LÀM RÕ CƠ CHẾ PHÂN TÁCH STREAM A VÀ STREAM B

Trong [pipeline_yoloe.py: dòng 257-290](../pipeline_yoloe.py#L257), sau khi mô hình YOLOE-26m thực hiện một lượt suy luận duy nhất (Single Forward Pass), kết quả phát hiện được phân tách thành 2 luồng xử lý riêng biệt:

### Stream A: Luồng Chủ thể Người (Human Subjects)
- **Vị trí code:** [pipeline_yoloe.py: dòng 263-272](../pipeline_yoloe.py#L263)
- **Điều kiện lọc:** `c_name in HUMAN_CLASSES` (gồm `person`, `child`).
- **Cơ chế theo dõi:** Sử dụng trực tiếp `Track ID` do ByteTrack cấp phát (`results.boxes.id`), sau đó đưa qua các thuật toán hình học cơ thể để xử lý các vấn đề đặc thù của người (xoay người đè 2 box, đi qua vùng khuất bị nhảy ID).
- **Lý do:** Con người là chủ thể chính thực hiện hành động, có đặc trưng chuyển động liên tục, cần bộ theo dõi chuyển động chuyên biệt như ByteTrack để duy trì danh tính ổn định.

### Stream B: Luồng Vật thể Tương tác Di động (Portable Interactive Objects)
- **Vị trí code:** [pipeline_yoloe.py: dòng 274-290](../pipeline_yoloe.py#L274)
- **Điều kiện lọc:** `c_name not in STATIC_FIXTURE_CLASSES` và `conf >= 0.40`. Tự động loại bỏ đồ nội thất cố định gắn tường/sàn (như bàn, ghế, tủ, bồn rửa, màn hình...).
- **Cơ chế theo dõi:** Không phụ thuộc vào ByteTrack ID của vật thể. Thay vào đó, vật thể được gom cụm và liên kết không-thời gian bằng thuật toán **Forward Object Tracklet Association & Stitching** ([pipeline_yoloe.py: dòng 395-475](../pipeline_yoloe.py#L395)).
- **Lý do kỹ thuật:** Khi đồ vật (túi xách, balo, điện thoại) được con người cầm, xách hoặc đặt xuống, chúng thường xuyên bị bàn tay hoặc cơ thể che khuất một phần. ByteTrack (vốn dựa trên mô hình chuyển động Kalman Filter dành cho người/xe) sẽ dễ bị mất dấu và liên tục đổi ID mới cho cùng một chiếc túi. Thuật toán liên kết không gian - thời gian riêng cho Stream B giải quyết triệt để hiện tượng này.

---

## 3. CHI TIẾT CÁC THUẬT TOÁN TRONG PIPELINE

### 1. Thuật toán Khử Trùng Lặp Thân Người (Anatomical Torso Safeguard)
- **Vị trí code:** [pipeline_yoloe.py: dòng 167-227](../pipeline_yoloe.py#L167)
- **Vấn đề thực tế:** Khi đối tượng xoay người, YOLOE thường dự đoán đồng thời 2 hộp: 1 hộp bao toàn thân và 1 hộp phụ bao nửa thân trên (torso). Nếu dùng NMS thông thường (xóa hộp con lọt >= 70%), hệ thống sẽ xóa nhầm người khi 2 người ôm nhau, người lớn bế trẻ em, hoặc 2 người đi lướt sát qua nhau.
- **Cơ chế hoạt động:** Chỉ loại bỏ hộp nhỏ nếu thỏa mãn đồng thời 5 điều kiện hình thái học:
  1. Cùng lớp đối tượng (`person` với `person`; không bao giờ xóa `child` trong `person`).
  2. Chung vị trí đỉnh đầu: `abs(y1_small - y1_large) <= 0.15 * h_large`.
  3. Chung trục sống lưng dọc: `abs(cx_small - cx_large) <= 0.25 * w_large`.
  4. Chiều cao bị cụt nửa dưới: `h_small <= 0.70 * h_large`.
  5. Độ bao hàm diện tích: `Intersection / Area_small >= 0.70`.
- **Lý do thiết kế:** Đảm bảo triệt tiêu hiện tượng đè 2 ID lên cùng một người nhưng không gây ra dương tính giả trong các tình huống tiếp xúc thân thể.

---

### 2. Thuật toán Lọc Đốm Nhiễu Vi mô (Physical Surveillance Plausibility Filter)
- **Vị trí code:** [pipeline_yoloe.py: dòng 268-272](../pipeline_yoloe.py#L268)
- **Vấn đề thực tế:** Trong môi trường giám sát, các phản xạ ánh sáng hoặc đốm hoa văn xa (ví dụ đốm phản quang trên lan can Video 1 kích thước 8 x 18 pixel) có thể bị mô hình nhận nhầm là người với độ tin cậy thấp.
- **Cơ chế hoạt động:** Loại bỏ các phát hiện người không thỏa mãn kích thước vật lý tối thiểu:
  - Diện tích hộp: `Area >= 400 pixel^2`.
  - Kích thước cạnh lớn nhất: `max(width, height) >= 35 pixel`.
  - Tỷ lệ khung dọc cơ thể: `height / width >= 0.4`.
- **Lý do thiết kế:** Trong camera an ninh góc rộng, một đối tượng người thực tế ở khoảng cách xa vẫn có chiều cao tối thiểu 40-60 pixel. Ràng buộc hình học loại bỏ triệt để nhiễu cảm biến mà không ảnh hưởng đến người thật.

---

### 3. Thuật toán Khâu Nối Quỹ Đạo Không Trùng Lặp (Smart Zero-Overlap Tracklet Stitching)
- **Vị trí code:** [pipeline_yoloe.py: dòng 303-345](../pipeline_yoloe.py#L303)
- **Vấn đề thực tế:** Khi đối tượng đi qua vùng che khuất tạm thời hoặc xoay lưng, ByteTrack có thể mất dấu vài khung hình rồi cấp một ID mới khi đối tượng xuất hiện lại (hiện tượng ID Switch).
- **Cơ chế hoạt động:** Hợp nhất 2 tracklet thành một ID thống nhất khi thỏa mãn các điều kiện:
  1. Không trùng lặp khung hình: Hai tracklet tuyệt đối không xuất hiện cùng nhau trong bất kỳ frame nào (`len(Frames_A ∩ Frames_B) == 0`).
  2. Khoảng cách không gian: Khoảng cách tâm giữa điểm kết thúc tracklet trước và điểm bắt đầu tracklet sau `< 200 pixel`.
  3. Khoảng cách thời gian: Thời gian gián đoạn giữa 2 tracklet `<= 90 frames` (~3 giây).
- **Lý do thiết kế:** Tránh việc một cá nhân bị chia thành nhiều ID khác nhau, đồng thời điều kiện `Zero Overlap` bảo đảm 2 người đi song song hoặc gặp nhau không bao giờ bị gộp nhầm.

---

### 4. Thuật toán Liên Kết & Khâu Nối Chuỗi Vật Thể (Object Tracklet Association & Stitching)
- **Vị trí code:** [pipeline_yoloe.py: dòng 395-475](../pipeline_yoloe.py#L395)
- **Vấn đề thực tế:** Đồ vật di động thường bị tay người hoặc vật cản che khuất ngắt quãng, tạo ra các chuỗi phát hiện rời rạc ngắn (tracklets).
- **Cơ chế hoạt động:**
  - Giai đoạn 1: Liên kết khung hình liên tiếp dựa trên khoảng cách tâm `< 100 pixel` hoặc `IoU > 0.15` trong phạm vi thời gian `<= 30 frames`.
  - Giai đoạn 2: Khâu nối các chuỗi cùng lớp đối tượng có khoảng cách không gian `< 220 pixel` và gián đoạn `<= 60 frames` (ví dụ từ trạng thái người cầm đi sang trạng thái đặt xuống bàn).
  - Giai đoạn 3: Nội suy tuyến tính (Linear Interpolation) các khung hình bị mất dấu ngắn `<= 45 frames` bên trong chuỗi.
- **Lý do thiết kế:** Đảm bảo vật thể duy trì một mã định danh duy nhất xuyên suốt quá trình tương tác (cầm -> mang đi -> đặt xuống).

---

### 5. Bộ Lọc Vật Thể Động Theo Không-Thời Gian (Active Entity Spatio-Temporal Filter)
- **Vị trí code:** [pipeline_yoloe.py: dòng 486-513](../pipeline_yoloe.py#L486)
- **Vấn đề thực tế:** Trong cảnh giám sát luôn tồn tại các vật thể nằm bất động (ví dụ chiếc balo bỏ quên trên sàn Video 1 không ai đụng tới). Nếu đánh dấu toàn bộ, VLM sẽ bị nhiễu ngữ cảnh và sinh ra quan hệ ảo (False Positive), làm giảm điểm Precision và mAP của benchmark.
- **Cơ chế hoạt động:** Sử dụng hàm toán học Peak-to-Peak (`np.ptp`) trên toàn bộ chuỗi tọa độ của vật thể:  
  `total_displacement = sqrt((ptp(center_x))^2 + (ptp(center_y))^2)`
  - Nếu `total_displacement < 20 pixel`: Xác định là vật thể bất động tĩnh (Inactive Clutter) -> Ẩn đi.
  - Nếu `total_displacement >= 20 pixel`: Xác định là vật thể tương tác chủ động (Active Object) -> Giữ lại và vẽ hộp nhận diện từ khung hình đầu tiên.
- **Lý do thiết kế:** Hàm `np.ptp` lấy giá trị lớn nhất trừ giá trị nhỏ nhất của toàn bộ video. Dù vật thể chỉ bị di chuyển ở vài giây cuối (ví dụ kịch bản bị nhặt hoặc trộm đi), độ dịch chuyển tổng vẫn vượt ngưỡng 20 pixel và được giữ lại nguyên vẹn.

---

### 6. Cơ Chế Giữ Vị Trí Vật Thể Đứng Yên (Stationary Forward-Fill)
- **Vị trí code:** [pipeline_yoloe.py: dòng 515-538](../pipeline_yoloe.py#L515)
- **Vấn đề thực tế:** Khi người mang vật thể vào phòng rồi đặt lên bàn (như Video 7) và rời đi, vật thể trở thành tĩnh và không còn người tương tác trực tiếp, khiến bộ phát hiện dễ bị trồi sụt hoặc mất box ở các khung hình sau.
- **Cơ chế hoạt động:** Khi một vật thể đã xác nhận là Active Object chuyển sang trạng thái dừng lại ở cuối quỹ đạo (độ dịch chuyển đuôi `tail_disp < 15 pixel` và không nằm sát biên ảnh) -> Tự động giữ nguyên tọa độ hộp phát hiện cuối cùng cho tới khung hình kết thúc của đoạn video.
- **Lý do thiết kế:** Giúp vật thể duy trì sự hiện diện liên tục trên bề mặt sau khi kết thúc hành vi đặt đồ, cung cấp ngữ cảnh trực quan đầy đủ cho VLM nhận định quan hệ kết thúc (`get_off` / `hold`).

---

### 7. Bộ Lấy Mẫu Khung Hình Thích Ứng Độ Rõ Thị Giác (Visibility-Aware Keyframe Sampling)
- **Vị trí code:** [pipeline_yoloe.py: dòng 622-680](../pipeline_yoloe.py#L622)
- **Vấn đề thực tế:** Phương pháp lấy mẫu đều theo khoảng cách toán học cố định dễ rơi vào các khung hình bị nhòe chuyển động (motion blur) hoặc thời điểm đối tượng bị che khuất tạm thời.
- **Cơ chế hoạt động:** Xác định chỉ số khung hình lý thuyết theo bước nhảy đều, sau đó quét trong cửa sổ cục bộ `+/- 12 frames` (~0.4 giây) xung quanh để chọn khung hình có điểm phạt thấp nhất:
  - Phạt nặng nếu thiếu các đối tượng đang trong vòng đời hoạt động (`missing_count * 1000`).
  - Phạt nhẹ theo độ lệch khoảng cách so với nhịp đều lý thuyết.
- **Lý do thiết kế:** Tối đa hóa số lượng thực thể hiển thị rõ nét trên 8 khung hình gửi cho VLM, đồng thời bảo đảm phân bố thời gian đồng đều của video.

---

### 8. Định Dạng Prompt VLM Vô Trùng & Từ Điển Đóng (Sterile 4-Pillar Prompt)
- **Vị trí code:** [pipeline_yoloe.py: dòng 704-790](../pipeline_yoloe.py#L704)
- **Cơ chế hoạt động:**
  1. Loại bỏ hoàn toàn trường `temporal_summary` để tránh rò rỉ thông tin trước hoặc bẫy ảo giác cho VLM.
  2. Giới hạn chặt chẽ không gian nhãn trong 60 lớp đối tượng ([configs/s_objects.json](../configs/s_objects.json)) và 26 lớp quan hệ chuẩn ([configs/relations.json](../configs/relations.json)).
  3. Cung cấp quy tắc phân định rõ ràng giữa tiếp xúc cơ thể (`touch`), mang vác di chuyển (`carry`), và cầm giữ tĩnh (`hold`).
- **Lý do thiết kế:** Chuẩn hóa đầu ra JSON theo cấu trúc nghiêm ngặt `{"triplets": [{"subject": "[ID]", "relation": "<verb>", "object": "[ID]", "reason": "..."}]}`, tương thích trực tiếp với pipeline đánh giá tự động.

---

## 4. BẢNG ĐỐI CHIẾU XỬ LÝ 3 TÌNH HUỐNG THỰC TẾ

| Tình huống thực tế | Phản ứng của hệ thống YOLOE | Kết quả mô hình VLM |
| :--- | :--- | :--- |
| **Kịch bản 1: Đặt đồ xuống (Video 7)**<br>Người xách túi vào phòng, đặt lên bàn rồi đi ra. | Độ dịch chuyển = 413.8 px (> 20 px) -> Xác nhận Active Object.<br>Kích hoạt Stationary Forward-Fill duy trì box túi trên bàn. | VLM nhận diện chính xác: `[1] carry [2]` và `[1] get_off [2]`. (Đúng 100% Ground Truth). |
| **Kịch bản 2: Lấy đồ đi / Trộm đồ (Giả định)**<br>Túi nằm im trên sàn, người khác đi qua nhặt và mang đi. | Độ dịch chuyển >= 150 px (> 20 px) -> Xác nhận Active Object.<br>Vẽ box túi từ khung hình đầu tiên khi còn trên sàn đến khi được mang đi. | VLM quan sát đầy đủ tiến trình: Túi trên sàn -> Tay chạm vào -> Mang đi -> Xuất quan hệ `[3] grab [2]`, `[3] carry [2]`. |
| **Kịch bản 3: Đồ vật bỏ quên không tương tác (Video 1)**<br>Balo nằm im trên sàn suốt 8 giây, mọi người chỉ đi ngang qua. | Độ dịch chuyển = 5.7 px (< 20 px) -> Phân loại là Clutter và ẩn đi. | Bảo vệ VLM khỏi đoán bừa quan hệ giả (`[1] hold [balo]`), tránh bị trừ điểm False Positive. |

---

## 5. TÓM TẮT 3 Ý TRỌNG TÂM

1. **Về Mô hình:** *"Hệ thống sử dụng mô hình đơn nhất YOLOE-26m chạy Single Forward Pass trên GPU để nhận diện đồng thời cả Người và 60 lớp Vật thể theo đúng taxonomy chuẩn của dự án."*
2. **Về Cơ chế Lọc:** *"Hai luồng Stream A (Người) và Stream B (Vật thể) được xử lý riêng biệt: Stream A áp dụng Anatomical Safeguard và Zero-Overlap Stitching để ổn định ID người; Stream B áp dụng Active Entity Filter dựa trên biến thiên quỹ đạo không-thời gian (Peak-to-Peak >= 20px) nhằm loại bỏ vật thể chết và bảo vệ VLM khỏi ảo giác dương tính giả."*
3. **Về Hậu xử lý & VLM:** *"Pipeline kết hợp Stationary Forward-Fill để duy trì vật thể sau khi đặt xuống bàn và Visibility-Aware Sampling để tự động chọn 8 khung hình có độ hiển thị thực thể rõ ràng nhất nạp vào VLM với Prompt vô trùng."*
