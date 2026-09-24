# Worklog — Bùi Tiến Phát

> Nhật ký thực tập tại VinFast - Người hướng dẫn (Mentor): Lường Mạnh Tú.
> Thời gian thực tập: 14/09/2026 - xx/10/2026.

---

## 2026-09-17

| Task | Khó khăn | Giải pháp | Kết quả | Status |
|---|---|---|---|---|
| **Thử nghiệm Qwen3.5-2B & sinh mô tả giải thích**:<br>Nghiên cứu nâng cấp lên Qwen3.5-2B và thêm prompt yêu cầu sinh lý do (reasoning) theo định hướng của Mentor. | Chạy thử trên Colab bị nghẽn tốc độ (~2 phút/lần do thiếu tối ưu kernel) và gặp hallucination nặng (đọc nhầm watermark OCR, bịa hành động nhặt túi). | **Chuyển hướng Qwen2.5-VL-3B & Tinh chỉnh pipeline**:<br>- Xóa watermark video tránh nhiễu OCR<br>- Đánh dấu Set-of-Marks [ID] tự động<br>- Gom cụm vận tốc v ~ 0 bắt vật thể tĩnh<br>- Ép Chain-of-Thought qua temporal_summary và ràng buộc 26 nhãn VidVRD | Model chạy nhanh, đạt độ chính xác cao: bắt đúng 100% [1] touch [2] và [2] get_off [4], loại bỏ hoàn toàn ảo giác. | ✅ Done |

**Tổng kết ngày:** Hôm nay mình tập trung nghiên cứu thử nghiệm model Qwen3.5-2B và xuất mô tả giải thích quan hệ (reasoning) theo định hướng của Mentor, tuy nhiên khi chạy trên Google Colab thì gặp hiện tượng suy luận bị nghẽn rất chậm (do kiến trúc lai mới chưa tương thích tốt thư viện tăng tốc) và kết quả xuất ra bị ảo giác nặng (tự bịa hành động nhặt túi, đọc nhầm watermark góc ảnh thành nhãn đối tượng, và sinh quan hệ ngoài danh mục cho phép). Để khắc phục, mình đã chuyển hướng sang thử nghiệm model thị giác chuyên dụng Qwen2.5-VL-3B-Instruct, đồng thời thiết kế lại toàn bộ pipeline tinh chỉnh: xóa watermark tránh nhiễu OCR, tự động gán nhãn số Set-of-Marks [ID] kết hợp gom cụm vận tốc để bắt vật thể bỏ rơi tĩnh, truyền mốc thời gian động vào từng frame, và ép tư duy chuỗi (Chain-of-Thought) qua tóm tắt diễn tiến thời gian (temporal_summary) để chặn hallucination; kết quả giúp model bắt chính xác 100% quan hệ chạm vai [1] touch [2] và rời bỏ hành lý [2] get_off [4] theo đúng 26 nhãn chuẩn VidVRD. Cuối ngày, mình đã báo cáo tiến độ và thảo luận cùng Mentor: giải trình lý do kiến trúc và ảo giác của 2B so với 3B, làm rõ ngữ nghĩa quy ước của quan hệ "get_off", qua đó Mentor chốt chuẩn hóa biến đầu vào cố định 8 frames và lên kế hoạch chạy benchmark lặp lại 3 lần mỗi con vào ngày (18/09) để đo lường định lượng độ ổn định, tốc độ và tính nhất quán giữa 2 model.

---

## 2026-09-18

| Task | Khó khăn | Giải pháp | Kết quả | Status |
|---|---|---|---|---|
| **Chuẩn hóa 8 frames & chạy benchmark baseline 3x**:<br>Chuẩn hóa mật độ lấy mẫu cố định 8 frames theo yêu cầu của Mentor và chạy benchmark lặp lại 3 lần trên Qwen2.5-VL-3B-Instruct. | Cần đảm bảo việc giảm số frame từ 10 xuống 8 không làm mất thông tin hành vi rời bỏ hành lý và không giảm độ chính xác. | Phân bổ 8 frames đều đặn trên đoạn clip 8s (1 frame/giây); chạy lặp lại 3 lần độc lập trên Colab T4 với greedy decoding để đo tính nhất quán. | Model đạt 100% nhất quán trên cả 3 lần chạy (bắt đúng 100% [1] touch [2] và get_off), thời gian suy luận ổn định ~20s/lần; lưu baseline vào repo. | ✅ Done |
| **Quy hoạch cấu trúc thư mục project**:<br>Sắp xếp lại cây thư mục repo để chuẩn bị mở rộng quy mô đa video. | Các file cấu hình và dữ liệu đang nằm rải rác ở thư mục gốc, dễ gây xung đột khi test nhiều video. | Gom các file taxonomy vào `configs/` (`s_objects.json`, `relations.json`), quy hoạch thư mục `data/frames/video1/` và `data/payloads/`. | Cấu trúc project gọn gàng, rõ ràng, code đọc đường dẫn linh hoạt và sẵn sàng cho việc mở rộng thêm các video mới. | ✅ Done |

**Tổng kết ngày:** Hôm nay mình tập trung chuẩn hóa số lượng frame đầu vào cố định 8 frames theo định hướng của Mentor để tối ưu thời gian suy luận và bộ nhớ. Sau đó, mình tiến hành chạy benchmark lặp lại 3 lần liên tiếp trên Google Colab T4 với model Qwen2.5-VL-3B-Instruct; kết quả đạt độ ổn định tuyệt đối (100% nhất quán trên cả 3 lần, F1 = 1.0, tốc độ ~20s/lần). Đồng thời, mình đã quy hoạch lại toàn bộ cấu trúc thư mục dự án (tách riêng `configs/`, `data/frames/video1/`, `data/payloads/`) giúp mã nguồn ngăn nắp, chuẩn hóa và sẵn sàng mở rộng thử nghiệm trên nhiều video tiếp theo.

---

## 2026-09-21

| Task | Khó khăn | Giải pháp | Kết quả | Status |
|---|---|---|---|---|
| **Xây dựng bộ định nghĩa & gom nhóm 26 quan hệ VidVRD**:<br>Xây dựng file định nghĩa chi tiết và phân nhóm ngữ nghĩa cho 26 quan hệ theo góp ý mở rộng của Mentor. | Ranh giới ngữ nghĩa giữa một số động từ khá hẹp (như touch vs hit, hold vs grab), cần định nghĩa rõ để VLM không bị lúng túng. | Tạo `configs/relations_definitions.json` (định nghĩa cụ thể từng từ) và `configs/relations_grouped.json` (phân loại thành 5 nhóm hành vi lớn). | Hoàn thiện 2 file cấu hình chuẩn hóa ngữ nghĩa cho bộ từ vựng 26 quan hệ của VidVRD, commit lưu trữ mốc 9dfed40. | ✅ Done |
| **Thực nghiệm diện rộng 4 model VLM trên 3 điều kiện prompt**:<br>Thử nghiệm so sánh 4 model (Qwen2.5-VL-3B, Qwen3-VL-4B, Qwen3.5-2B, Qwen3.5-4B) qua 3 biến thể prompt. | Qwen3.5 (2B và 4B) chạy rất chậm trên Colab và sinh ảo giác; việc hoán đổi liên tục 4 model qua 3 điều kiện prompt dễ nhầm lẫn. | Viết và đồng bộ script Colab nạp động từng model và từng payload; ghi chép nhật ký thực nghiệm độc lập cho từng lượt chạy. | Thu thập đầy đủ kết quả thực nghiệm của 4 model trên 3 điều kiện (prompt thô, prompt định nghĩa chi tiết, prompt gom nhóm). | ✅ Done |

**Tổng kết ngày:** Hôm nay mình tập trung vào việc nghiên cứu mở rộng ngữ nghĩa và thử nghiệm đa mô hình theo yêu cầu của Mentor. Mình đã hoàn thành xây dựng 2 bộ cấu hình định nghĩa chi tiết và gom nhóm ngữ nghĩa cho 26 quan hệ VidVRD. Tiếp đó, mình đã chạy thử nghiệm toàn diện 4 mô hình (Qwen2.5-VL-3B, Qwen3-VL-4B, Qwen3.5-2B, Qwen3.5-4B) trên cả 3 điều kiện prompt (prompt thô, định nghĩa từng từ, định nghĩa gom nhóm). Trong ngày, mình cũng đã trao đổi với Mentor về định hướng hạ tầng mạng và chuẩn bị dữ liệu thực nghiệm để lập bảng so sánh năng lực giữa các mô hình.

---

## 2026-09-22

| Task | Khó khăn | Giải pháp | Kết quả | Status |
|---|---|---|---|---|
| **Đánh giá tổng hợp 4 model & chọn lọc mô hình cốt lõi**:<br>Phân tích bảng kết quả so sánh định lượng của 4 model qua 3 điều kiện để quyết định hướng đi tiếp theo. | Qwen3.5-2B/4B bị loại do ảo giác nặng; Qwen3-VL-4B chạy prompt gom nhóm bị nghẽn VRAM (~14.5/15GB) và sinh quan hệ dư thừa/lặp. | Thống nhất loại bỏ Qwen3.5; quyết định quay về giữ nhánh Qwen2.5-VL-3B-Instruct với prompt thô gọn nhẹ, ổn định làm xương sống chính. | Tiết kiệm tài nguyên tính toán, bảo toàn tính ổn định cao và tốc độ suy luận nhanh của pipeline. | ✅ Done |
| **Mở rộng thử nghiệm Prompt thô sang Video mới (`video7.mp4`)**:<br>Kiểm thử khả năng tổng quát của Qwen2.5-VL-3B trên phân cảnh người mang túi xách vào lớp học (1:50 - 2:10). | Cần xác định đoạn cắt tối ưu để không bị quá dài; khi chạy lần 1 model sinh nhãn ngoài từ vựng (`walk`) và sót quan hệ `get_off`. | Tối ưu đoạn cắt vàng 16 giây (1:54 - 2:10, 8 frames); phân tích nguyên nhân do prompt overfit chữ "floor" (video 7 túi để trên bàn) và túi thiếu nhãn lúc di chuyển. | Cắt và băm thành công 8 frame chuẩn; xác định chính xác nguyên nhân gốc để tối ưu lại prompt tổng quát (`surface`) và luật đóng từ vựng. | ✅ Done |

**Tổng kết ngày:** Hôm nay mình đã tổng hợp và phân tích bảng so sánh thực nghiệm của 4 model: chính thức loại bỏ Qwen3.5 (2B/4B) do không đáp ứng được yêu cầu, đồng thời nhận thấy nhánh Qwen3-VL-4B với prompt gom nhóm chiếm dụng gần cạn VRAM Colab (14.5/15GB) và bị spam quan hệ lặp. Do đó, mình quyết định giữ lại nhánh Qwen2.5-VL-3B-Instruct với prompt thô gọn nhẹ làm mô hình chủ lực. Buổi chiều, mình mở rộng kiểm thử mô hình trên video mới (video7.mp4, cắt đoạn vàng 16s từ 1:54 - 2:10 theo gợi ý của Mentor Tú). Khi chạy thử lần đầu trên video 7, model bị rò rỉ nhãn ngoài từ vựng (`walk`) và bỏ sót `get_off`. Mình đã bóc tách nguyên nhân kỹ thuật: do prompt cũ bị thiên kiến chữ "floor" và visual mark của túi bị thiếu ở các frame đầu, từ đó tối ưu lại Prompt hệ thống mang tính tổng quát mọi bề mặt (`surface`) kết hợp luật khóa từ vựng đóng (`STRICT CLOSED VOCABULARY`) để chuẩn bị cho việc hoàn thiện pipeline.

---

## 2026-09-23

| Task | Khó khăn | Giải pháp | Kết quả | Status |
|---|---|---|---|---|
| **Nâng cấp Pipeline thị giác đa cơ chế (Backward Association, Gap Filling, Adaptive Sampling)**:<br>Nâng cấp thuật toán xử lý dữ liệu để giải quyết triệt để các hạn chế thị giác trên `video7.mp4`. | Túi xách mang vào phòng bị thiếu bounding box ở các frame đầu; hiện tượng detector flicker ngắn khiến việc lấy mẫu cách đều dễ rơi trúng frame bị mất dấu người. | - Tích hợp Backward Spatio-Temporal Association dò ngược mark túi xách về tay người mang.<br>- Bổ sung Tracklet Gap Filling vá khoảng trống <= 5 frames.<br>- Tích hợp Constrained Lifespan-Aware Adaptive Sampling lấy mẫu thông minh (+-3 frames) ưu tiên frame đủ thực thể nhưng vẫn tôn trọng vòng đời khi người đã rời phòng. | Băm 8 frames sạch; chạy thực nghiệm lần 1 trên Video 7 đạt F1 = 1.0 (bắt đúng `carry` và `get_off`, triệt tiêu từ `walk`); commit lưu mốc `a5faea2`. | ✅ Done |
| **Kiểm chứng chéo Video 1, tối ưu Prompt trung tính & Căn chỉnh phân loại (Taxonomy Grounding)**:<br>Thử nghiệm chéo để kiểm tra tính tổng quát hóa, bóc tách lỗi ảo giác và chuẩn hóa từ vựng đóng. | Khi test chéo Video 1, prompt mớm kịch bản làm model 3B bị ảo giác `hold`/`grab` và mất `touch` (F1 tụt 0.33); khi đổi sang prompt trung tính thì Video 7 lại tự sinh nhãn ngoài từ điển `place` ([X] Invalid). | - Khử mớm kịch bản bằng prompt trung tính tối giản (cân bằng quan sát tiếp xúc người và trạng thái vật thể).<br>- Căn chỉnh học thuật cho nhãn `get_off` (định nghĩa get_off bao hàm cả hành vi buông/đặt đồ xuống bề mặt theo chuẩn VidVRD mà không hardcode).<br>- Thêm cơ chế bảo vệ trong notebook Colab chống nạp lẫn frame cũ. | Video 7 đạt F1 = 1.0 tuyệt đối (xóa sạch lỗi Invalid `place`); Video 1 đạt Precision 100%, F1 = 0.80 (bắt chuẩn `touch` và `get_off`, sạch 100% ảo giác `hold`/`grab`); commit lưu mốc `85baba5`. | ✅ Done |

**Tổng kết ngày:** Hôm nay mình tập trung nâng cấp toàn diện pipeline thị giác và chuẩn hóa hệ thống prompt trên cả hai phân cảnh video:
1. Hoàn thiện `pipeline.py` với cơ chế dò vết ngược (Backward Association) để theo dấu túi xách trên tay người trước khi đặt xuống bàn, bổ sung Gap Filling vá lỗi mất dấu ngắn và thuật toán lấy mẫu thích ứng (Adaptive Sampling) chống flicker mà không làm méo mó nhịp thời gian.
2. Khi kiểm chứng chéo trên Video 1, phát hiện hiện tượng prompt mớm kịch bản gây ảo giác `hold`/`grab` và mất nhãn `touch`. Mình đã bóc tách nguyên nhân, áp dụng giải pháp "dao mổ tối giản" khử hoàn toàn ám thị kịch bản, đồng thời căn chỉnh định nghĩa ngữ nghĩa nhãn `get_off` (bao gồm cả hành vi đặt đồ xuống bề mặt / placing down theo đúng quy ước phân loại của VidVRD).
3. Kết quả nghiệm thu thực nghiệm trên Colab: Video 7 đạt F1 = 1.0 tuyệt đối (sạch bóng lỗi Invalid `place`), Video 1 đạt Precision 100%, F1 = 0.80 (khôi phục hoàn hảo quan hệ `touch` và `get_off`, triệt tiêu hoàn toàn ảo giác). Lưu trữ 2 mốc commit quan trọng `a5faea2` và `85baba5` lên GitHub repo.

---

## [YYYY-MM-DD]

| Task | Khó khăn | Giải pháp | Kết quả | Status |
|--------|------|--------|--------|------|
| | | | | |

**Tổng kết ngày:**

---

<!-- Format: copy block trên cho mỗi ngày làm việc -->
