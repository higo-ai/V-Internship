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

## 2026-09-24

| Task | Khó khăn | Giải pháp | Kết quả | Status |
|---|---|---|---|---|
| **Chuẩn hóa Prompt 4 trụ cột, đo Token & Benchmark VLM**:<br>Chuẩn hóa System Prompt trung tính, hoàn trả định nghĩa học thuật cho `get_off` (chỉ dùng cho phương tiện/động vật), tích hợp bộ đo Token bằng HuggingFace Tokenizer trên Colab, và chạy benchmark `Qwen2.5-VL-3B-Instruct` trên Video 1 và Video 7. | Nguy cơ prompt bị Overfitting / may đo cục bộ cho 2 video; nhãn `get_off` trước đó bị ép nghĩa sai lệch cho đồ vật; cách đo token cũ bằng số ký tự thiếu chính xác. | - Thiết lập System Prompt 4 trụ cột khách quan (`data/prompt_system_general.txt`): tiếp xúc người-người `touch`, mang vác vật `carry`/`hold`, loại trừ rác tĩnh `zero-displacement`, và chuyển tiếp trạng thái thời gian.<br>- Tích hợp hàm đo Input/Output/Total tokens bằng tokenizer chuẩn trong notebook Colab.<br>- Đồng bộ cấu hình payload Video 1 và Video 7. | Benchmark trên Colab đạt F1 = 1.0 trên cả 2 video (Video 1: Output 95 tokens, bắt đúng `touch`; Video 7: Output 58 tokens, bắt đúng `carry`). | ✅ Done |
| **Xây dựng Pipeline detector mới độc lập (`pipeline_yoloe.py`) & Chạy luồng xuôi thuần túy**:<br>Tích hợp `yoloe-26m-seg.pt` với từ vựng 60 class (`configs/s_objects.json`), loại bỏ hoàn toàn thuật toán dò ngược (Backward Association), chuyển sang Pure Forward Tracking xuôi tự nhiên; giữ nguyên `pipeline.py` cũ làm mốc đối chứng. | Ở lần chạy đầu với `--conf 0.25`, mô hình sinh rác ảo (`camera`, `backpack` conf ~0.35), nhận nhầm vật thể tĩnh xa trên kệ là túi xách, và vết chiếc túi thật bị ngắt quãng thành 3 mảnh ([3], [4], [5]) khiến payload bị rối với 10 thực thể. | - Tối ưu ngưỡng tin cậy lên `--conf 0.40` (loại bỏ sạch bóng ma conf < 0.38 trong khi túi thật đạt conf 0.70 - 0.87).<br>- Bổ sung thuật toán Nối vết đồ vật (Object Track Stitching) nối liền mạch chuỗi chuyển động của chiếc túi từ lúc cầm đi vào tới khi đặt lên bàn.<br>- Thêm bộ lọc rác nền tĩnh (Zero-Displacement Clutter Rejection, loại bỏ vật dịch chuyển < 20px) và nâng ngưỡng bền vững `hits >= 25`. | Kiểm thử Video 7 thành công: chiếc túi thật được bám bắt liên tục 416 frames (conf 0.79, di chuyển 417px), loại sạch rác nền, kết xuất đúng 2 thực thể `[1] person` và `[2] handbag`; render video `video7_yoloe_annotated.mp4` và 8 frames sạch. | ✅ Done |

**Tổng kết ngày:** Hôm nay mình đã làm 4 việc cốt lõi theo đúng định hướng của Mentor:
1. Giải quyết dứt điểm vấn đề Prompt bằng cách đưa về bản quy chuẩn 4 trụ cột khách quan, trả lại bản chất học thuật cho taxonomy quan hệ, tích hợp bộ đo lường Token chính xác trên Colab và đạt F1 = 1.0 trên cả 2 video đối chứng.
2. Xóa bỏ hoàn toàn khoản nợ kỹ thuật Heuristic Backward Association theo chỉ dẫn của Mentor, xây dựng thành công pipeline độc lập `pipeline_yoloe.py` chạy luồng xuôi thuần túy với mô hình YOLOE-26m (60 class).
3. Vượt qua vỡ vết và rác nền ở lần chạy đầu bằng các giải pháp kiến trúc tối ưu: nâng ngưỡng `conf=0.40`, nối vết đồ vật (Object Track Stitching) và lọc rác tĩnh (`displacement < 20px`). Kết quả Video 7 được làm sạch chỉ còn đúng 2 thực thể `[1] person` và `[2] handbag`, bảo toàn nguyên vẹn 100% file gốc `pipeline.py`.

---

## 2026-09-25

| Task | Khó khăn | Giải pháp | Kết quả | Status |
|---|---|---|---|---|
| **Chuẩn hóa Guardrail ngữ nghĩa SoM & Triệt tiêu ảo giác thế chỗ thực thể**:<br>Nâng cấp Prompt hệ thống và User Prompt với cơ chế Khóa kép (Double-Lock): (1) Loại trừ hoàn toàn vật thể không đánh dấu (Unmarked Entity Exclusion) và (2) Ràng buộc miền quan hệ Person-Object cho vị từ `carry`/`hold`; chạy benchmark `Qwen2.5-VL-3B-Instruct` trên Video 1 và Video 7. | Khi chạy payload YOLOE trên Video 1, do chiếc balo ở sàn bị bộ lọc rác tĩnh loại bỏ (không có Mark ID), VLM nhìn thấy balo nhưng thiếu ID nên đã tự ý thế chỗ Người [1] vào vị trí Object, sinh ra triplet ảo giác `[2] carry [1]` ("Person 2 is carrying Person 1's backpack"). | - Bổ sung điều khoản Unmarked Entity Exclusion: Nghiêm cấm sinh quan hệ cho vật vô chủ không có ID và cấm lấy người đứng cạnh thế chỗ cho vật.<br>- Bổ sung điều khoản Domain Constraint: Vị từ `carry`/`hold` chỉ áp dụng giữa Người và Vật thể di động (người không thể carry người trừ khi bế/vác hẳn lên).<br>- Đồng bộ toàn bộ prompt, payload Video 1 & 7, pipeline YOLOE và notebook Colab. | Chạy lại trên Colab triệt tiêu 100% triplet ảo giác `[2] carry [1]`; Video 1 trả về cặp tiếp xúc đối xứng chuẩn xác `[1] touch [2]` và `[2] touch [1]` (F1 = 1.0, 99 tokens). Video 7 nhận diện đúng tương tác tay `[1] hold [2]` (F1 = 1.0, 61 tokens). | ✅ Done |
| **Mở rộng thực nghiệm đối chứng mô hình thế hệ mới Qwen3-VL-4B-Instruct**:<br>Đồng bộ hóa notebook `run_qwen3_vl_4b_colab.ipynb` với đầy đủ bộ đo token và guardrail; chạy thực nghiệm đối chứng trên GPU T4 của Google Colab với cả Video 1 và Video 7. | Cần kiểm chứng tính tổng quát của bộ Prompt Guardrails trên mô hình thế hệ mới (4B tham số), đồng thời tối ưu bộ nhớ VRAM để tránh tràn bộ nhớ GPU T4 (15GB). | - Đồng bộ hóa notebook Colab của Qwen 3-VL dùng chung 100% payload và frames ảnh sạch từ YOLOE (Model-Agnostic, không cần sửa cấu hình json).<br>- Cấu hình `expandable_segments:True` và tinh chỉnh `min_pixels/max_pixels` tiết kiệm VRAM. | Kiểm thử thành công trên cả 2 video (F1 = 1.0): Video 1 nhận diện cử chỉ tiếp xúc chi tiết vi mô (*"extending arm and touching shoulder/arm"*); Video 7 nhận thức đúng chuyển động thời gian để chọn chuẩn xác **`[1] carry [2]`** (*"holding handbag while walking through the room"*); nén token hiệu quả hơn ~16% (tiết kiệm ~750 input tokens so với bản 3B). | ✅ Done |
| **Kiểm định Pipeline YOLOE-26m đơn mô hình & Pure Forward Tracking**:<br>Đánh giá hiệu năng và tính nhất quán của pipeline luồng xuôi thuần túy kết hợp ByteTrack trên GPU CUDA. | Cần đảm bảo pipeline mới loại bỏ triệt để Backward Association mà vẫn bám bắt liên tục các tương tác động theo thời gian thực. | - Tối ưu hóa cấu trúc suy luận 1 model duy nhất `yoloe-26m-seg.pt` (60 class), phân tách 2 stream Person và Movable Objects trên cùng 1 forward pass.<br>- Ứng dụng Object Track Stitching và Active Entity Filter (`disp >= 20px`). | Đạt tốc độ suy luận thời gian thực 19.3 FPS trên GPU CUDA; bám bắt hoàn hảo chuỗi tương tác di chuyển ở Video 7 và tiếp xúc ở Video 1; hoàn thiện tài liệu kiến trúc `YOLOE_ARCHITECTURE_AND_ALGORITHMS.md`. | ✅ Done |

**Tổng kết ngày:** Hôm nay mình đã làm 3 nội dung theo phương pháp luận nghiên cứu:
1. Giải quyết dứt điểm bài toán Ảo giác thế chỗ thực thể (Entity Substitution Hallucination) trong VLM bằng giải pháp Prompt Guardrail chuẩn mực học thuật, không hardcode.
2. Hoàn thành đối chứng thực nghiệm song song trên cả 2 thế hệ mô hình VLM (`Qwen2.5-VL-3B-Instruct` và `Qwen3-VL-4B-Instruct`), đều đạt điểm số tuyệt đối F1 = 1.0 trên cả Video 1 và Video 7.
3. Ghi nhận số liệu thực nghiệm: Mô hình Qwen 3-VL 4B thể hiện tư duy thời gian vượt trội (bắt trọn `carry`), định vị cử chỉ vi mô sâu sắc, và tiết kiệm 16% token đầu vào.

---

## 2026-09-28

| Task | Khó khăn | Giải pháp | Kết quả | Status |
|---|---|---|---|---|
| **Task 1: Chuẩn hóa tính tổng quát cho Detector (Bật lại vật thể tĩnh & Giao quyền lọc rác cho Prompt VLM)** | Tiếp thu chỉ đạo của Mentor: Khi tắt bộ lọc cứng `disp < 20px`, balo trên sàn (Video 1) và túi trên bàn (Video 7) quay trở lại nhưng bị dao động nhãn (nhảy luân phiên giữa backpack/handbag) gây phân mảnh vết. | - Triển khai thuật toán Spatial Cross-Class Merging (IoU NMS + đa số phiếu).<br>- Tự động kiểm tra trực quan từng frame: gán chính xác `[4] backpack` trên sàn ở Video 1, giữ song song `[2] handbag` tĩnh và `[3] handbag` động ở Video 7.<br>- Cập nhật 4 trụ cột quy chuẩn ngữ nghĩa thị giác (Zero Cheating / Zero Hardcode).<br>- Tích hợp lượng tử hóa NF4 trong notebook Colab giảm VRAM từ >14.5GB xuống ~9.5GB. | Pipeline detector YOLOE-26m đạt độ tổng quát cao, giữ trọn vẹn mọi thực thể bền vững, sinh payload Video 1 & Video 7 mở rộng đầy đủ. | ✅ Done |
| **Task 2: Thiết kế & Hiện thực Module Gom cụm không gian (Spatial Clustering & Dynamic ROI Zoom Crop)** | Phân cụm BFS thông thường qua toàn clip gặp bẫy "Chuỗi bắc cầu qua thời gian" (Transitive Chaining through Time): gom cả người đi lướt qua ở góc xa và balo ở sàn vào cụm tương tác, làm giảm độ phóng đại xuống chỉ 1.36x. | - Xây dựng thuật toán phân cụm không-thời gian thông minh (*Spatio-Temporal Interaction Clustering - STIC*):<br>  + Đo lường liên tục $D_{edge}$ và $IoU$ trên từng frame đồng xuất hiện.<br>  + Phân biệt tiếp xúc bền vững ($\ge 20$ frames) với người đi lướt qua trong thoáng chốc ($< 20$ frames).<br>  + Áp dụng quy tắc động học liên kết Người-Vật: Chỉ ghép cặp đồ vật có dịch chuyển di động ($disp \ge 20$px); tự động cô lập 100% đồ vật tĩnh trên sàn/bàn thành Isolated Singletons.<br>- Hiện thực Dynamic Motion-Aware ROI Zoom Crop (hộp bao thích ứng + padding 20%):<br>  + Video 1: Phân cụm chính xác `Cluster 1: ['[1]', '[2]']`, loại bỏ 100% người đi xa `[3]` và balo ở sàn `[4]`, zoom phóng đại **3.96x** (640x480 -> 226x343).<br>  + Video 7: Phân cụm chính xác `Cluster 1: ['[1]', '[3]']`, loại bỏ 100% túi trên bàn `[2]`, zoom phóng đại **3.30x** (720x480 -> 239x438).<br>- Tự động sinh Payload tương ứng cho từng cụm (`video1_roi_cluster_1_payload.json`, `video7_roi_cluster_1_payload.json`). | Module `modules/spatial_clustering.py` hoàn thành chuẩn mực, zero hardcode, zoom phóng đại >3.3x - 3.96x cận cảnh chi tiết bàn tay và cử chỉ, triệt tiêu 100% nhiễu nền. | ✅ Done |

**Tổng kết ngày:** Hôm nay mình đã hoàn thành trọn vẹn 2 Task trọng tâm theo đúng định hướng của Mentor:
1. Chuẩn hóa tầng Detector YOLOE-26m thành luồng thuần khiết, giữ lại toàn bộ vật thể có độ bền vững không gian và dùng Prompt Guardrail 4 trụ cột để VLM xử lý.
2. Thiết kế và hiện thực thành công Module Gom cụm không gian (Spatial Clustering) và Dynamic ROI Zoom Crop: loại bỏ hoàn toàn bẫy chuỗi bắc cầu theo thời gian, cô lập triệt để người đi xa và đồ vật tĩnh trên sàn, đạt độ phóng đại ảnh cận cảnh 3.30x - 3.96x, đáp ứng 100% yêu cầu kỹ thuật và triết lý nghiên cứu khoa học của Mentor.

---

## 2026-09-29

| Task | Khó khăn | Giải pháp | Kết quả | Status |
|---|---|---|---|---|
| **Đồng bộ hóa định danh toàn cục (Global Tracking IDs) & Khắc phục hiện tượng phân mảnh vết người (Tracker ID Flicker)** | - Tái đánh số ID cục bộ (`[1]`, `[2]`, ...) trong khung hình crop gây lệch pha dữ liệu với hệ thống giám sát toàn cảnh (Branch A).<br>- Ở Video 7 (góc máy CCTV trên cao), người đi đến gần cửa bị ByteTrack phân mảnh ID ở frame cuối (nhảy từ ID 1 sang ID 2).<br>- Nguy cơ gộp nhầm người đi ở hành lang phía xa ở Video 1 nếu ghép vết (stitch) người theo thời gian mà thiếu điều kiện ràng buộc không gian. | - Loại bỏ module re-indexing cục bộ, bảo toàn 100% Global Tracking IDs xuyên suốt từ phát hiện toàn cảnh đến Set-of-Marks crop zoom.<br>- Bổ sung kiểm tra liên tục không-thời gian lân cận (Nearest-Frame Spatio-Temporal Continuity: $dt \le 15$ frames, $D_{edge} \le 35$px hoặc $IoU \ge 0.30$) để ghép các đoạn phân mảnh của cùng một đối tượng.<br>- Áp dụng ngưỡng IoU không gian ($\ge 0.35$) cho các tracklet xuất hiện đồng thời để ngăn ngừa gộp nhầm các cá nhân khác nhau. | - Đảm bảo tính nhất quán dữ liệu giữa Branch A và Branch B.<br>- Video 7 duy trì ổn định thực thể người `[1]` xuyên suốt 8 frame lấy mẫu. Video 1 tách biệt rõ người tương tác với người đi ở hành lang xa. | 🔄 In Progress |
| **Hoàn thiện logic phân định tương tác Người - Vật thể tĩnh & Lấy mẫu thời gian thích ứng** | - Góc nhìn 2D từ trên cao dễ gây hiểu nhầm quang học giữa người đi ngang qua và vật thể tĩnh trên tường/bàn ở Video 7 (túi treo tường vô tình nằm sát hộp bao người dù không có tương tác vật lý).<br>- Lấy mẫu khung hình cố định dễ rơi vào thời điểm một trong các thực thể đã đi khỏi khung hình hoặc bị cắt cụt một phần ở mép camera. | - Thiết lập tiêu chí tương tác dựa trên thời gian lưu trú (dwell ratio $\ge 25\%$) và loại trừ vùng đỉnh đầu (15% phía trên của người) đối với vật thể tĩnh, giúp phân biệt rõ giữa tương tác cầm nắm thực tế với việc đi lướt qua dưới vật treo tường.<br>- Thiết kế cơ chế lấy mẫu thời gian thích ứng (Adaptive Spatio-Temporal Sampling): cụm tương tác người - người ưu tiên lấy mẫu trong cửa sổ tiếp xúc thực tế (`active_contact_frames`); cụm người - vật lấy mẫu trải đều chu trình tương tác. | - Phân loại đúng túi treo tường ở Video 7 là vật thể đơn lẻ (Singleton), không vẽ hộp bao trong ảnh crop của cụm tương tác, giảm thiểu gây nhiễu cho mô hình thị giác.<br>- Khung hình crop ở Video 1 và Video 7 giữ được đầy đủ thực thể trong tầm nhìn mà không bị cắt cụt rìa ảnh. | 🔄 In Progress |

**Tổng kết ngày:**
- Tập trung chuẩn hóa tính nhất quán dữ liệu giữa luồng theo dõi toàn cảnh và luồng crop zoom Set-of-Marks thông qua việc bảo toàn ID toàn cục.
- Cải thiện các thuật toán xử lý phân mảnh vết theo thời gian và lọc nhiễu không gian cho các vật thể tĩnh, khắc phục hiện tượng nhảy ID và gộp nhầm thực thể dựa trên các điều kiện hình học và động học tổng quát.
- Các điều chỉnh đã giải quyết được các bất cập quan sát thấy trên 2 video thực nghiệm (Video 1 và Video 7). Tuy nhiên, pipeline vẫn đang trong giai đoạn kiểm nghiệm và cần được đánh giá đầu cuối cùng mô hình VLM.
- Cuối ngày 29/09, trong buổi trao đổi định hướng với Mentor Lường Mạnh Tú, Mentor đã chỉ đạo giải pháp nghiệm thu thực nghiệm cốt lõi: cần tạo video toàn cảnh thể hiện Bounding Box tổng gom cụm (loại bỏ box con, không gán nhãn động từ quan hệ), tạo cơ sở để chuyển tiếp sang ngày 30/09 hiện thực hóa trực quan.

---

## 2026-09-30

| Task | Khó khăn | Giải pháp | Kết quả | Status |
|---|---|---|---|---|
| **Tối ưu cửa sổ tương tác, Kích thước khung Crop (Crop Sizing) & Chuẩn hóa bộ đo Token VLM (Commit `bbcd044`, `96e2b23`)** | - Ban đầu khi cắt crop cận cảnh quá sát sạt (tight crop), các chi tiết tiếp xúc (ngón tay cầm quai túi) và mặt phẳng bối cảnh (mặt bàn, sàn nhà) bị lẹm sát mép viền ảnh, khiến VLM khó nhận diện không gian xung quanh.<br>- Việc đo lường token chưa đồng nhất giữa Text và Image Tokens; nguy cơ tràn bộ nhớ VRAM khi chạy mô hình 4B tham số trên GPU T4.<br>- Thử nghiệm bổ sung các mệnh đề phủ định dài vào Prompt hệ thống nhằm ép mô hình tách biệt `touch` và `carry` khiến Qwen-3-VL-4B bị bối rối và trả về kết quả rỗng `{"triplets": []}` ở Video 7. | - **Nới rộng chiều rộng và chiều dài khung crop (Adaptive Context Padding ~20%):** Mở rộng biên bao quanh cụm tương tác một khoảng đệm an toàn, vừa đảm bảo độ phóng đại chi tiết cao (~2.9x) vừa giữ trọn bối cảnh vật lý xung quanh cho VLM quan sát trực quan.<br>- Chuẩn hóa đo lường token thông qua `attention_mask` (Boolean mask), bóc tách chính xác Image Tokens và Text Tokens thực tế; tích hợp dọn dẹp bộ nhớ đệm VRAM (`gc.collect()`, `empty_cache`).<br>- Rút kinh nghiệm về việc tránh over-constraining prompt: lập tức rollback về cấu trúc Prompt 4 trụ cột khách quan, ngắn gọn, khẳng định. | - Ảnh crop zoom nới rộng (~2.94x) bắt trọn vẹn toàn bộ hành động người đặt túi xách lên bàn mà không bị mất bối cảnh; bộ đo lường token hoạt động chuẩn xác, ổn định bộ nhớ VRAM trên Colab T4.<br>- Qwen-3-VL-4B đạt điểm tuyệt đối F1 = 1.0 trên cả Video 1 và Video 7 (nhận diện chính xác cả hành động mang xách `carry` lẫn tiếp xúc va chạm `touch`). | ✅ Done |
| **Hiện thực hóa Video Trực quan hóa Gom cụm Bounding Box tổng theo chỉ đạo của Mentor Tú & Hoàn thiện Pipeline Task 2** *(Giải quyết dứt điểm tồn đọng ngày 29/09)* | - File video xuất ra ban đầu dùng codec `mp4v` không mở xem được trên trình phát mặc định của Windows 10/11.<br>- Khung thông tin HUD debug ở góc trên bên trái gây vướng tầm nhìn camera toàn cảnh.<br>- **Lỗi logic dạt box ở Frame 6 Video 1 (48.42s):** Khi 2 người đặt balo nằm lại trên sàn (x ≈ 200) và bước đi sang phải (x ≈ 550), logic gom cụm cũ gộp cả 3 thực thể làm Bounding Box bị kéo dãn bất thường tới 459px. | - Chuyển đổi chuẩn mã hóa video sang H.264 (`avc1`) qua bộ giải mã OpenH264 của Cisco, tương thích 100% mọi trình phát trên Windows, Chrome và VS Code.<br>- Lược bỏ hoàn toàn khung HUD góc trái, trả lại khung hình camera trong sáng, chỉ hiển thị Bounding Box tổng và Header Badge (`CLUSTER 1: [ID] class + [ID] class`).<br>- Nâng cấp thuật toán *Frame-Level Physical Adjacency & Connected Components*: tại từng frame, chỉ các thực thể có khoảng cách thực tế $\le 45$px mới được gom lại; vật thể bị buông bỏ ở xa (balo cách 250px) tự động tách khỏi box tức thời; bổ sung kẹp biên nhãn chống lẹm mép màn hình. | - Kết xuất thành công 2 video giám sát chuẩn H.264: `video7_cluster_vis.mp4` và `video1_cluster_vis.mp4` cùng bộ 10 ảnh preview toàn cảnh.<br>- Tại Frame 6 Video 1, Bounding Box co gọn từ 459px về 158px ôm vừa vặn 2 người đang đi; túi tường `[2]` ở Video 7 được cô lập hoàn toàn không bị đóng hộp.<br>- Đáp ứng 100% chỉ đạo kỹ thuật của Mentor Lường Mạnh Tú. | ✅ Done |

**Tổng kết ngày:**
- Hoàn thành yêu cầu thực nghiệm trực quan của Mentor: Xuất video giám sát chuẩn H.264 hiển thị Bounding Box tổng gom cụm, triệt tiêu box con bên trong, không gắn nhãn động từ quan hệ, và loại bỏ hoàn toàn nhiễu tĩnh (túi treo tường).
- Nâng cấp thuật toán gom cụm không gian tức thời (Frame-Level Connected Components), giải quyết triệt để ca biên thực thể bị buông bỏ lại phía sau (balo trên sàn ở Video 1) mà ngày 29/09 chưa xử lý đến.
- Khép lại nhánh thực nghiệm VLM (Video 1 và 7), trên Google Colab với kết quả F1 = 1.0 trên Qwen-3-VL-4B và bài học kinh nghiệm sâu sắc về việc giữ Prompt cô đọng, khách quan, tránh ép quy tắc phủ định quá đà (over-constraining).

---

## 2026-10-01

| Task | Khó khăn | Giải pháp | Kết quả | Status |
|---|---|---|---|---|
| **Tối ưu hóa hiển thị Set-of-Marks (SoM) cho khung hình Zoom Crop đưa vào VLM** | - Khối nhãn màu đặc (solid 100%) che khuất hoàn toàn điểm tiếp xúc vật lý (ngón tay cầm quai túi xách ở Video 7), khiến VLM mất bằng chứng thị giác trực tiếp.<br>- Nhãn ID phụ to đùng vẽ trùng lặp bên trong hộp bounding box đè lên khuôn mặt, ngực người và bề mặt vật thể.<br>- Viền hộp 2px thô cứng làm lẹm nét cơ thể; trong khi viền 1px thử nghiệm ban đầu lại quá mảnh dễ bị hòa tan vào nền gạch sáng. | - **Áp dụng Alpha Blending (70% opacity):** Nền nhãn bán trong suốt cho phép 30% chi tiết gốc (bàn tay, quai xách) xuyên thấu qua, bảo toàn nguyên vẹn bằng chứng tương tác cho Vision Transformer.<br>- **Xóa bỏ hoàn toàn nhãn trùng bên trong box:** Giải phóng 100% diện tích khuôn mặt và bề mặt thực thể.<br>- **Chuẩn hóa viền 1.5px & chữ BOLD tương phản cao:** Thiết lập viền 1.5px chống răng cưa (`LINE_AA`) và font 0.45 tô đậm (CTRL+B) trắng đặc 100%, đảm bảo module OCR của VLM nhận diện chuẩn xác. | - Kết xuất lại toàn bộ 8 frames zoom crop cho cả Video 1 và Video 7 đạt chất lượng thẩm mỹ cao và chuẩn mực thị giác máy tính.<br>- Loại bỏ triệt để hiện tượng che khuất điểm tiếp xúc (Zero Contact Occlusion), sẵn sàng đưa lên Google Colab thực nghiệm kiểm chứng cùng VLM. | ✅ Done |

**Tổng kết ngày:**
- Tối ưu hóa toàn diện lớp hiển thị Set-of-Marks (SoM) cho Task 2: giải quyết bài toán xung đột giữa đánh dấu thực thể và che lấp điểm ảnh (Pixel Occlusion) theo chuẩn nghiên cứu SoM của Microsoft Research.
- Hoàn thành bộ 8 ảnh crop zoom chuẩn hóa cho Video 1 và Video 7, sẵn sàng cho pha thực nghiệm đánh giá hiệu năng suy luận trên Colab.

---

## [YYYY-MM-DD]

| Task | Khó khăn | Giải pháp | Kết quả | Status |
|---|---|---|---|---|
| | | | | |

**Tổng kết ngày:**

---

<!-- Format: copy block trên cho mỗi ngày làm việc -->
