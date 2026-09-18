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

## [YYYY-MM-DD]

| Task | Khó khăn | Giải pháp | Kết quả | Status |
|--------|------|--------|--------|------|
| | | | | |

**Tổng kết ngày:**

---

<!-- Format: copy block trên cho mỗi ngày làm việc -->
