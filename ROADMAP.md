# Lộ trình từng bước Dự án VinFast Internship
Theo dõi tiến độ nghiên cứu & triển khai giải pháp VLM End-to-End cho bài toán Video Relation Detection.

---

[Pha 1: Chốt Đầu Vào & Đầu Ra] 

[Pha 2: Chọn Mô hình VLM 2B phù hợp] 

[Pha 3: Thiết kế cách đưa Bounding Box vào VLM (Cốt lõi bài toán)] 

[Pha 4: Chạy thử nghiệm nhỏ (PoC)]

---

### - [x] PHA 1: Xác định rõ Đầu Vào (Input) và Đầu Ra (Output)
Trước khi đụng vào code, ta cần hiểu máy sẽ nhận gì và nhả ra gì:
* **Đầu vào (Input):**
  1. Một đoạn video ngắn (khoảng 3 – 5 giây) từ camera CCTV.
  2. Danh sách tọa độ (Bounding Boxes) của các đối tượng trong video theo thời gian (ví dụ: ID 1 là Người, ID 2 là Balo).
* **Đầu ra mong muốn (Output):**
  * Một cấu trúc dữ liệu gọn gàng (dạng JSON), ví dụ:
    `json
    [
      {" subject\: \person_1\, \relation\: \carry\, \object\: \backpack_2\}
 ]
 `
* Trong đó, 
elation bắt buộc phải nằm trong đúng danh sách 26 từ.

---

### - [x] PHA 2: Chọn Mô hình VLM 2B (Model Selection)
**Yêu cầu:** Dùng model VLM khoảng 2B để chạy end-to-end.

Hiện nay trên thế giới ở phân khúc ~2B (siêu nhẹ, chạy nhanh, ít tốn RAM/GPU), lựa chọn tốt nhất là:
 * **Qwen2-VL-2B-Instruct** (của Alibaba Cloud): Là mô hình VLM 2B mạnh nhất hiện nay về xử lý video, hỗ trợ nhận diện đối tượng và hiểu câu lệnh cực kỳ tốt. Mô hình này hoàn toàn miễn phí và mã nguồn mở.

---

### - [ ] PHA 3: Giải quyết TRỌNG TÂM CỦA BÀI TOÁN
"Nghiên cứu cách cung cấp thông tin các frame và bounding boxes để output ra được relation từ model VLM 2B?"

Có 2 cách tiếp cận để giải quyết vấn đề này:
* **Cách 1: Kỹ thuật thị giác (Visual Prompting / Set-of-Marks - SoM) [KHUYÊN DÙNG]**
  * Thay vì bắt con AI 2B đọc những con số tọa độ khô khan (mô hình nhỏ 2B đọc số tọa độ rất dễ nhầm lẫn), ta dùng code vẽ thẳng khung màu và gắn nhãn số lên các khung hình của video:
  * Khung đỏ có nhãn [1] quanh người.
  * Khung xanh có nhãn [2] quanh chiếc túi xách.
  * Sau đó gửi video đã vẽ khung này vào VLM kèm câu hỏi: *\Trong 26 hành động này đối tượng [1] đang làm hành động gì với đối tượng [2]?\*.
  * Con AI nhìn thấy nhãn [1] và [2] bằng mắt, nó sẽ trả lời cực kỳ chính xác!
* **Cách 2: Kỹ thuật văn bản (Coordinate Prompting):**
  * Gửi video gốc và gửi tọa độ bằng chữ dạng số [x1, y1, x2, y2] vào prompt. Cách này mô hình lớn (như GPT-4o) làm được, nhưng mô hình nhỏ 2B rất dễ bị loạn thị không tính được hình học.

---

### - [ ] PHA 4: Viết một kịch bản thử nghiệm nhỏ (Mini PoC)
Chuẩn bị một kịch bản gồm:
1. Lấy 1 video mẫu ngắn.
2. Thiết kế câu Prompt chuẩn chỉnh (chứa 26 động từ).
3. Kiểm tra xem mô hình trả về có chuẩn không.
