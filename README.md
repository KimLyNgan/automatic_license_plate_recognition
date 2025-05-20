Hệ thống Nhận diện Biển số Tự động (ANPR)
Đây là một dự án triển khai hệ thống Nhận diện Biển số Tự động (Automatic License Plate Recognition - ANPR) cho biển số xe Việt Nam. Hệ thống sử dụng các mô hình học sâu để phát hiện biển số, nhận diện từng ký tự, và ghép chúng lại thành chuỗi biển số hoàn chỉnh, đồng thời cung cấp khả năng tra cứu tỉnh thành tương ứng.

Tính năng chính
Phát hiện biển số: Xác định vị trí biển số trên ảnh.

Nhận diện ký tự: Trích xuất và nhận diện từng ký tự trên biển số.

Xử lý biển số 1 dòng & 2 dòng: Thuật toán sắp xếp ký tự thông minh để xử lý cả biển số có một hoặc hai dòng.

Tra cứu tỉnh thành: Xác định tỉnh/thành phố dựa trên mã số biển số.

API Web (Flask): Cung cấp giao diện API để xử lý ảnh và trả về kết quả.

Giao diện người dùng (HTML/JS): Giao diện web đơn giản để tải ảnh từ thiết bị hoặc URL và hiển thị kết quả trực quan.

Chi tiết mô hình
Hệ thống sử dụng hai mô hình học sâu, đều được tinh chỉnh (fine-tuned) từ kiến trúc YOLOv8n:

Mô hình Phát hiện Biển số: Chuyên biệt để phát hiện vị trí của biển số trên ảnh đầu vào.

Mô hình Nhận diện Ký tự: Chuyên biệt để nhận diện từng ký tự riêng lẻ trên ảnh biển số đã được cắt.

Cả hai mô hình này được triển khai hiệu quả bằng ONNX Runtime, giúp tối ưu hóa tốc độ suy luận (inference) trên CPU.

Luồng xử lý chính
Tiền xử lý ảnh đầu vào: Ảnh được tải lên hoặc lấy từ URL sẽ được tiền xử lý (resize, chuẩn hóa) để phù hợp với kích thước đầu vào của mô hình ONNX (640x640).

Phát hiện biển số: Ảnh đã tiền xử lý được đưa vào mô hình phát hiện biển số. Kết quả là các bounding box (bbox) của biển số.

Cắt ảnh biển số: Vùng biển số được phát hiện sẽ được cắt ra khỏi ảnh gốc. (Hệ thống được thiết kế để xử lý một biển số duy nhất trong khung hình).

Nhận diện ký tự: Ảnh biển số đã cắt được tiền xử lý và đưa vào mô hình nhận diện ký tự. Kết quả là các bbox của từng ký tự cùng với class ID của chúng.

Xử lý kết quả bbox và sắp xếp ký tự:

Các bbox ký tự được hậu xử lý để trích xuất tọa độ, chiều cao, và class ID.

Ghép chuỗi và tra cứu tỉnh thành:

Các class ID của ký tự được ánh xạ thành ký tự chữ/số tương ứng và ghép lại thành chuỗi biển số hoàn chỉnh.

Hai ký tự số đầu tiên của chuỗi biển số được sử dụng để tra cứu tên tỉnh/thành phố tương ứng từ một bảng dữ liệu đã định nghĩa.

Trả về kết quả: API sẽ trả về chuỗi biển số, tên tỉnh/thành phố, ảnh gốc có vẽ bbox biển số, và ảnh biển số đã cắt có vẽ bbox ký tự (dưới dạng Base64) để hiển thị trên giao diện web.

Hướng dẫn cài đặt và chạy
Để chạy ứng dụng này, bạn cần cài đặt Python và các thư viện cần thiết.

Clone Repository:

git clone https://github.com/KimLyNgan/automatic_license_plate_recognition.git

cd automatic_license_plate_recognition

Tạo môi trường ảo (khuyên dùng):

python -m venv .venv
# Trên Windows
.venv\Scripts\activate
# Trên macOS/Linux
source .venv/bin/activate

Cài đặt các thư viện Python:

pip install -r requirements.txt

(Đảm bảo file requirements.txt đã được tạo và chứa đầy đủ các thư viện như Flask, numpy, opencv-python, onnxruntime, requests, gunicorn, Flask-Cors.)

Đặt các file mô hình ONNX:

Tải hai file mô hình ONNX của bạn (best_new.onnx và kytubiensoxe.onnx).

Đặt chúng vào thư mục models/ trong project của bạn.

automatic_license_plate_recognition/
├── models/
│   ├── best_new.onnx
│   └── kytubiensoxe.onnx
└── ...

Chạy ứng dụng Flask:

gunicorn app:app -b 0.0.0.0:5050
# Hoặc nếu bạn đang trong giai đoạn phát triển và muốn debug:
# python app.py

Ứng dụng sẽ chạy trên http://0.0.0.0:5050/.

Cách sử dụng
Sau khi ứng dụng Flask đã chạy:

Truy cập giao diện web:
Mở trình duyệt web của bạn và truy cập địa chỉ: http://localhost:5050/

Xử lý ảnh từ thiết bị:

Trong phần "Upload Image from Your Device", nhấn "Choose File" để chọn một ảnh từ máy tính của bạn.

Nhấn "Upload and Process Image".

Ảnh gốc bạn chọn sẽ hiển thị bên trái, và sau khi xử lý, ảnh gốc có bbox biển số, ảnh biển số đã crop có bbox ký tự, chuỗi biển số và tên tỉnh thành sẽ hiển thị.

Xử lý ảnh từ URL:

Trong phần "Process Image from URL", nhập một đường dẫn URL hợp lệ của một hình ảnh (ví dụ: https://example.com/your-plate-image.jpg).

Nhấn "Process Image from URL".

Hệ thống sẽ tải ảnh từ URL, xử lý và hiển thị kết quả tương tự như khi tải ảnh từ thiết bị.

Ghi chú triển khai (Deploy)
Ứng dụng này được thiết kế để triển khai trên các nền tảng PaaS như Render.

Procfile (web: gunicorn app:app) được cung cấp để hướng dẫn Render cách khởi động ứng dụng.

Các mô hình AI được tải một lần khi ứng dụng khởi động để tối ưu hiệu suất.
