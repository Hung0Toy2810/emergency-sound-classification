# Sound Classification Interactive Web Demo

> [Quay lại Root README.md](../../README.md) để xem tổng quan dự án và hướng dẫn cài đặt chung.

---

Web demo so sánh 2 mô hình phân loại âm thanh nguy hiểm:

- `SoundCNN`
- `ViTAST (Audio Spectrogram Transformer)`

Ứng dụng nhận file audio, tạo spectrogram, chạy cả 2 model, rồi hiển thị kết quả side-by-side để so sánh.

## Nhãn lớp

- `Crash_Collision`
- `Siren`
- `Scream_Shout`
- `Gunshot_Explosion`

## Cấu trúc chính

- `app.py`: Flask backend
- `audio_processor.py`: preprocess audio cho cả 2 pipeline
- `model_loader.py`: load checkpoint
- `src/cnn_model.py`: kiến trúc CNN
- `src/ast_model.py`: kiến trúc AST/ViT
- `templates/index.html`: giao diện web
- `static/js/main.js`: logic frontend
- `static/css/style.css`: style
- `models/`: checkpoint đã train

## Chạy demo

```bash
pip install -r requirements.txt
python app.py
```

Sau đó mở:

```text
http://localhost:5000
```

## Endpoint hữu ích

- `GET /health`: kiểm tra trạng thái server và model
- `POST /predict`: upload file audio và nhận kết quả dự đoán
- `GET /metrics`: metrics hardcoded để hiển thị dashboard
- `GET /images/cnn_cm`: confusion matrix của CNN
- `GET /images/ast_cm`: confusion matrix của AST

## Lưu ý

- Lần đầu chạy AST có thể cần cache/tải backbone Hugging Face.
- Nếu một model load lỗi, app vẫn chạy và báo rõ model nào unavailable.
- File upload hiện hỗ trợ các đuôi phổ biến như `wav`, `mp3`, `flac`, `ogg`, `m4a`, `aac`, `wma`.
