# 🎧 Audio Spectrogram Transformer (AST) Module

> 👈 **[Quay lại Root README.md](../../README.md)** để xem tổng quan dự án, kiến trúc hệ thống và hướng dẫn cài đặt chung.

---

Dự án này triển khai mô hình phân loại âm thanh đa nhãn dựa trên bài báo khoa học **"AST: Audio Spectrogram Transformer"** (MIT CSAIL, 2021). Mô hình thích nghi kiến trúc Vision Transformer (ViT) được tiền huấn luyện trên ImageNet (`google/vit-base-patch16-224`) sang xử lý phổ Mel Spectrogram của tín hiệu âm thanh 1 kênh.

Mô hình được tối ưu hóa chạy trên GPU CUDA sử dụng kỹ thuật Huấn luyện độ chính xác hỗn hợp (AMP FP16), Tiền xử lý dữ liệu ngoại tuyến (Offline Preprocessing) và hỗ trợ khôi phục tiến trình tự động (Checkpoint Resume).

---

## 📂 1. Cấu Trúc Thư Mục Dự Án

Thư mục dự án được tổ chức chuyên nghiệp và phân chia theo đúng vai trò:

```text
Trans/
│
├── bin/                                # Chứa các file thực thi ffmpeg và ffprobe
├── checkpoints/                        # Lưu các checkpoint huấn luyện (best, latest)
├── data_processing/                    # Chứa các script crawl và xử lý tập dữ liệu gốc
│   ├── split_metadata.py               # Phân chia metadata thành tập Train/Test độc lập
│   ├── reorganize_dataset.py           # Phân tách vật lý các file âm thanh vào thư mục train/test
│   └── preprocess_dataset.py           # Tính toán trước và lưu phổ log-Mel dưới dạng .pt
│
├── downloads/                          # Chứa bộ dữ liệu âm thanh gốc (.wav) và nhãn (.csv)
│   ├── train/                          # 1407 file âm thanh dùng cho huấn luyện
│   └── test/                           # 447 file âm thanh dùng cho kiểm thử
│
├── evaluation_results/                 # Chứa các biểu đồ kết quả đánh giá (PR curves, confusion matrix...)
├── src/                                # Mã nguồn lõi của mô hình PyTorch
│   ├── dataset.py                      # Lớp nạp dữ liệu AudioDataset & SpecAugment
│   ├── model.py                        # Kiến trúc mô hình ViTAST và thích nghi lớp Positional Embedding
│   └── train.py                        # Vòng lặp huấn luyện, đánh giá & AMP FP16
│
├── run.py                              # Script chính điều phối việc tiền xử lý và chạy huấn luyện
├── predict.py                          # Script chạy nhận diện thực tế trên một file âm thanh đơn lẻ
├── evaluate.py                         # Script đánh giá mô hình và vẽ đồ thị báo cáo
└── README.md                           # Tài liệu hướng dẫn dự án (File này)
```

---

## 🛠️ 2. Hướng Dẫn Từng Bước Chạy Dự Án

### Bước 1: Thiết lập Môi trường & Thư viện
Dự án yêu cầu cài đặt Python 3.10+ cùng các thư viện deep learning hỗ trợ GPU. Hãy cài đặt các thư viện sau:

```bash
# Cài đặt PyTorch hỗ trợ GPU CUDA 12.4
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Cài đặt các thư viện bổ trợ
pip install transformers==4.49.0 pandas scikit-learn matplotlib tqdm
```

### Bước 2: Chuẩn bị Dữ liệu (Phân chia tập Train/Test)
Chạy các script trong thư mục `data_processing` để chia tập dữ liệu theo tỷ lệ và phân bổ vật lý các file âm thanh:

```bash
# 1. Phân chia tệp metadata gốc thành train/test metadata
python data_processing/split_metadata.py

# 2. Phân chia các file .wav thực tế vào các thư mục downloads/train và downloads/test
python data_processing/reorganize_dataset.py
```

### Bước 3: Tiền xử lý dữ liệu ngoại tuyến (Offline Preprocessing)
Để loại bỏ nghẽn cổ chai CPU khi phải Resample và trích xuất phổ Mel liên tục lúc huấn luyện, chúng ta thực hiện tiền xử lý ngoại tuyến toàn bộ tập dữ liệu một lần duy nhất. File âm thanh sẽ được convert thành các tensor phổ `.pt` lưu trực tiếp trên ổ cứng:

```bash
python run.py --preprocess
```
*Thời gian huấn luyện sẽ nhanh hơn gấp ~20 lần sau bước này.*

### Bước 4: Huấn luyện Mô hình (Training & Resume)
Tiến hành huấn luyện mô hình trên GPU CUDA với AMP FP16. Lớp phân loại sẽ học nhận diện 4 nhãn âm thanh nguy hiểm: `Crash_Collision`, `Siren`, `Scream_Shout`, `Gunshot_Explosion`.

```bash
# Huấn luyện mô hình trong 10 Epochs, batch size 16
python run.py --epochs 10 --batch_size 16 --duration 10
```

**Khôi phục huấn luyện khi bị dừng đột ngột:**
Nếu quá trình huấn luyện bị ngắt giữa chừng (mất điện, tắt máy), bạn chỉ cần chạy lại lệnh kèm cờ `--resume`. Mô hình sẽ tự động nạp trạng thái gần nhất để huấn luyện tiếp tục:
```bash
python run.py --epochs 10 --batch_size 16 --duration 10 --resume
```

### Bước 5: Đánh giá & Vẽ đồ thị báo cáo (Evaluation)
Sau khi hoàn thành 10 epochs, chạy script sau để sinh ra các biểu đồ phục vụ cho slide báo cáo và tài liệu nghiên cứu:
```bash
python evaluate.py
```
Các file sau sẽ được tạo ra trong thư mục `/evaluation_results/`:
*   `precision_recall_curves.png`: Đường cong PR và điểm AP riêng cho từng lớp.
*   `confusion_matrix.png`: Ma trận nhầm lẫn chuẩn 4x4.
*   `specaugment_demo.png`: Minh họa phổ Mel gốc và phổ đã bị che (SpecAugment) của dự án.

### Bước 6: Chạy nhận diện thực tế trên một file âm thanh bất kỳ (Inference)
Dùng script `predict.py` để test nhanh mô hình với một tệp âm thanh `.wav` bất kỳ:
```bash
python predict.py --audio "downloads/test/Siren/sample_12.wav"
```
**Kết quả đầu ra trực quan:**
```text
===================================
       AST PREDICTION RESULTS       
===================================
Crash_Collision   :   0.26%  [....................]
Siren             :  99.83%  [###################.]
Scream_Shout      :   0.17%  [....................]
Gunshot_Explosion :   0.31%  [....................]
===================================
```

---

## ⚡ 3. Các Điểm Nhấn Kỹ Thuật (Phân Tích So Với Bài Báo AST)

Dự án đã bám sát và hiện thực hóa chính xác các kỹ thuật cốt lõi từ bài báo của tác giả Yuan Gong (MIT CSAIL, 2021):

1.  **Chuyển đổi 2D Positional Embedding bằng Bilinear Interpolation:**
    Do ảnh ImageNet ($224 \times 224$ chia ô không overlap) chỉ có 196 patches, trong khi âm thanh 10 giây của chúng ta tạo ra $12 \times 99 = 1188$ patches. Mô hình đã tự động tính toán lưới 2D và áp dụng nội suy song tuyến để co dãn positional embedding của ViT sang khớp với kích thước âm thanh mà không làm mất thông tin vị trí đã học.
2.  **Trung bình hóa trọng số kênh đầu vào:**
    Mô hình ViT được thiết kế nhận ảnh RGB 3 kênh. Dự án đã thích nghi sang âm thanh 1 kênh bằng cách tính trung bình cộng trọng số của 3 kênh đầu vào để gán cho kênh đơn của phổ âm thanh.
3.  **Data Augmentation (SpecAugment):**
    Tích hợp trực tiếp bộ lọc che ngẫu nhiên dải tần số (Frequency Masking) và dải thời gian (Time Masking) ngay trong luồng huấn luyện để chống Overfitting hiệu quả.
4.  **Cơ chế Self-Attention tự nhiên cho dữ liệu biến thiên:**
    Transformer xử lý các patch dưới dạng chuỗi đầu vào tuyến tính, giúp hệ thống có khả năng nhận diện các âm thanh có độ dài biến thiên (từ 1 giây đến 10 giây) một cách tự nhiên mà không cần thiết kế lại cấu trúc mạng.

---

## 📊 4. Kết Quả Huấn Luyện (Huấn luyện hoàn tất Epoch 10/10)

Độ chính xác trung bình toàn hệ thống đạt **mAP = 92.02%** trên tập kiểm thử độc lập (Test Set), các chỉ số chi tiết của mô hình:

*   **Train Loss**: `0.0353`
*   **Val Loss**: `0.2343`
*   **Val mAP**: `0.9202`
*   **Val F1 (Macro)**: `0.8469`

**Độ chính xác trung bình (Average Precision) từng lớp âm thanh:**
*   Tiếng còi hú cảnh báo (**Siren**): **98.57%** AP
*   Tiếng hét/la hét (**Scream_Shout**): **92.00%** AP
*   Tiếng súng nổ/nổ (**Gunshot_Explosion**): **89.71%** AP
*   Tiếng va chạm xe (**Crash_Collision**): **87.81%** AP
