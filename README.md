# Đồ án Tốt nghiệp: Nghiên cứu Phương pháp Học sâu cho bài toán Phân tích Cảm xúc tiếng Việt

Dự án thực nghiệm và so sánh toàn diện các phương pháp học máy và học sâu cho bài toán **Phân tích Cảm xúc phản hồi sinh viên (Sentiment Analysis)** tiếng Việt, sử dụng bộ ngữ liệu chuẩn **UIT-VSFC**. Các mô hình được nghiên cứu bao gồm Multinomial Naive Bayes, Support Vector Machine (SVM), Bidirectional LSTM và mô hình ngôn ngữ tiền huấn luyện **PhoBERT** (Full Fine-tuning).

## Thông tin thực hiện

| Thông tin | Nội dung |
|---|---|
| Sinh viên thực hiện | Nguyễn Thị Ngọc Mai |
| Mã số sinh viên | 22130161 |
| Lớp | DH22DTB |
| Trường | Đại học Nông Lâm TP. Hồ Chí Minh |
| Giảng viên hướng dẫn | TS. Nguyễn Văn Dũ |

## Bộ dữ liệu

Sử dụng bộ ngữ liệu **UIT-VSFC** (Vietnamese Students' Feedback Corpus) gồm 16.175 câu phản hồi ngắn của sinh viên, được gán nhãn theo 3 lớp cảm xúc:

| Nhãn | Lớp | Số lượng (Train / Val / Test) |
|---|---|---|
| 0 | Tiêu cực (Negative) | 8.190 / 1.132 / 1.409 |
| 1 | Trung tính (Neutral) | 1.180 / 164 / 167 |
| 2 | Tích cực (Positive) | 2.056 / 287 / 1.590 |

## Cấu trúc thư mục

```
code/
│
├── data/                        # Dữ liệu gốc UIT-VSFC và từ điển teencode
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   └── teencode.txt             # Từ điển 478 quy tắc ánh xạ teencode & viết tắt
│
├── fn/                          # Dữ liệu sau tiền xử lý (tạo ra khi chạy notebook 00)
│   ├── cleaned_train.csv
│   ├── cleaned_validation.csv
│   └── cleaned_test.csv
│
├── pkl/                         # Mô hình đã huấn luyện (tạo ra khi chạy notebook 01, 02)
│   ├── tfidf_vectorizer.pkl
│   ├── naive_bayes_model.pkl
│   ├── svm_best_model.pkl
│   ├── lstm_vocab.json
│   └── best_lstm_model_correct.pth
│
├── notebooks/                   # Jupyter Notebook — chạy theo thứ tự từ 00 đến 03
│   ├── 00_Tien_Xu_Ly.ipynb      # Tiền xử lý dữ liệu tiếng Việt
│   ├── 01_Baseline_NB_SVM.ipynb # Mô hình TF-IDF + Naive Bayes & SVM
│   ├── 02_BiLSTM.ipynb          # Mô hình học sâu Bidirectional LSTM
│   └── 03_PhoBERT.ipynb         # Fine-tuning PhoBERT & So sánh tổng thể
│
├── src/                         # Script Python độc lập (có thể chạy bằng dòng lệnh)
│   ├── preprocessing.py
│   ├── baselines.py
│   ├── lstm.py
│   ├── train_phobert.py
│   ├── generate_plots.py
│   └── predict.py
│
├── image/                       # Biểu đồ kết quả (tạo ra khi chạy notebook 01–03)
├── .gitignore
└── README.md
```

## Hướng dẫn cài đặt

### Bước 1: Tạo và kích hoạt môi trường ảo

```bash
python3 -m venv env_doan
source env_doan/bin/activate
```

### Bước 2: Cài đặt các thư viện

```bash
pip install --upgrade pip
pip install pandas numpy scikit-learn underthesea torch transformers \
            peft datasets accelerate matplotlib seaborn joblib
```

## Hướng dẫn chạy Notebooks

Mở Jupyter Lab hoặc Jupyter Notebook, sau đó chạy lần lượt theo thứ tự:

### Notebook 00 — Tiền xử lý dữ liệu (`00_Tien_Xu_Ly.ipynb`)

Thực hiện các bước làm sạch và chuẩn hóa văn bản:
- Chuẩn hóa Unicode (NFC), chuyển về chữ thường
- Xóa ký tự lặp lại kéo dài (ví dụ: `dạaaaaa` → `dạa`)
- Ánh xạ teencode và từ viết tắt qua từ điển 478 quy tắc
- Loại bỏ dấu câu và ký hiệu đặc biệt
- Tách từ tiếng Việt bằng thư viện `underthesea`

Kết quả được lưu ra thư mục `fn/`.

### Notebook 01 — Mô hình Baseline (`01_Baseline_NB_SVM.ipynb`)

Xây dựng các mô hình học máy truyền thống:
- Trích xuất đặc trưng TF-IDF (max 5000 từ, ngram 1–2)
- Huấn luyện Multinomial Naive Bayes
- Huấn luyện Linear SVM (LinearSVC)
- Đánh giá Accuracy, Precision, Recall, F1-Score trên tập Test
- Lưu mô hình ra thư mục `pkl/`

### Notebook 02 — Mô hình Bi-LSTM (`02_BiLSTM.ipynb`)

Xây dựng và huấn luyện mạng Bidirectional LSTM:
- Xây dựng từ điển từ vựng từ tập Train
- Mã hóa văn bản thành chuỗi token ID, padding về độ dài 100
- Kiến trúc: Embedding → Spatial Dropout → Bi-LSTM → Global Average Pooling → FC
- Áp dụng **Class Weights** để xử lý mất cân bằng dữ liệu
- Huấn luyện với cơ chế **Early Stopping**

### Notebook 03 — Fine-tuning PhoBERT (`03_PhoBERT.ipynb`)

Tinh chỉnh mô hình ngôn ngữ PhoBERT-base:
- Tokenization bằng PhoBERT-base Tokenizer (max_length=256)
- **Full Fine-tuning** toàn bộ 135 triệu tham số
- Sử dụng `transformers.Trainer` với cơ chế lưu checkpoint tốt nhất
- Đánh giá kết quả và so sánh tổng thể 4 mô hình

> **Lưu ý:** Bước huấn luyện PhoBERT yêu cầu GPU có tối thiểu 8 GB VRAM. Khuyến nghị sử dụng Google Colab (T4 GPU).

## Kết quả thực nghiệm

| Mô hình | Accuracy | F1-Score (Macro) |
|---|---|---|
| Multinomial Naive Bayes | 86,42% | 59,94% |
| SVM (LinearSVC) | 88,69% | 63,97% |
| Bidirectional LSTM | 83,07% | 69,58% |
| **PhoBERT (Full Fine-tuning)** | **94,92%** | **86,10%** |

## Công nghệ sử dụng

| Thành phần | Thư viện / Công cụ |
|---|---|
| Xử lý dữ liệu | `pandas`, `numpy`, `regex` |
| Tách từ tiếng Việt | `underthesea` |
| Học máy truyền thống | `scikit-learn` |
| Học sâu | `PyTorch` |
| Mô hình ngôn ngữ | `transformers` (Hugging Face), `PhoBERT-base` |
| Tinh chỉnh hiệu quả | `peft` (LoRA) |
| Trực quan hóa | `matplotlib`, `seaborn` |
