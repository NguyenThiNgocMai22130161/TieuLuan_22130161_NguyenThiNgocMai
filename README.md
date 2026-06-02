# 🎓 Đồ án Tốt nghiệp: Nghiên cứu Phương pháp Học sâu cho bài toán Phân tích Cảm xúc tiếng Việt

Dự án này tập trung vào bài toán **Phân tích Cảm xúc phản hồi của Sinh viên (Student Feedback Sentiment Analysis)** tiếng Việt, sử dụng bộ ngữ liệu chuẩn **UIT-VSFC**. Đồ án nghiên cứu, thực nghiệm và so sánh toàn diện giữa các phương pháp học máy truyền thống (Multinomial Naive Bayes, SVM) kết hợp trích xuất đặc trưng TF-IDF với các kiến trúc học sâu hiện đại bao gồm Bi-directional LSTM và mô hình ngôn ngữ tiền huấn luyện PhoBERT (tinh chỉnh qua LoRA hoặc Full Fine-tuning).

---

## 👤 Thông tin Thực hiện
* **Sinh viên thực hiện:** Nguyễn Thị Ngọc Mai
* **Mã số sinh viên:** 22130161
* **Lớp:** DH22DTB
* **Trường:** Đại học Nông Lâm TP.HCM
* **Giảng viên hướng dẫn:** TS. Nguyễn Văn Dũ

---

## 📊 Bộ dữ liệu (Dataset)
Sử dụng bộ ngữ liệu **UIT-VSFC** (Vietnamese Students’ Feedback Corpus) gồm hơn 16.000 câu phản hồi ngắn của sinh viên được gán nhãn 3 lớp cảm xúc:
* **0 - Tiêu cực (Negative)**
* **1 - Trung tính (Neutral)**
* **2 - Tích cực (Positive)**

---

## 📁 Cấu trúc Thư mục Dự án

```text
code/
│
├── data/                  # Dữ liệu gốc (.csv) và từ điển teencode
│   ├── train.csv          # Dữ liệu huấn luyện gốc
│   ├── validation.csv     # Dữ liệu kiểm định gốc
│   ├── test.csv           # Dữ liệu kiểm thử gốc
│   ├── teencode.txt       # Từ điển ánh xạ teencode & viết tắt (478 quy tắc)
│   ├── new_review.csv
│   └── new_review_labeled.csv
│
├── fn/                    # Dữ liệu đã làm sạch và tách từ (sau khi chạy preprocessing.py)
│   ├── cleaned_train.csv
│   ├── cleaned_validation.csv
│   └── cleaned_test.csv
│
├── pkl/                   # Lưu trữ model đã huấn luyện và từ điển
│   ├── tfidf_vectorizer.pkl
│   ├── naive_bayes_model.pkl
│   ├── svm_best_model.pkl
│   ├── lstm_vocab.json
│   └── best_lstm_model_correct.pth
│
├── notebooks/             # Chứa các file Jupyter Notebook (.ipynb) phục vụ nghiên cứu thử nghiệm
│   ├── 01_Test_Data.ipynb
│   ├── 02_Baseline_TF-IDF_SVM-NB.ipynb
│   ├── 03_Test_Tien_xu_ly.ipynb
│   ├── 01_LSTM.ipynb
│   ├── tl_02.ipynb
│   └── vehinh.ipynb
│
├── image/                 # Chứa các hình vẽ ma trận nhầm lẫn & biểu đồ (phục vụ viết LaTeX)
│   ├── data_distribution.png
│   ├── preprocessing_flowchart.png
│   ├── system_architecture.png
│   ├── cm_naive_bayes.png
│   ├── cm_svm.png
│   ├── cm_lstm.png
│   ├── cm_phobert.png
│   ├── model_comparison.png
│   ├── lstm_learning_curve.png
│   └── phobert_learning_curve.png
│
├── src/                   # Mã nguồn Python độc lập (.py)
│   ├── preprocessing.py   # Quy trình tiền xử lý và tách từ tiếng Việt
│   ├── baselines.py       # Huấn luyện & Đánh giá Multinomial NB và SVM
│   ├── lstm.py            # Định nghĩa, huấn luyện & Đánh giá mạng Bi-LSTM
│   ├── train_phobert.py   # Huấn luyện tinh chỉnh PhoBERT (LoRA / Full Fine-tuning)
│   ├── generate_plots.py  # Đánh giá tổng thể và tự động sinh toàn bộ biểu đồ
│   └── predict.py         # Chương trình chạy thử nghiệm dự đoán cảm xúc (CLI/Interactive)
│
├── .gitignore             # Cấu hình bỏ qua tệp tin rác & checkpoints nặng khi đẩy lên Git
└── README.md              # Hướng dẫn sử dụng dự án này
```

---

## 🚀 Hướng dẫn Cài đặt & Chạy Chương trình

### 1. Thiết lập Môi trường ảo (Virtual Environment)
Mở terminal tại thư mục dự án và thực hiện các bước:

```bash
# Tạo môi trường ảo
python3 -m venv env_doan

# Kích hoạt môi trường ảo (trên macOS/Linux)
source env_doan/bin/activate
```

Sau khi kích hoạt, tiến hành nâng cấp `pip` và cài đặt các thư viện lõi:

```bash
python3 -m pip install --upgrade pip
pip install pandas numpy scikit-learn underthesea regex torch transformers peft datasets accelerate matplotlib seaborn joblib
```

---

### 2. Hướng dẫn Chạy mã nguồn trong thư mục `src/`

Mọi tệp tin mã nguồn đều được thiết kế để chạy trực tiếp từ dòng lệnh. Vui lòng chạy theo thứ tự sau để tái lập kết quả đồ án:

#### Bước A: Tiền xử lý dữ liệu (Unicode, Teencode, Tách từ tiếng Việt)
Script này sẽ làm sạch cột bình luận phản hồi bằng các quy tắc Unicode NFC chuẩn, sửa teencode bằng từ điển và tách từ thông qua thư viện `underthesea`, sau đó lưu kết quả ra thư mục `fn/`:
```bash
python src/preprocessing.py
```

#### Bước B: Huấn luyện và Đánh giá các mô hình Baseline (Naive Bayes & SVM)
Script này sẽ nạp dữ liệu sạch, trích xuất đặc trưng TF-IDF, huấn luyện mô hình Multinomial Naive Bayes và Linear Support Vector Machine, hiển thị báo cáo hiệu năng chi tiết và lưu mô hình vào `pkl/`:
```bash
python src/baselines.py
```

#### Bước C: Huấn luyện và Đánh giá mô hình Học sâu Bi-LSTM
Script này tự động xây dựng từ vựng, tính toán Class Weights (trọng số lớp chống mất cân bằng dữ liệu), huấn luyện mạng Bi-LSTM tuần tự bằng PyTorch kết hợp cơ chế Early Stopping để chống quá khớp:
```bash
python src/lstm.py
```

#### Bước D: Tinh chỉnh mô hình ngôn ngữ PhoBERT (Tùy chọn)
Chạy script để tinh chỉnh mô hình PhoBERT-base trên bộ dữ liệu UIT-VSFC sử dụng thư viện Transformers. Mặc định chạy phương pháp LoRA (PEFT) tiết kiệm tài nguyên:
```bash
# Huấn luyện PhoBERT sử dụng kỹ thuật LoRA (Mặc định)
python src/train_phobert.py --method lora --epochs 5

# Huấn luyện PhoBERT tinh chỉnh toàn bộ trọng số (Full Fine-tuning)
python src/train_phobert.py --method full --epochs 5 --lr 2e-5 --batch_size 16
```

#### Bước E: Tự động Sinh toàn bộ Biểu đồ kết quả
Đánh giá lại toàn bộ các mô hình và tự động sinh các hình ảnh ma trận nhầm lẫn (Confusion Matrix) kèm biểu đồ so sánh tổng thể và đường cong học tập, lưu thẳng vào thư mục `image/`:
```bash
python src/generate_plots.py
```

---

## 🔮 Chạy thử nghiệm Dự đoán Cảm xúc cho câu mới
Dự án cung cấp một chương trình dự đoán cảm xúc cho phép cậu nhập một câu văn phản hồi bất kỳ và nhận kết quả phân loại (Tích cực, Tiêu cực, hay Trung tính) kèm mức độ tin cậy của mô hình.

#### Chạy dự đoán trực tiếp cho một câu:
```bash
python src/predict.py --model svm --text "Thầy dạy rất nhiệt tình, slide bài giảng chi tiết dễ hiểu."
```

#### Chạy ở chế độ Tương tác (Interactive Mode) nhập liên tục:
```bash
python src/predict.py --model svm
```
*(Gõ `exit` để thoát chương trình tương tác).*
