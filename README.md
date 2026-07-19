# Action-SigLIP: Hyperparameter Tuning Module

Branch này chứa các file cấu hình và kịch bản (scripts) tối ưu hoá cho mục đích **Hyperparameter Tuning** mô hình Video Understanding (kết hợp SigLIP, LoRA và Hybrid Temporal Adapter).

## Cấu trúc thư mục hiện tại
- `model_tuning.py`: Chứa class mô hình SigLIP tích hợp LoRA động (`target_modules`) và Temporal Adapter.
- `dataset.py`: Logic tiền xử lý, phân chia dữ liệu Train/Test theo Group (chống Data Leakage) và cơ chế đọc video Fallback (Decord -> OpenCV).
- `tuning.py`: Vòng lặp Tuning chính, tự động lặp qua các cấu hình LoRA, đánh giá Validation Accuracy, quản lý VRAM nghiêm ngặt và xuất báo cáo.
- `requirements.txt`: Các thư viện cần thiết.
- `.env`: File cấu hình môi trường chứa đường dẫn tới dataset.

## Hướng dẫn cài đặt và chạy

### 1. Cài đặt thư viện
Tạo môi trường ảo (khuyến nghị) và cài đặt thư viện:
```bash
python -m venv venv
# Active venv (Windows):
.\venv\Scripts\activate
# Cài đặt requirements
pip install -r requirements.txt
```

### 2. Cấu hình biến môi trường
Tạo file `.env` ở thư mục gốc của project (nếu chưa có) và trỏ biến `UCF11_DATA_DIR` tới thư mục chứa dữ liệu UCF11/UCF50 của bạn:
```env
UCF11_DATA_DIR=C:\path\to\your\dataset\UCF11_updated_mpg
```

### 3. Chạy Hyperparameter Tuning (Full)
Khi mọi thứ đã sẵn sàng, hãy chạy file chính. Quá trình này sẽ sinh ra file `ucf11_lora_tuning_results.csv` và biểu đồ `tuning_heatmap.png`.
```bash
python tuning.py
```