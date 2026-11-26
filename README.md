# Định giá và Phát hiện Tin đăng Bất động sản Lệch giá tại Việt Nam (CS114 - Nhóm 8)

Hệ thống ML/NLP kết hợp dữ liệu bảng, TF-IDF, PhoBERT (`vinai/phobert-base`), và Conformalized Quantile Regression (CQR, mục tiêu độ phủ 90%) trên bộ dữ liệu `tinixai/vietnam-real-estates` (3.500.744 tin đăng, 06/2025–03/2026).

## Tổ chức & Phân công 4 Thành viên
- **TV1 (`src/data/`, `src/pipelines/prepare_data.py`)**: Tải 10 shard có kiểm tra SHA256, tạo `row_id`, bộ tách giá/diện tích tiếng Việt độc lập, lọc tin bán/cho thuê/nhiễu, chuẩn hóa địa giới & hướng, khử trùng MinHash không dùng giá, chia tập Kịch bản A (thời gian) & Kịch bản B (quận/huyện chưa gặp), xuất audit & `data_report.md`.
- **TV2 (`src/features/tabular_features.py`, `tfidf_features.py`, `src/models/m0*`, `m1*`, `m2*`, `m4*`)**: Baseline M0, CatBoost bảng M1 (+ phân vị 5%/95%), TF-IDF Ridge M2, TF-IDF SVD + CatBoost M4, feature importance SHAP.
- **TV3 (`src/data/text_masker.py`, `src/features/phobert_embedder.py`, `src/models/m3*`, `m5*`, `m6*`, `src/pipelines/extract_embeddings.py`)**: Bộ ẩn mọi số tiền (`MASK_GIA`), tách từ `pyvi`, trích xuất & cache embedding PhoBERT (`max_length=256`, `fp16`) theo `row_id`, mô hình M3, M5, đối chứng rò rỉ E3 và M6 fine-tune tùy chọn.
- **TV4 (`src/eval/`, `src/pipelines/train_eval.py`, `build_reports.py`, `app/streamlit_app.py`)**: Kiểm tra chống rò rỉ split, bộ metrics chuẩn từ prediction Parquet, conformal C0–C3 (+ calibration gần nhất vs cũ), gắn cờ G1/G2, audit cờ E6, biểu đồ, báo cáo và demo Streamlit.

## Hướng dẫn Chạy Nhanh (Local ổ `E:` hoặc Kaggle)

### 1. Chạy trên máy Local (mẫu kiểm chứng `configs/local_smoke.yaml` hoặc full RTX 3050 `configs/local_rtx3050_full.yaml`)
```powershell
$env:HF_HOME="E:\hf_cache"
.venv\Scripts\python.exe -m pytest -v
.venv\Scripts\python.exe -m src.pipelines.prepare_data --config configs/local_smoke.yaml
.venv\Scripts\python.exe -m src.pipelines.extract_embeddings --config configs/local_smoke.yaml
.venv\Scripts\python.exe -m src.pipelines.train_eval --config configs/local_smoke.yaml
.venv\Scripts\python.exe -m src.pipelines.build_reports --config configs/local_smoke.yaml
.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

### 2. Chạy Full Option A trên Kaggle GPU (T4 / P100)
Bật Internet cho Kaggle Notebook rồi chạy toàn bộ các cell trong [`notebooks/kaggle_runner.ipynb`](notebooks/kaggle_runner.ipynb) với `configs/kaggle_full.yaml`: cell đầu tự `git clone` nhánh `feat/full-pipeline` (đổi bằng `VN_RE_BRANCH` / `VN_RE_REPO_URL`, hoặc tự upload repo và đặt `VN_RE_PROJECT_ROOT`). Sau khi chạy xong, tải `models/bundle_a/` và `outputs/` về thư mục dự án local để mở Streamlit demo với trọng số full.

### 3. Phiên bản & chạy lại
- Phiên bản thư viện đã kiểm thử: [`requirements-tested.txt`](requirements-tested.txt) (`pyproject.toml` chỉ ghi cận dưới).
- Khi đổi luật ẩn giá / tách giá (`src/data/text_masker.py`, `price_area_parser.py`), tăng `rules_version` và `text_version` trong `configs/base.yaml` rồi chạy lại **cả ba bước** `prepare_data` → `extract_embeddings` → `train_eval`: `train_eval` từ chối cache embedding có `text_version` khác.
- `train_eval` ghi `winner_a/winner_b` (MAE_log validation thấp nhất, mọi mô hình) và `conformal_model_a/b` (mô hình dùng cho conformal + bundle, chỉ M1/M4/M5 vì cần hồi quy phân vị) trong `outputs/metrics/run_summary.json`.
