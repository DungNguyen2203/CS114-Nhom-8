# Spec Thiết Kế Hệ Thống: Định Giá và Phát Hiện Tin Đăng Bất Động Sản Lệch Giá Tại Việt Nam

**Ngày:** 2026-11-25  
**Tài liệu gốc:** [De_cuong_do_an_dinh_gia_bat_dong_san_Viet_Nam.docx](../../De_cuong_do_an_dinh_gia_bat_dong_san_Viet_Nam.docx)  
**Phạm vi:** Triển khai đầy đủ nhiệm vụ của 4 thành viên (TV1–TV4), tương thích chạy song song trên máy local (`E:` + RTX 3050 4 GB VRAM) ở chế độ medium sample kiểm chứng và trên Kaggle (T4/P100 16 GB VRAM) ở chế độ full Option A (100k–200k mẫu).

---

## 1. Mục tiêu, Phạm vi & Ràng buộc Toàn cục

### 1.1 Mục tiêu cốt lõi
- Dự đoán **giá chào bán** bất động sản theo thang `log(price_vnd)` từ thuộc tính bảng và mô tả tiếng Việt (không phải giá giao dịch thực tế hay thẩm định giá).
- Xây dựng lớp khoảng dự đoán **Conformal Prediction** (mục tiêu độ phủ danh nghĩa 90%, `alpha = 0.10`, kiểm tra thêm `0.20`) và hệ thống gắn cờ tin lệch giá / mâu thuẫn mô tả (`G1`, `G2`).
- Đảm bảo tái lập hoàn toàn (`seed` cố định, `row_id` bất biến, lưu SHA/checksum dữ liệu, tách chặt `train / val / cal / test`, không rò rỉ giá trong văn bản hay thống kê toàn cục).

### 1.2 Ràng buộc đã kiểm chứng từ dữ liệu & phần cứng
1. **Nguồn dữ liệu:** `tinixai/vietnam-real-estates` trên Hugging Face, revision `ca3fedcbb089bf65f7e4cfb03316cce4bd0d779f`, gồm 10 file Parquet (`shard_0000.parquet`–`shard_0009.parquet`), `3,500,744` dòng, `published_at` từ `06/2025` đến `03/2026`.
2. **Loại hình bất động sản thực tế:** Cột `property_type_name` không thiếu và có đúng 5 giá trị: `Nhà`, `Đất`, `Căn hộ chung cư`, `Biệt thự/Nhà liền kề`, `Shophouse`.
3. **Nhiễu cực trị & khuyết thiếu bảng:** `price` có giá trị `0` / thiếu (`~6,24%`) / ngoại lệ tới `2.66e16`; `area` lên tới `8.2e7`; `frontage_width`, `road_width`, `floor_count`, `bedroom_count`, `bathroom_count`, `house_depth` chứa cực trị phi lý và tỷ lệ thiếu cao (`45%–98%`); `house_direction` (31 giá trị) và `balcony_direction` (16 giá trị) lẫn dấu gạch ngang và đa hướng. Bắt buộc chuẩn hóa hướng, xử lý giá trị ngoài miền hợp lý theo tập `train`, và giữ cờ missing indicator.
4. **Lệch địa giới cũ/mới & tiền thuê trong tin bán:** Một số tin từ `07/2025` trở đi có cột bảng giữ tên trước sáp nhập (`Bình Dương / Dĩ An`, `Vũng Tàu / 8`) trong khi `description` ghi tên mới (`TP. HCM`, `Phường Tam Thắng`); nhiều tin bán hợp lệ ghi kèm dòng tiền thuê (`HDT 20tr/t`, `đang cho thuê 20tr/tháng`). Parser phải phân biệt tin thuần cho thuê với tin bán có hợp đồng thuê, và bộ ẩn số tiền phải che toàn bộ các khoản tiền đó.
5. **PhoBERT (`vinai/phobert-base`, revision `01daacda68afe13d83023d16ec647239e344a1e6`):** `max_position_embeddings = 258`, nhưng `tokenizer.model_max_length` mặc định không giới hạn (`1e30`). Bắt buộc truyền tường minh `max_length=256, truncation=True`, tách từ tiếng Việt trước tokenization, và ánh xạ placeholder ẩn giá sang `<mask>` (1 token có sẵn trong từ vựng PhoBERT) để tránh cắt mảnh BPE hay khởi tạo ngẫu nhiên embedding trên encoder đóng băng.
6. **Lưu trữ & VRAM:** Trên máy local, ổ `C:` chỉ còn ~3,8 GB trống nên `.venv`, cache Hugging Face (`HF_HOME`), dữ liệu thô/đã xử lý, embedding cache và model artifacts đều đặt trên ổ `E:`. Đọc 10 shard theo từng shard bằng `pyarrow` để đỉnh RAM giữ trong khoảng `4–6 GB`. GPU RTX 3050 4 GB dùng `fp16` + `batch_size=16`; profile Kaggle T4/P100 16 GB dùng `fp16` + `batch_size=64`.

---

## 2. Cấu trúc Dự án & Cấu hình Đa Môi trường

```text
E:\CS114 - Nhóm 8\
├── pyproject.toml
├── configs/
│   ├── base.yaml                   # Siêu tham số, quy tắc lọc, seeds [42, 17, 2026], đường dẫn tương đối
│   ├── local_smoke.yaml            # Profile local trên E: (full TV1 audit/split + sample 30k train / 5k val-cal-test, RTX 3050 fp16 bs=16)
│   ├── local_rtx3050_full.yaml     # Profile local 100k train / 20k+ eval nếu chạy full tại máy
│   ├── kaggle_full.yaml            # Profile Kaggle T4/P100 (100k-200k train / 20k+ val-cal-test, bs=64, tùy chọn M6)
│   └── location_aliases_v1.yaml    # Bảng ánh xạ chuẩn hóa tỉnh/quận/phường/hướng có phiên bản
├── data/
│   ├── raw/                        # 10 shard gốc + manifest.json (revision + sha256)
│   └── processed/                  # clean.parquet, noisy_flagged.parquet, rental_unknown.parquet, splits/, embeddings/
├── src/
│   ├── __init__.py
│   ├── config.py                   # Loader config + env paths (PROJECT_ROOT, HF_HOME)
│   ├── data/                       # TV1 (+ masker phối hợp TV3)
│   │   ├── download.py
│   │   ├── text_normalizer.py
│   │   ├── price_area_parser.py
│   │   ├── text_masker.py
│   │   ├── location_normalizer.py
│   │   ├── tabular_cleaner.py
│   │   ├── dedup.py
│   │   └── splits.py
│   ├── features/                   # TV2 + TV3
│   │   ├── tabular_features.py
│   │   ├── tfidf_features.py
│   │   └── phobert_embedder.py
│   ├── models/                     # TV2 + TV3
│   │   ├── contract.py             # Schema chuẩn prediction Parquet + base helpers
│   │   ├── m0_median.py
│   │   ├── m1_catboost.py
│   │   ├── m2_tfidf_ridge.py
│   │   ├── m3_phobert_ridge.py
│   │   ├── m4_tfidf_catboost.py
│   │   ├── m5_phobert_catboost.py
│   │   └── m6_finetune.py          # Tùy chọn Kaggle-only sau khi E1-E6 hoàn thành
│   ├── eval/                       # TV4
│   │   ├── leakage_checks.py
│   │   ├── metrics.py
│   │   ├── conformal.py
│   │   ├── flagger.py
│   │   ├── error_analysis.py
│   │   └── plots.py
│   └── pipelines/                  # CLI orchestration end-to-end
│       ├── prepare_data.py         # Bước TV1
│       ├── extract_embeddings.py   # Bước TV3
│       ├── train_eval.py           # Bước TV2 + TV3 + TV4 (E1-E6)
│       └── build_reports.py        # Xuất bảng báo cáo, audit samples, hình vẽ
├── notebooks/
│   └── kaggle_runner.ipynb         # Notebook mỏng gọi CLI cho Kaggle Option A
├── tests/                          # Unit & integration tests (pytest)
├── outputs/                        # predictions/, metrics/, audits/, figures/
├── models/                         # Saved pipelines, CatBoost/Ridge/PCA/SVD, retrieval index từ train
├── app/
│   └── streamlit_app.py            # Demo TV4
└── reports/                        # data_report.md, data_dictionary.md, experiment_report.md, slide_outline.md
```

---

## 3. Thiết kế Chi tiết theo 4 Thành viên

### 3.1 TV1 — Dữ liệu, Parser, Chuẩn hóa & Chia tập (`src/data/`)

#### A. Tải & Quản lý Phiên bản (`download.py`)
- Tải đủ 10 shard bằng `huggingface_hub.hf_hub_download` với `revision="ca3fedcbb089bf65f7e4cfb03316cce4bd0d779f"`, `local_dir` nằm dưới `data/raw/`.
- Tính SHA256 và số dòng từng shard (`pyarrow.parquet.ParquetFile.metadata.num_rows`), ghi `data/raw/manifest.json` cùng `rules_version = "v1.0.0"`.
- Quy ước `row_id`: `f"{shard_idx:02d}_{row_idx:07d}"` (`shard_idx` từ `00` đến `09`, `row_idx` là chỉ số dòng trong shard).

#### B. Chuẩn hóa văn bản & Che PII (`text_normalizer.py`)
- `normalize_text(text)`: Unicode NFC, `html.unescape`, xóa HTML tags còn sót, chuẩn hóa khoảng trắng/ký tự điều khiển.
- `redact_pii(text)`: Thay thế số điện thoại (kể cả dạng có dấu chấm/khoảng trắng hoặc đã che một phần), email, link/tài khoản Zalo/Facebook, cụm `LH: <tên>` / `liên hệ: <tên>` và số nhà cụ thể (`số 123/45 đường...` → `số [ĐỊA_CHỈ] đường...`) khi xuất tập audit và tin tham khảo trong demo.

#### C. Bộ tách giá & diện tích độc lập (`price_area_parser.py`)
- Đầu vào: `(name_norm, description_norm, tabular_area)`. **Tuyệt đối không đọc cột `price` trong hàm tách giá**; việc so khớp với `price` thực hiện ở hàm đánh giá nhãn riêng biệt.
- Nhận diện giao dịch (`tx_type` ∈ `{"sale", "rent", "unknown"}`):
  - Tín hiệu cho thuê thuần: tiêu đề/đoạn mở đầu chứa `cho thuê`, `cần thuê`, `sang nhượng mặt bằng`, hoặc mọi mức giá đều mang đơn vị `/tháng`, `/th`, `/ngày`, `/năm` mà không có giá bán.
  - Ngoại lệ quan trọng (đã xác minh trong dữ liệu thật): tin bán hợp lệ có nhắc doanh thu thuê kèm theo (`đang có HDT 20tr/t`, `cho thuê sẵn 25tr/tháng`, `dòng tiền cho thuê ...`) vẫn giữ `tx_type = "sale"` nếu có ngữ cảnh bán rõ (`bán`, `chính chủ bán`, `cần bán`, hoặc có giá bán tổng/m² tiền tỷ).
  - Tín hiệu gộp nhiều BĐS (`is_bundle = True`): `bán cả cặp`, `2 căn liền`, `3 lô`, `tổng X lô` khi không rõ giá từng căn.
- Trích xuất số tiền với nhãn vai trò (`role` ∈ `{"total_sale", "unit_per_m2", "rental_income", "deposit_or_discount", "other"}`), `value_vnd`, `is_approx` (`nhỉnh`, `hơn`, `~`, `x` như `8,x tỷ`, `từ`), và `confidence` (`"high"`, `"medium"`, `"low"`):
  - Hỗ trợ các mẫu: `X tỷ Y triệu`, `X tỷ Y` (`9 tỷ 1` / `9 tỷ 100`), `X,Y tỷ` / `X.Y tỷ` / `ty` / `tỉ`, `X triệu/m2` / `tr/m²`, `X tr` / `X triệu`, số tiền bằng chữ tiếng Việt thông dụng (`một tỷ rưỡi`, `hai tỷ ba`, `năm trăm triệu`).
  - Hỗ trợ phát hiện lỗi đánh dấu thập phân kiểu `2.050 tỷ` (= 2,05 tỷ nếu số sau dấu chấm là 3 chữ số với giá trị hàng nghìn nhỏ hơn 100 hay xung đột bậc độ lớn với `price`).
- Trích xuất diện tích từ văn bản (`text_area_m2`) từ các mẫu `dt 60m2`, `diện tích: 48 m²`, `4m x 12m` (tính `4 * 12 = 48`), bỏ qua chiều rộng đường/mặt tiền (`đường 8m`, `mt 5m`).
- Hàm so khớp nhãn `reconcile_labels(row, parsed)`:
  - Nếu `tx_type == "rent"` → `split_bucket = "rental"`.
  - Nếu `price` rỗng / `<= 0` hoặc văn bản ghi `thỏa thuận` / `liên hệ giá` không có số → `split_bucket = "missing_or_negotiable"`.
  - Nếu `is_bundle` hoặc `tx_type == "unknown"` → `split_bucket = "multi_or_unknown"`.
  - Kiểm tra miền giá trị vật lý cơ bản của `price` và `area` (`area` ngoài `[5.0, 50000.0]`, `price` ngoài `[5e7, 2e12]`, hoặc `price/area` ngoài `[5e5, 3e9]` VND/m² sơ bộ) → `noisy_extreme_domain`.
  - So khớp `text_sale_price` (quy đổi từ `total_sale` hoặc `unit_per_m2 * area` khi có) với `price`:
    - Độ lệch tương đối `ratio = max(p_tab / p_txt, p_txt / p_tab)`:
    - Nếu lệch gần bội số `10x`, `100x`, `1000x` (`ratio >= 5.0`) → `noisy_order_of_magnitude`.
    - Nếu không xấp xỉ và lệch vượt ngưỡng (`ratio > 1.35`, hoặc với `is_approx=True` vượt `1.60`) → `noisy_price_mismatch`.
    - Nếu `text_area_m2` có độ tin cậy cao và lệch `area` bảng vượt `30%` (trừ trường hợp diện tích sàn vs diện tích đất khi có nhiều tầng) → gắn cờ cảnh báo `area_mismatch_flag`.
  - Kết quả sạch (`clean_consistent` hoặc `clean_no_text_price`) đi vào `clean.parquet`; nhóm mâu thuẫn (`noisy_*`) đi vào `noisy_flagged.parquet`; nhóm `rental` và `multi_or_unknown` lưu riêng vào `rental_unknown.parquet` kèm bảng thống kê phễu lọc đầy đủ.

#### D. Chuẩn hóa Vị trí & Thuộc tính Bảng (`location_normalizer.py`, `tabular_cleaner.py`)
- `location_aliases_v1.yaml`: ánh xạ chuẩn hóa `province_name` (63 tỉnh chuẩn), loại bỏ tiền tố dư thừa hoặc thống nhất dạng số (`"8"` ↔ `"Quận 8"`, `"5"` ↔ `"Phường 5"`, `"p. Tân Thành"` → `"Tân Thành"`), chuẩn hóa khoảng trắng/viết hoa.
- Chuẩn hóa hướng nhà & ban công (`house_direction`, `balcony_direction`) về 8 hướng chính (`Đông`, `Tây`, `Nam`, `Bắc`, `Đông Nam`, `Đông Bắc`, `Tây Nam`, `Tây Bắc`), `Đa hướng` (khi liệt kê nhiều hướng), hoặc `Thiếu`.
- Làm sạch thuộc tính số (`tabular_cleaner.py`): với `frontage_width` (`[1, 100]` m), `house_depth` (`[1, 200]` m), `road_width` (`[0.5, 100]` m), `floor_count` (`[1, 60]`), `bedroom_count` (`[1, 50]`), `bathroom_count` (`[1, 50]`) — giá trị ngoài miền hợp lý đặt thành `NaN` và bật cờ `*_was_outlier` + `*_is_missing` (quy tắc thuần miền dữ liệu, không phụ thuộc nhãn test).
- Thống kê độ lệch địa giới cũ/mới (tỷ lệ tin có tỉnh/quận ở cột bảng khác với tên tỉnh/phường sau 07/2025 xuất hiện trong `description`) đưa vào `data_report.md`.
- **Tách biệt lọc cứng vs ngưỡng học từ train:** Bộ lọc vật lý/nghiệp vụ cố định (`tx_type == "sale"`, nhãn nhất quán, `area in [5, 50000]`, `price in [5e7, 2e12]`, `price/m² in [5e5, 3e9]`) được áp dụng **đồng nhất trước khi chia tập** cho cả `train / val / cal / test` nhằm giữ giả định hoán đổi (exchangeability) cho conformal prediction. Riêng ngưỡng phân vị `price/m²` theo `property_type_name` (`[0.5%, 99.5%]`) chỉ **học trên `train` của từng kịch bản** để gắn cờ kiểm tra/thống kê số mẫu ngoài miền trên `val / cal / test` (tuyệt đối không xóa điểm `test` khó để làm đẹp kết quả).

#### E. Bộ ẩn số tiền (`text_masker.py`, phối hợp TV1 & TV3)
- Che tất cả thực thể tiền (`X tỷ Y triệu`, `X,Y tỷ`, `X tr/m2`, `HDT 20tr/t`, `giảm 500tr`, `cần 400tr`, số tiền viết bằng chữ, các ký hiệu viết tắt `tỷ/tỉ/ty/tr/triệu/ngàn/k/đồng/vnd` đi kèm chữ số hoặc `x` ẩn số như `8,x tỷ`) bằng một token duy nhất: `MASK_GIA`.
- **Bảo toàn số phi tiền tệ:** Giữ nguyên diện tích (`60m2`, `4x15m`), số tầng (`5 tầng`, `5T`), số phòng (`3PN 2WC`), chiều rộng ngõ/mặt tiền (`mt 4.6m`, `ngõ 8m`), số ngõ/địa chỉ đường (`ngõ 192`, `VĐ 2.5`).
- Cung cấp hàm kiểm tra tự động `find_leaked_money(text) -> list[str]` trả về mọi cụm tiền nghi còn sót (mục tiêu = 0 trên toàn bộ mẫu kiểm tra `audit_masking_100.csv`).

#### F. Khử trùng & Chia tập (`dedup.py`, `splits.py`)
- Tạo `dup_group_id` từ: `(text_masked_norm, province_norm, district_norm, round(area), floor_count, bedroom_count)` — **không dùng `price`**.
  - Giai đoạn 1: Exact hash trên tuple chuẩn hóa.
  - Giai đoạn 2: MinHash LSH (`datasketch`, 64 permutations, Jaccard ≥ 0.85 trên shingle 3-gram từ của `text_masked_norm`) trong cùng block `(province_norm, district_norm, property_type_name, area_bin)` để chạy nhanh và chính xác trên hàng triệu dòng.
  - Thống kê riêng các `dup_group_id` có `price` biến động giữa các lần đăng lại.
- **Kịch bản A (Theo thời gian):**
  - Trích `month = YYYY-MM` từ `published_at`.
  - Mặc định: `train` (`2025-06`..`2025-10`), `val` (`2025-11`..`2025-12`), `cal` (`2026-01`..`2026-02`), `test` (`2026-03`) — kiểm tra sản lượng sạch từng tháng trong `prepare_data.py` và tự động ghi log nếu có tháng thiếu mẫu cần gộp liền kề.
  - Mỗi `dup_group_id` chỉ giữ **bản xuất hiện sớm nhất** (`published_at` nhỏ nhất) để tin đăng lại ở tháng sau không lọt vào `val/cal/test`.
- **Kịch bản B (Quận/huyện chưa gặp):**
  - Lấy mỗi `dup_group_id` một bản đại diện (sớm nhất) và nhóm theo thành phần liên thông của `district_key = f"{province_norm}|{district_norm}"` (loại bỏ dòng thiếu `district_norm` hoặc gộp vào bucket kiểm soát riêng; báo cáo rõ số dòng thiếu quận).
  - Dùng `GroupShuffleSplit` phân tầng theo quy mô nhóm quận (~70% train / 10% val / 10% cal / 10% test số quận/huyện).
- **Lấy mẫu ngân sách tính toán (`sample_budget`):**
  - Ở `local_smoke.yaml`: lấy mẫu phân tầng cố định theo `(province_bucket, property_type_name, month)` tối đa `30,000` train / `5,000` val / `5,000` cal / `5,000` test mỗi kịch bản.
  - Ở `kaggle_full.yaml`: tối đa `100,000` (hoặc `200,000`) train và tối thiểu `20,000` mỗi tập val/cal/test (hoặc toàn bộ nếu trong ngân sách).
- Lưu `splits/scenario_a_manifest.parquet`, `splits/scenario_b_manifest.parquet` (gồm `row_id`, `split`, `dup_group_id`, `district_key`, `month`, `y_true = log(price)`).

---

### 3.2 TV2 & TV3 — Đặc trưng, Embedding & Mô hình M0–M5 (`src/features/`, `src/models/`)

#### A. Hợp đồng Đầu ra Mô hình (`models/contract.py`)
Mọi mô hình xuất kết quả đánh giá theo schema Parquet thống nhất:
- Cột bắt buộc: `row_id: str`, `split: str` (`val` | `cal` | `test`), `scenario: str` (`A` | `B`), `model_id: str` (`M0`..`M6`), `seed: int`, `y_true: float64`, `y_pred: float64`, `q_lo: float64` (nullable), `q_hi: float64` (nullable).

#### B. Pipeline Bảng & Baseline (TV2)
- **`M0` (`m0_median.py`):** Tính trung vị `price / area` trên `train` theo `(province_norm, district_norm, property_type_name)`. Khi số mẫu `< 30`, lùi dần sang `(province_norm, property_type_name)` → `(property_type_name)` → trung vị toàn `train`. `y_pred = np.log(median_ppm2 * area)`.
- **`tabular_features.py`:** Fit **duy nhất trên `train`**:
  - Phân loại (`cat_features`): `property_type_name`, `province_norm`, `district_norm`, `ward_norm`, `street_norm`, `project_norm`, `house_direction_norm`, `balcony_direction_norm` — các giá trị hiếm (`count < min_freq_train`, mặc định `10` hoặc `20`) gộp thành `"__RARE__"` dựa trên từ điển học từ `train`.
  - Số: `area`, `log_area`, `floor_count`, `frontage_width`, `house_depth`, `road_width`, `bedroom_count`, `bathroom_count`, cùng bộ cờ `*_is_missing`, `*_was_outlier`, và từ khóa nhị phân trích từ văn bản đã ẩn giá (`has_so_hong`, `has_hem_xe_hoi`, `has_mat_tien`, `has_thang_may`, `has_full_noi_that`, `is_ban_gap`, `is_chinh_chu`).
- **`M1` (`m1_catboost.py`):** `CatBoostRegressor` thử tối đa 8 cấu hình trên `val`: `loss_function in ["RMSE", "MAE"]`, `depth in [6, 8]`, `learning_rate in [0.05, 0.1]`, `iterations=2000`, `early_stopping_rounds=100`. Chọn cấu hình có `val MAE_log` tốt nhất; chạy lặp qua `seeds = [42, 17, 2026]`. Huấn luyện cặp mô hình phân vị `Quantile:alpha=0.05` và `Quantile:alpha=0.95` trên cùng cấu hình thắng cuộc.
- **`M2` (`m2_tfidf_ridge.py`):** `TfidfVectorizer` trên văn bản đã ẩn giá (`name_masked + " " + description_masked`), `ngram_range in [(1,1), (1,2)]`, `max_features=100000`, `min_df=5`, `token_pattern=r"(?u)\b\w+\b"`; `Ridge` với `alpha in [1.0, 10.0, 100.0]`, tối ưu trên `val`.
- **`M4` (`m4_tfidf_catboost.py`):** `TruncatedSVD(n_components in [64, 128], random_state=seed)` fit trên ma trận TF-IDF của `train`, nối thêm các thành phần `svd_0..svd_k` vào đặc trưng `M1`, huấn luyện `CatBoostRegressor`.

#### C. Pipeline NLP & Kết hợp PhoBERT (TV3)
- **Tạo chuỗi đầu vào (`phobert_embedder.py`):**
  - Ghép theo thứ tự ưu tiên thông tin lõi để giảm mất mát khi cắt ở 256 token:
    `"{property_type_name}. Khu vực: {district_norm}, {province_norm}. {name_masked}. {description_masked}"`
  - Tách từ tiếng Việt bằng `pyvi.ViTokenizer.tokenize` trên CPU (đo và ghi lại thời gian tách từ/1.000 văn bản).
  - Thay thế `MASK_GIA` sau tách từ bằng `<mask>` của `PhobertTokenizer`, gọi tokenizer với `max_length=256, truncation=True, padding=True`. Ghi nhận tỷ lệ văn bản bị cắt (`truncated_ratio`).
  - Hỗ trợ tùy chọn `chunk_pooling` (chia đoạn cửa sổ trượt nếu `truncated_ratio` cao, dùng cho thử nghiệm mở rộng/so sánh).
- **Trích xuất & Cache Embedding:**
  - Dùng `vinai/phobert-base` (SHA `01daacda68afe13d83023d16ec647239e344a1e6`) ở chế độ `eval()`, `torch.no_grad()`, `torch.amp.autocast('cuda', dtype=torch.float16)` (`batch_size=16` local, `64` Kaggle), masked mean pooling theo `attention_mask`.
  - Lưu cache `data/processed/embeddings/phobert_cache.fp16.npy` + `phobert_index.parquet` (`row_id`, `offset`, `text_version`, `model_sha`). Chạy 1 lần trên **hợp (union)** các `row_id` xuất hiện trong Kịch bản A và B; mọi truy xuất embedding đều join/assert khớp đúng `row_id` (tuyệt đối không ghép theo thứ tự dòng). Ghi nhận peak VRAM (MB) và tốc độ suy luận.
- **`M3` (`m3_phobert_ridge.py`):** L2-normalize embedding rồi huấn luyện `Ridge(alpha in [1.0, 10.0, 100.0])` trên `train`, chọn trên `val`.
- **`M5` (`m5_phobert_catboost.py`):** `StandardScaler` + `PCA(n_components in [64, 128], random_state=seed)` fit strictly trên `train` embeddings, ghép với đặc trưng `M1` → `CatBoostRegressor` điểm (và huấn luyện cặp phân vị `alpha=0.05, 0.95` nếu M5 hoặc mô hình kết hợp tốt nhất thắng trên `val`, đảm bảo luôn có phân vị cho mô hình điểm tốt nhất để chạy CQR).
- **Đối chứng rò rỉ & Ablation (`E3`):** Chạy thêm M5 với: (a) văn bản **chưa ẩn giá** (tách riêng trong sub-cache nhỏ hoặc trên validation để đo mức độ phình điểm do leakage), (b) M5 chỉ dùng `name_masked` / bỏ `description`.
- **Giải thích mô hình:** Tính `permutation_importance` trên `val` cho đặc trưng bảng và tương quan phần dư với các tín hiệu văn bản (`bán gấp`, `ngộp`, `kẹt tiền`, `chính chủ`).
- **`M6` (`m6_finetune.py`, Tùy chọn):** Chỉ kích hoạt bằng cờ cấu hình trong `kaggle_full.yaml` sau khi E1–E6 hoàn tất (`lr in [1e-5, 2e-5]`, 2–3 epoch, `bf16`/`fp16`, gradient accumulation, chọn checkpoint bằng `val MAE_log`).

---

### 3.3 TV4 — Conformal, Gắn cờ, Đánh giá & Demo (`src/eval/`, `app/`)

#### A. Bộ kiểm tra rò rỉ (`leakage_checks.py`)
- Kiểm tra trên mỗi kịch bản (A, B):
  - Giao `row_id` giữa mọi cặp `(train, val, cal, test)` == `0`.
  - Giao `dup_group_id` giữa mọi cặp split == `0`.
  - Kịch bản B: giao `district_key` giữa mọi cặp split == `0`.
  - Quét `find_leaked_money` trên 100% văn bản đã ẩn giá trong tập đánh giá (và xuất 100 mẫu chi tiết).

#### B. Bộ chỉ số (`metrics.py`)
Tính thống nhất từ prediction Parquet:
- **Điểm:** `MAE_log`, `RMSE_log`, `MdAPE_pct` (`median(|exp(y_pred)-exp(y_true)|/exp(y_true)) * 100`), `within_10_pct`, `within_20_pct`, `WAPE_vnd_pct`.
- **Khoảng:** `coverage`, `mean_width_log`, `median_ratio_hi_lo` (`median(exp(q_hi - q_lo))`), `interval_score` (Winkler score ở `alpha`), `crossing_rate_raw` (tỷ lệ `q_lo > q_hi` trước sửa cận đảo).
- **Phân rã nhóm:** theo `property_type_name`, theo `region_bucket` (`Hà Nội`, `TP.HCM`, `Các tỉnh khác`), theo `province_name` (`N >= 500`), theo `price_tier` (chia tứ phân vị giá thật trên `train`), theo `month`; xuất `worst_group_coverage` và `max_coverage_gap_from_90` cho các nhóm `N >= 500`.
- **So sánh cặp (Bootstrap):** Bootstrap theo cụm `district_key` (1.000 lần) cho khoảng tin cậy 95% của hiệu `MAE_log(model_A) - MAE_log(model_B)`.

#### C. Lớp Conformal (`conformal.py`)
Trên cấu hình mô hình điểm tốt nhất theo `val` của mỗi kịch bản:
- **`C0`:** Khoảng phân vị 5%–95% chưa hiệu chỉnh `[q_lo, q_hi]` (kèm sửa thứ tự nếu đảo cận sau khi đã ghi nhận `crossing_rate_raw`).
- **`C1`:** Split conformal chuẩn quanh `y_pred`: điểm `E_i = |y_i - y_pred_i|` trên `cal`, lấy `Q = quantile_{ceil((n+1)*(1-alpha))/n}(E)` → `[y_pred - Q, y_pred + Q]`.
- **`C2`:** Conformalized Quantile Regression (CQR): trước hiệu chỉnh, áp dụng sắp xếp an toàn `(q_lo_eff, q_hi_eff) = (min(q_lo, q_hi), max(q_lo, q_hi))` sau khi đã log `crossing_rate_raw`; trên `cal`, tính `E_i = max(q_lo_eff_i - y_i, y_i - q_hi_eff_i)`, lấy `Q = quantile_{ceil((n+1)*(1-alpha))/n}(E)` → `[q_lo_eff - Q, q_hi_eff + Q]`.
- **`C3`:** Group-conditional CQR theo `(property_type_name, price_tier_pred)` (dựa trên `y_pred` để không rò rỉ nhãn!). Nếu nhóm trên `cal` có `< 500` mẫu, tự động fallback về ngưỡng `Q` của `property_type_name` (nếu `>= 500`) hoặc ngưỡng `Q` toàn cục của `C2`.
- **Thí nghiệm `E5` (Stale vs Recent Calibration):** So sánh `C2` hiệu chỉnh trên `cal` sát `test` (`01–02/2026`) với `C2` hiệu chỉnh trên lát cắt cũ cuối giai đoạn `train` (`09–10/2025`), và kiểm tra thêm `alpha = 0.20`.

#### D. Gắn cờ Lệch giá & Audit (`flagger.py`, `error_analysis.py`)
- **`G1` (Lệch mô hình C2):** `price < exp(q_lo_c2)` → `flag_below_interval`; `price > exp(q_hi_c2)` → `flag_above_interval`.
- **`G2` (Mâu thuẫn văn bản độc lập):** kế thừa kết quả từ `price_area_parser.py` trên văn bản gốc → `flag_order_of_magnitude`, `flag_text_price_mismatch`, `flag_area_mismatch`.
- Xuất `outputs/audits/audit_flags_300_plus_300.csv` (lấy trên `test` Kịch bản A + tập `noisy_flagged` đối với nhánh G2: 300 mẫu có cờ phân tầng theo loại G1-below / G1-above / G2 + 300 mẫu không cờ, đã che PII) và `outputs/audits/error_buckets_45.csv` (mẫu đại diện 3 nhóm lỗi lớn / trung bình / tốt để kiểm tra nguyên nhân).

#### E. Báo cáo & Ứng dụng Streamlit (`plots.py`, `build_reports.py`, `app/streamlit_app.py`)
- `build_reports.py`: Sinh tự động các biểu đồ PNG vào `outputs/figures/` và tổng hợp số liệu thực tế chạy được vào `reports/experiment_report.md` (kèm `reports/slide_outline.md`).
- `app/streamlit_app.py`:
  - **Trang 1 (Định giá & Kiểm tra tin):** Nhập biểu mẫu hoặc chọn từ 8 tin mẫu đã che PII; chạy đúng pipeline ẩn giá + model đã lưu + `C2` CQR + `G2` parser; hiển thị giá ước lượng (VND), khoảng 90% (VND), trạng thái giá chào (thấp hơn / trong / cao hơn khoảng), cảnh báo khu vực ít dữ liệu (`N_train < 30`) và ghi chú phạm vi (`06/2025–03/2026`, giá chào bán không phải thẩm định). Hiển thị 3–5 tin tham khảo gần nhất tìm từ kho `train` đã lưu (lọc theo loại hình + tỉnh/quận, xếp hạng theo khoảng cách diện tích + tương đồng đặc trưng, đã che PII).
  - **Trang 2 (Kết quả Thực nghiệm & Độ phủ):** Đọc trực tiếp bảng metrics (`E1–E6`) và biểu đồ đã lưu trong `outputs/`.

---

## 4. Kế hoạch Kiểm chứng & Chạy Thử

1. **Unit & Integration Tests (`pytest tests/`):**
   - `test_price_area_parser.py`: mọi biến thể giá (`tỷ`, `triệu/m2`, chữ, `2.050 tỷ`, `8,x tỷ`, tin bán có `HDT 20tr/t`, tin thuê thuần, tin gộp nhiều lô).
   - `test_text_masker.py`: kiểm tra xóa sạch số tiền trong văn bản và giữ nguyên diện tích, số tầng, số phòng, đường rộng, địa chỉ ngõ; kiểm tra `find_leaked_money`.
   - `test_location_and_cleaner.py`: chuẩn hóa quận/phường dạng số, chuẩn hóa hướng, chặn cực trị bảng.
   - `test_dedup_and_splits.py`: kiểm chứng không giao `row_id`, `dup_group_id`, `district_key` trên dữ liệu tổng hợp; đúng giữ bản sớm nhất ở kịch bản A.
   - `test_conformal_and_metrics.py`: công thức finite-sample CQR, fallback nhóm `< 500`, xử lý cận đảo, chính xác công thức `MAE_log`, `MdAPE`, `WAPE`, `interval_score`.
2. **Chạy thực tế trên Local Profile (`configs/local_smoke.yaml`) trên ổ `E:`:**
   - Tải 10 shard về `E:\CS114 - Nhóm 8\data\raw\`, chạy trọn bộ pipeline TV1 → TV2 → TV3 → TV4 với ngân sách sample vừa (`30k` train / `5k` val-cal-test) để tạo đủ dữ liệu audit, metrics `E1–E6`, checkpoint, hình vẽ, báo cáo và kiểm tra ứng dụng Streamlit chạy thực tế không lỗi.
3. **Đóng gói chạy Kaggle (`notebooks/kaggle_runner.ipynb` + `configs/kaggle_full.yaml`):**
   - Kiểm tra dry-run cấu trúc tham số CLI để bạn đưa lên Kaggle chạy full Option A (`100k`–`200k` mẫu) chỉ bằng một lần Run All.
