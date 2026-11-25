# Vietnam Real Estate Valuation & Mispricing Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or superpowers:subagent-driven-development) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify an end-to-end reproducible Vietnamese real estate asking-price valuation, Conformal Prediction (90% target coverage), and mispricing/mismatch flagging system fulfilling all 4 team members' deliverables (TV1–TV4), runnable both locally on `E:` (RTX 3050 4GB `fp16` medium profile) and on Kaggle (T4/P100 full Option A profile).

**Architecture:** Modular Python package under `src/` partitioned cleanly by member responsibilities (`src/data` for TV1, `src/features` & `src/models` for TV2/TV3, `src/eval` for TV4, `src/pipelines` for CLI entrypoints), driven by YAML environment profiles (`configs/local_smoke.yaml`, `configs/local_rtx3050_full.yaml`, `configs/kaggle_full.yaml`) and unified via a strict `row_id`-keyed prediction Parquet contract. All heavy storage (`.venv`, `HF_HOME`, raw shards, processed splits, PhoBERT caches) lives on `E:` locally or `/kaggle/working` on Kaggle, with shard-by-shard `pyarrow` streaming to keep peak RAM under 6 GB.

**Tech Stack:** Python 3.12, `uv`, PyTorch (CUDA), `transformers` (`vinai/phobert-base`), `pyvi`, `datasets`, `huggingface_hub`, `pandas`, `pyarrow`, `scikit-learn`, `catboost`, `datasketch`, `shap`, `matplotlib`, `seaborn`, `streamlit`, `pyyaml`, `pytest`.

**Spec:** [2026-11-25-vietnam-real-estate-valuation-design.md](../specs/2026-11-25-vietnam-real-estate-valuation-design.md)

## Global Constraints

- Target variable is `y_true = log(price)` in VND on sale listings (`property_type_name` in `{"Nhà", "Đất", "Căn hộ chung cư", "Biệt thự/Nhà liền kề", "Shophouse"}`); `price`, `text_price`, and `price_per_m2` never enter model features.
- Raw dataset revision pinned to `ca3fedcbb089bf65f7e4cfb03316cce4bd0d779f` (`tinixai/vietnam-real-estates`); PhoBERT pinned to `vinai/phobert-base` revision `01daacda68afe13d83023d16ec647239e344a1e6` with explicit `max_length=256, truncation=True`.
- Stored masked text uses `MASK_GIA` (and `<mask>` before PhoBERT tokenization) with automated zero-leakage verification; non-money numbers (`m2`, floors, rooms, road width, alley number) must be preserved.
- Splits (`train`, `val`, `cal`, `test`) must guarantee zero overlap of `row_id` and `dup_group_id` (and `district_key` in Scenario B); all transformers (`TF-IDF`, `SVD`, `PCA`, `Scaler`, rare-category maps, train `price/m²` percentiles) fit strictly on `train`.
- Prediction Parquet schema: `row_id, split, scenario, model_id, seed, y_true, y_pred, q_lo, q_hi`.
- Local `.venv`, `HF_HOME`, `UV_CACHE_DIR`, and all data/model artifacts must reside on `E:` (never bloat `C:`).

## Review Focus

- **Sale listing with embedded rental yield (`"giá 6.3 tỷ, đang có HDT 20tr/t"`):** Parser must classify `tx_type="sale"`, extract `total_sale=6.3e9` (ignoring `20e6` as `rental_income`), and masker must mask BOTH `6.3 tỷ` and `20tr/t`.
- **Vietnamese thousand-separator ambiguity (`"2.050 tỷ"` vs `price=20.5e9` vs `price=2.05e9`):** Parser and reconciler must flag 10x/100x/1000x magnitude mismatches as `noisy_order_of_magnitude` without silently mutating raw labels.
- **Post-07/2025 numeric districts & direction variants (`district_name="8"`, `house_direction="Đông - Nam"`):** Location and direction normalizers must map `"8"` in HCM to `"Quận 8"` and `"Đông - Nam"` to `"Đông Nam"` identically across train and inference.
- **PhoBERT sequence length & embedding alignment:** Tokenizer must always truncate at `max_length=256` (avoiding `1e30` overflow on `max_position_embeddings=258`), and embedding joins must assert exact `row_id` parity rather than row order.
- **Quantile crossing & small-group C3 fallback:** CQR must log raw `crossing_rate_raw` (`q_lo > q_hi`) before sorting bounds for interval adjustment, and C3 must automatically fall back when any `(property_type, price_tier_pred)` calibration bucket has `< 500` rows.

---

### Task 1: Environment Setup on `E:`, Git Init, Config System & Package Skeleton

**Files:**
- Create: `.gitignore`, `pyproject.toml`, `configs/base.yaml`, `configs/local_smoke.yaml`, `configs/local_rtx3050_full.yaml`, `configs/kaggle_full.yaml`, `configs/location_aliases_v1.yaml`, `src/__init__.py`, `src/config.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `load_config(config_path: str | Path) -> dict[str, Any]`
  - `resolve_project_paths(cfg: dict[str, Any], root: Path | None = None) -> dict[str, Path]`
  - `set_global_seed(seed: int) -> None`

- [ ] **Step 1: Initialize git repo, `.gitignore`, `pyproject.toml`, and E:-drive `.venv`**
  Create `.gitignore` (ignoring `.venv/`, `data/raw/`, `data/processed/`, `models/`, `outputs/`, `.superpowers/`, `__pycache__/`, `*.parquet`, `*.npy`, `*.cbm`), initialize `git init`, create `E:\CS114 - Nhóm 8\.venv` via `uv venv --python 3.12` with `UV_CACHE_DIR=E:\uv_cache`, and install all dependencies (`torch` CUDA index, `transformers`, `datasets`, `huggingface_hub`, `pandas`, `pyarrow`, `scikit-learn`, `catboost`, `pyvi`, `datasketch`, `shap`, `streamlit`, `matplotlib`, `seaborn`, `pyyaml`, `pytest`, `nbformat`). Verify `torch.cuda.is_available()` returns `True` and `C:` free space remains > 3 GB.
- [ ] **Step 2: Write failing test `tests/test_config.py`**
  Test `load_config` merging `base.yaml` with `local_smoke.yaml` and `kaggle_full.yaml`, verifying exact pinned revisions (`ca3fedcbb089bf65f7e4cfb03316cce4bd0d779f`, `01daacda68afe13d83023d16ec647239e344a1e6`), `seeds == [42, 17, 2026]`, `max_length == 256`, and `resolve_project_paths` honoring `PROJECT_ROOT` / Kaggle env override.
- [ ] **Step 3: Run failing test**
  Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -v` -> Expected: FAIL (`ModuleNotFoundError: No module named 'src.config'`).
- [ ] **Step 4: Write `configs/*.yaml`, `src/__init__.py`, and `src/config.py`**
  Implement deep YAML inheritance (`inherits: base.yaml`), path resolution, `HF_HOME` environment configuration, and `set_global_seed`.
- [ ] **Step 5: Verify passing & commit**
  Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -v` -> Expected: PASS. Commit Task 1 files.

---

### Task 2: TV1 — Text Normalizer, PII Redactor, Independent Price/Area Parser & Money Masker

**Files:**
- Create: `src/data/__init__.py`, `src/data/text_normalizer.py`, `src/data/price_area_parser.py`, `src/data/text_masker.py`, `tests/test_price_area_parser.py`, `tests/test_text_masker.py`

**Interfaces:**
- Consumes: `configs/base.yaml` domain thresholds (`rules_version`)
- Produces:
  - `normalize_text(text: str | None) -> str`
  - `redact_pii(text: str | None) -> str`
  - `parse_listing_text(name: str | None, description: str | None, tabular_area: float | None = None) -> ParsedListing`
  - `reconcile_labels(price: float | None, area: float | None, parsed: ParsedListing, domain_cfg: dict[str, Any]) -> ReconcileResult`
  - `mask_money_text(text: str | None, token: str = "MASK_GIA") -> str`
  - `find_leaked_money(text: str | None) -> list[str]`

- [ ] **Step 1: Write failing tests `tests/test_price_area_parser.py` & `tests/test_text_masker.py`**
  Pin all critical Vietnamese parsing & masking cases:
  - Sale formats: `"9 tỷ 100 triệu"`, `"7.45 tỷ"`, `"8,x tỷ"` (`is_approx=True`), `"nhỉnh 110tr/m2"` (`unit_per_m2`), `"62tr/1m2"`, `"một tỷ rưỡi"`, `"2.050 tỷ"` vs `price=20.5e9` (`noisy_order_of_magnitude`).
  - Sale with rental income: `"Bán căn hộ 72m2 giá 6.3 tỷ, đang có HDT 20tr/t"` -> `tx_type="sale"`, `sale_price_vnd=6.3e9`, `reconcile.bucket="clean"`.
  - Pure rental: `"Cho thuê căn hộ 12 triệu/tháng"` -> `tx_type="rent"`, `reconcile.bucket="rental"`.
  - Bundle/negotiable: `"Giá thỏa thuận"` with `price=None` -> `missing_or_negotiable`; `"bán 2 căn liền"` -> `is_bundle=True`.
  - Area parsing: `"Diện tích: 4m x 12m"` -> `48.0`, ignoring `"đường 8m"`.
  - Masking: masks all money amounts (`6.3 tỷ`, `HDT 20tr/t`, `Giảm 500tr`, `62tr/1m2`, `một tỷ rưỡi`) to `MASK_GIA`, `find_leaked_money(masked) == []`, while keeping `"5 tầng"`, `"3PN"`, `"48m2"`, `"ngõ 192"`, `"mt 4.6m"` intact.
  - PII redaction: masks phone, Zalo, email, explicit broker contact phrases and street house numbers.
- [ ] **Step 2: Run failing tests**
  Run: `.venv\Scripts\python.exe -m pytest tests/test_price_area_parser.py tests/test_text_masker.py -v` -> Expected: FAIL.
- [ ] **Step 3: Implement `text_normalizer.py`, `price_area_parser.py`, and `text_masker.py`**
  Ensure `parse_listing_text` NEVER accepts `price` as parameter (architectural anti-leakage boundary). Implement regex + word-number parser + role classification + `reconcile_labels` + `mask_money_text` + `find_leaked_money`.
- [ ] **Step 4: Run tests to verify green + commit**
  Run full `pytest` -> Expected: ALL PASS. Commit Task 2.

---

### Task 3: TV1 — Location & Direction Normalizer, Tabular Cleaner, Dedup, Splits & Downloader

**Files:**
- Create: `src/data/location_normalizer.py`, `src/data/tabular_cleaner.py`, `src/data/dedup.py`, `src/data/splits.py`, `src/data/download.py`, `tests/test_location_and_cleaner.py`, `tests/test_dedup_and_splits.py`

**Interfaces:**
- Consumes: `configs/location_aliases_v1.yaml`, `mask_money_text`
- Produces:
  - `LocationNormalizer(alias_path: Path).normalize_df(df: pd.DataFrame) -> pd.DataFrame` (adds `province_norm`, `district_norm`, `ward_norm`, `street_norm`, `project_norm`, `district_key`, `house_direction_norm`, `balcony_direction_norm`, `admin_drift_flag`)
  - `clean_tabular_domain(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame`
  - `assign_duplicate_groups(df: pd.DataFrame, threshold: float = 0.85, num_perm: int = 64) -> pd.Series`
  - `build_scenario_a_split(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame`
  - `build_scenario_b_split(df: pd.DataFrame, cfg: dict[str, Any], seed: int = 42) -> pd.DataFrame`
  - `download_dataset_shards(raw_dir: Path, repo_id: str, revision: str, rules_version: str, max_shards: int | None = None) -> dict[str, Any]`

- [ ] **Step 1: Write failing tests `tests/test_location_and_cleaner.py` & `tests/test_dedup_and_splits.py`**
  - Verify `"Hồ Chí Minh"` + `district_name="8"` -> `"Quận 8"`, `ward_name="5"` -> `"Phường 5"`, `"Đông - Nam"` -> `"Đông Nam"`, `"Đông, Tây Nam"` -> `"Đa hướng"`.
  - Verify domain cleaning turns `area=82e6`, `frontage=9e6`, `road_width=103000`, `floor_count=255` into `NaN` + sets `*_was_outlier=1` & `*_is_missing=1`.
  - Verify dedup groups identical & near-duplicate masked texts regardless of differing `price`, Scenario A keeps only earliest per `dup_group_id` and preserves chronological months, and Scenario B produces zero overlap of `row_id`, `dup_group_id`, and `district_key` across `train/val/cal/test`.
- [ ] **Step 2: Run failing tests** -> Expected: FAIL.
- [ ] **Step 3: Implement `location_normalizer.py`, `tabular_cleaner.py`, `dedup.py`, `splits.py`, and `download.py`**
  Keep memory footprint low using vectorized pandas / `datasketch.MinHash` + `MinHashLSH` within `(province_norm, district_norm, property_type_name, area_bucket)` blocks plus exact hash deduplication.
- [ ] **Step 4: Run tests + commit** -> Expected: ALL PASS.

---

### Task 4: TV1 — Streaming `prepare_data.py` Pipeline, Audit Exports & Data Reports

**Files:**
- Create: `src/pipelines/__init__.py`, `src/pipelines/prepare_data.py`, `tests/test_prepare_data_pipeline.py`

**Interfaces:**
- Consumes: `src/data/*`, `src/config.py`
- Produces:
  - `run_prepare_data(cfg: dict[str, Any], paths: dict[str, Path], raw_shards: list[Path] | None = None) -> dict[str, Any]`
  - CLI `python -m src.pipelines.prepare_data --config configs/local_smoke.yaml`
  - Artifacts: `data/processed/clean.parquet`, `data/processed/noisy_flagged.parquet`, `data/processed/rental_unknown.parquet`, `data/processed/splits/scenario_a_manifest.parquet`, `data/processed/splits/scenario_b_manifest.parquet`, `data/processed/splits/split_manifest.json`, `outputs/audits/audit_labels_300.csv`, `outputs/audits/audit_masking_100.csv`, `reports/data_report.md`, `reports/data_dictionary.md`.

- [ ] **Step 1: Write failing integration test `tests/test_prepare_data_pipeline.py`**
  Create 2 synthetic temporary Parquet shards with sale, rental, order-of-magnitude mismatch, duplicate across months, and multiple districts; run `run_prepare_data` end-to-end and assert all processed files, split non-overlap invariants, funnel counts, audit CSVs (PII redacted, 0 leaked money in masked text), and reports exist and match expected schema.
- [ ] **Step 2: Run failing test** -> Expected: FAIL.
- [ ] **Step 3: Implement `src/pipelines/prepare_data.py`**
  Stream shard-by-shard with `pyarrow.parquet.ParquetFile.iter_batches`, process & write partitioned/appended Parquet cleanly, run dedup + split A & B + stratified budget sampling + train-only percentile diagnostics, export PII-redacted `audit_labels_300.csv` & `audit_masking_100.csv` + `data_report.md` + `data_dictionary.md`.
- [ ] **Step 4: Run full `pytest` + commit** -> Expected: ALL PASS.

---

### Task 5: TV2 — Prediction Contract, Tabular Features, Baselines M0, M1 (+ Quantiles), TF-IDF M2 & M4

**Files:**
- Create: `src/features/__init__.py`, `src/features/tabular_features.py`, `src/features/tfidf_features.py`, `src/models/__init__.py`, `src/models/contract.py`, `src/models/m0_median.py`, `src/models/m1_catboost.py`, `src/models/m2_tfidf_ridge.py`, `src/models/m4_tfidf_catboost.py`, `tests/test_tabular_and_baselines.py`, `tests/test_tfidf_models.py`

**Interfaces:**
- Produces:
  - `make_prediction_df(...) -> pd.DataFrame` & `save_predictions(...)` / `load_predictions(...)` enforcing schema `row_id, split, scenario, model_id, seed, y_true, y_pred, q_lo, q_hi`
  - `TabularFeatureExtractor.fit(train_df) / transform(df) -> (pd.DataFrame, list[str])`
  - `M0GroupMedianModel.fit(train_df) / predict(df)`
  - `M1CatBoostModel` (point tuning + multi-seed + quantile 0.05/0.95 + permutation importance / SHAP helper)
  - `TfidfSvdExtractor.fit(train_texts) / transform_sparse(texts) / transform_svd(texts)`
  - `M2TfidfRidgeModel` & `M4TfidfCatBoostModel`

- [ ] **Step 1: Write failing tests `tests/test_tabular_and_baselines.py` & `tests/test_tfidf_models.py`**
  Verify M0 fallback hierarchy when `< 30` rows, rare-category `"__RARE__"` learned ONLY from `train` (unseen val/test categories map cleanly to `"__RARE__"` without error), prediction Parquet schema enforcement, M1 tuning + quantile models producing `q_lo`, `q_hi`, M2 sparse Ridge tuning, and M4 SVD+CatBoost end-to-end on synthetic frames.
- [ ] **Step 2: Run failing tests** -> Expected: FAIL.
- [ ] **Step 3: Implement `contract.py`, `tabular_features.py`, `tfidf_features.py`, `m0_median.py`, `m1_catboost.py`, `m2_tfidf_ridge.py`, `m4_tfidf_catboost.py`**
- [ ] **Step 4: Run tests + commit** -> Expected: ALL PASS.

---

### Task 6: TV3 — Vietnamese Word Segmentation, PhoBERT Embedding Cache, M3, M5 & Optional M6

**Files:**
- Create: `src/features/phobert_embedder.py`, `src/models/m3_phobert_ridge.py`, `src/models/m5_phobert_catboost.py`, `src/models/m6_finetune.py`, `src/pipelines/extract_embeddings.py`, `tests/test_phobert_and_fusion.py`

**Interfaces:**
- Produces:
  - `format_phobert_input(row) -> str`
  - `segment_vietnamese_texts(texts: list[str]) -> tuple[list[str], float]` (replaces `MASK_GIA` with `<mask>` post-segmentation; returns elapsed seconds)
  - `PhoBERTEmbedder`: `extract_and_cache(df, cache_dir, model_name, revision, batch_size, max_length=256, fp16=True) -> dict` and `load_embeddings_for_row_ids(row_ids: Sequence[str], cache_dir: Path) -> np.ndarray` (asserts 100% `row_id` match & order)
  - `M3PhoBERTRidgeModel`
  - `M5PhoBERTCatBoostModel` (PCA fit strictly on `train` + point & optional quantile pair + supports E3 masked/unmasked/no-desc variants)
  - `M6FineTuneRegressor` (optional Kaggle-only trainer script)

- [ ] **Step 1: Write failing test `tests/test_phobert_and_fusion.py`**
  - Verify `segment_vietnamese_texts` preserves `<mask>` without splitting it across tokens, verify `PhoBERTEmbedder` respects `max_length=256` on 500-word input without crashing, verify `load_embeddings_for_row_ids` raises `KeyError` on missing `row_id` and aligns out-of-order `row_ids` identically, and verify M3 & M5 fit/predict cleanly (using a lightweight deterministic dummy/cached embedder in fast unit tests plus explicit PhoBERT tokenizer check).
- [ ] **Step 2: Run failing test** -> Expected: FAIL.
- [ ] **Step 3: Implement `phobert_embedder.py`, `m3_phobert_ridge.py`, `m5_phobert_catboost.py`, `m6_finetune.py`, `extract_embeddings.py`**
- [ ] **Step 4: Run tests + commit** -> Expected: ALL PASS.

---

### Task 7: TV4 — Leakage Verification, Metrics, Conformal C0–C3, Flagging G1/G2 & Error Analysis

**Files:**
- Create: `src/eval/__init__.py`, `src/eval/leakage_checks.py`, `src/eval/metrics.py`, `src/eval/conformal.py`, `src/eval/flagger.py`, `src/eval/error_analysis.py`, `tests/test_eval_conformal_flagger.py`

**Interfaces:**
- Produces:
  - `verify_split_leakage(manifest_df, text_df=None, scenario="A") -> dict[str, Any]` (raises/flags if overlap > 0 or leaked money > 0)
  - `compute_point_metrics(y_true, y_pred) -> dict[str, float]` & `compute_interval_metrics(y_true, q_lo, q_hi, alpha=0.10) -> dict[str, float]`
  - `evaluate_predictions_df(pred_df, meta_df, train_df, alpha=0.10) -> dict[str, Any]` (includes group breakdowns `N >= 500`, Hà Nội vs TP.HCM vs Others, `worst_group_coverage`, `max_coverage_gap`, district cluster bootstrap CI)
  - `fit_apply_conformal(cal_df, eval_df, method: str, alpha: float = 0.10, min_group_size: int = 500) -> pd.DataFrame` (supports `"C0"`, `"C1"`, `"C2"`, `"C3"`)
  - `flag_mispriced_listings(eval_df, parsed_df, clean_bounds=None) -> pd.DataFrame` (`G1`, `G2`, subtype breakdown + `audit_flags_300_plus_300.csv`)
  - `sample_error_buckets(pred_df, meta_df, n_per_bucket=15) -> pd.DataFrame` & `analyze_residual_keywords(pred_df, meta_df) -> pd.DataFrame`

- [ ] **Step 1: Write failing test `tests/test_eval_conformal_flagger.py`**
  - Verify `verify_split_leakage` catches row/dup/district overlap and residual money tokens.
  - Verify exact finite-sample CQR quantile `ceil((n+1)*(1-alpha))/n`, raw crossing rate measurement + safe interval sorting, C3 group `< 500` fallback to C2, exact `MAE_log`, `MdAPE`, `within_10_pct`, `WAPE_vnd`, `interval_score`, and G1/G2 flagging taxonomy.
- [ ] **Step 2: Run failing test** -> Expected: FAIL.
- [ ] **Step 3: Implement `leakage_checks.py`, `metrics.py`, `conformal.py`, `flagger.py`, `error_analysis.py`**
- [ ] **Step 4: Run tests + commit** -> Expected: ALL PASS.

---

### Task 8: End-to-End Pipelines (`train_eval.py`, `build_reports.py`), Plotting, Kaggle Notebook & Streamlit Demo

**Files:**
- Create: `src/eval/plots.py`, `src/pipelines/train_eval.py`, `src/pipelines/build_reports.py`, `app/streamlit_app.py`, `app/inference.py`, `notebooks/kaggle_runner.ipynb`, `README.md`, `tests/test_demo_and_pipeline.py`

**Interfaces:**
- Produces:
  - `run_train_eval(cfg, paths)` -> runs E1–E6 across Scenario A & B, saves all predictions, conformal outputs, tuning tables, importance, timing/VRAM logs, and best deployable inference bundle (`models/bundle_a/`)
  - `run_build_reports(cfg, paths)` -> generates `outputs/figures/*.png`, `outputs/audits/*`, `reports/experiment_report.md`, `reports/slide_outline.md`
  - `ValuationInferenceService.load(bundle_dir) / predict_single(listing_dict) -> dict` (used by `app/streamlit_app.py` so UI imports clean testable inference service)
  - `notebooks/kaggle_runner.ipynb` (valid nbformat notebook running `prepare_data`, `extract_embeddings`, `train_eval`, `build_reports` on `configs/kaggle_full.yaml`)

- [ ] **Step 1: Write failing test `tests/test_demo_and_pipeline.py`**
  Test `ValuationInferenceService` on a mini saved bundle (masks money in user input description, predicts VND estimate + C2 90% interval + price position relative to interval + retrieves 3–5 PII-redacted comps from train index + emits low-data warning when district count `< 30`), and validate `kaggle_runner.ipynb` JSON structure.
- [ ] **Step 2: Run failing test** -> Expected: FAIL.
- [ ] **Step 3: Implement `plots.py`, `train_eval.py`, `build_reports.py`, `app/inference.py`, `app/streamlit_app.py`, `notebooks/kaggle_runner.ipynb`, `README.md`**
- [ ] **Step 4: Run full `pytest` + commit** -> Expected: ALL PASS.

---

### Task 9: Execute Real End-to-End Local Pipeline (`configs/local_smoke.yaml`), Verify Audits, Reports & Streamlit + Final Review

- [ ] **Step 1: Download 10 Parquet shards to `E:\CS114 - Nhóm 8\data\raw\` & verify checksums**
  Run `download.py` / `prepare_data.py` with `HF_HOME=E:\hf_cache` so `C:` is untouched.
- [ ] **Step 2: Run `prepare_data.py` (`configs/local_smoke.yaml`)**
  Verify full 3.5M row streaming EDA, conversion funnel, `data_report.md`, `data_dictionary.md`, `audit_labels_300.csv`, `audit_masking_100.csv`, and zero-leakage split manifests for Scenarios A & B. Inspect `audit_labels_300.csv` and `audit_masking_100.csv` directly and refine rules if label precision `< 95%` or any money leak appears!
- [ ] **Step 3: Run `extract_embeddings.py` (`configs/local_smoke.yaml`)**
  Extract and cache `vinai/phobert-base` `fp16` (`batch_size=16`) embeddings on RTX 3050 for the union of Scenario A & B sample `row_id`s; verify peak VRAM, segmentation timing, truncation rate, and zero `row_id` mismatch.
- [ ] **Step 4: Run `train_eval.py` & `build_reports.py` (`configs/local_smoke.yaml`)**
  Run all experiments E1–E6 (M0–M5, multi-seed `[42, 17, 2026]` for top models, E3 unmasked/no-desc control, Scenario B generalization, C0–C3 + stale vs recent calibration, G1/G2 flagging + `audit_flags_300_plus_300.csv`, plots, `experiment_report.md`, `slide_outline.md`). Inspect `audit_flags_300_plus_300.csv` & metrics!
- [ ] **Step 5: Verify Streamlit demo & full `pytest` suite**
  Run `pytest -v` (all unit + integration tests pass), headless smoke check `app/streamlit_app.py` startup & prediction via `ValuationInferenceService`.
- [ ] **Step 6: Final Whole-Branch Code Review**
  Dispatch a fresh `research` or `self` code-review subagent to inspect the full implementation against the Spec and Review Focus, resolve any Critical/Important issues with TDD, and confirm final passing test suite & artifacts.
