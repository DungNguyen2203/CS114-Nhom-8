# Kiểm toán độc lập hệ thống định giá BĐS Việt Nam

Ngày 26/11/2026 · nhánh `feat/full-pipeline` · commit `101c079ab78dd00e864b3bbdf9ea2307dcfa98f6` · Windows, Python 3.12.10, pandas 3.1.1. Các đường dẫn và số dòng dưới đây tính từ gốc repo ở commit này. Với Parquet, báo cáo ghi cả dòng mã đọc/ghi và khóa bản ghi để đối chiếu.

## 1. Tóm tắt đánh giá tổng quan

**Demo và luồng huấn luyện rút gọn chạy được. Chưa đủ điều kiện nghiệm thu kết luận “không rò rỉ giá”, kết quả so sánh độc lập trên B, hoặc khả năng chạy Kaggle full từ đầu đến cuối.** Có 211 tin thật trên hợp 94.854 ID của hai manifest vẫn chứa một số thập phân ngay sau chữ “giá”, bằng đúng giá bảng theo đơn vị tỷ đồng, sau khi áp dụng bộ che của HEAD. Bộ dò trả về danh sách rỗng cho cả 211. Con số này là kết quả của một phép tìm hẹp. Chưa đo được mức thay đổi MAE do những tin này. [Bộ che](../src/data/text_masker.py#L128) · [Bộ dò](../src/data/text_masker.py#L152) · [Đường đưa văn bản vào train](../src/pipelines/train_eval.py#L142)

| Kiểm chứng trực tiếp | Kết quả |
|---|---|
| Pytest trong `.venv`, với `HF_HOME=E:\hf_cache` và tắt ghi pytest cache | **54/54 pass**, 22,26 giây. Đo lại bằng coverage.py 7.17.5 trong thư mục tạm, 54/54 pass. Trên `src,app`, **57%** theo thước đo kết hợp câu lệnh và nhánh của coverage.py, 1.824/3.078 câu lệnh đã chạy (59,26%). Điểm vào test ở `pyproject.toml:39`. |
| Cú pháp, import, annotation, notebook | Parse AST 52 file Python thuộc src/app/tests, import 40 module src/app, giải được type hints của 147 hàm/phương thức đã duyệt, 0 lỗi. Notebook qua `nbformat.validate`. Đây là các phép kiểm runtime/cấu trúc. Chưa chạy static type checker toàn repo. `pyproject.toml:29`, `tests/test_demo_and_pipeline.py:87`. |
| Chạy tích hợp độc lập bằng dữ liệu thật và model thật | Lấy 500 dòng đầu mỗi shard trong 10 shard, ghép thành 5.000 dòng tạm. TV1 giữ 4.299 tin sạch. Mỗi scenario dùng train/val/cal/test **250/70/70/70**, hợp **846** ID. PhoBERT → M0–M5 → E3 → conformal → audit → bundle → báo cáo hoàn thành, sinh **6 hình**. Đã giảm CatBoost còn 5 vòng, một cấu hình, seed 42 và SVD/PCA 4 chiều. Thử nghiệm này kiểm đường chạy, không ước lượng hiệu năng của cấu hình full. `src/pipelines/prepare_data.py:486`, `src/pipelines/extract_embeddings.py:16`, `src/pipelines/train_eval.py:827`, `src/pipelines/build_reports.py:16`. |
| PhoBERT và UI | Tokenizer tại SHA trong config trả đúng **hai token ID 64000** cho hai `MASK_GIA` sau PyVi. CPU và CUDA đều trả ma trận hữu hạn `(2,768)` fp16, chuỗi dài chạm ngưỡng 256. AppTest mở và submit form qua bundle M4 đang lưu, 0 exception, 3 metric. `src/features/phobert_embedder.py:54`, `src/features/phobert_embedder.py:174`, `app/streamlit_app.py:107`. |
| Đối chiếu nguồn thiết kế | Đã đọc spec và trích được 411 đoạn XML trong DOCX, gồm các đoạn 4, 75, 102–115, 123, 157, 269–275 về giá chào bán, hai kịch bản, chống lộ giá, calibration và demo. DOCX dùng số đoạn XML vì số trang phụ thuộc trình đọc. Các yêu cầu đối chiếu dưới đây dùng số dòng ổn định trong `docs/superpowers/specs/2026-11-25-vietnam-real-estate-valuation-design.md`, viết tắt là **spec**. |

## 2. Phát hiện theo mức độ nghiêm trọng

### Critical

| ID và file:dòng | Cơ chế và tác động | Bằng chứng hoặc tái hiện tối thiểu | Đề xuất sửa |
|---|---|---|---|
| **C1. Giá chào đi qua cả hai lớp chống leakage.** `src/data/text_masker.py:66`, `:109`, `:133`, `:157`; `src/pipelines/prepare_data.py:105`; spec:132 | Bộ che và bộ dò cùng bỏ sót số thập phân có ngữ cảnh giá nhưng khuyết đơn vị. Với số nguyên liền chữ `đ`, nhánh dành cho số không có phân cách còn đòi hậu tố đơn giá. NLP nhận số tiền trực tiếp. | ID `00_0009784` còn “Giá 6.68”, bảng 6.680.000.000 VND. ID `00_0029524` còn “Giá bán: 2,85”, bảng 2.850.000.000 VND. Cả hai có zero hit. Ca tổng hợp `Bán nhà giá 8500000000đ` giữ nguyên toàn chuỗi. Phép đếm 211 tin dùng độc lập nhãn bảng **chỉ trong audit**. | Che cụm giá có ngữ cảnh rõ và dạng số liền đơn vị, giữ đối chứng diện tích/địa chỉ. Bổ sung ca kiểm độc lập dựa vào nhãn ở bước audit ngoại tuyến. Đổi phiên bản dữ liệu, tái tạo cache rồi đo lại E3 và metric. |
| **C2. CQR có thể báo phủ một khoảng rỗng.** `src/eval/conformal.py:77`, `:81`, `:124`; `src/eval/metrics.py:50`; `src/eval/flagger.py:19`; `app/inference.py:311` | Score CQR có thể âm. Khi khoảng test hẹp hơn mức co, cận hiệu chỉnh có thể đảo. Metrics và G1 sắp cận lại lần nữa, biến tập rỗng thành khoảng rộng. Demo dùng cận theo thứ tự gốc. | 10 hàng cal có `lo=0, y=5, hi=10`. Test có `lo=4, y=5, hi=6`. Hàm cho `Q=-5`, cận `[9,1]`, raw crossing bằng 0. Evaluator ghi coverage **1**, width **8**, G1 trả `in_interval`. C2 trong artifact M4 cũ có 0/6.000 hàng đảo, nên chưa quy sai số này cho bảng M4 đang lưu. | Chọn một quy ước cho tập rỗng xuyên suốt đánh giá, gắn cờ và demo. Nếu cần luôn xuất khoảng có thứ tự, dùng biến thể CQR bảo thủ `Q=max(Q,0)`, ghi rõ trong phương pháp và hiệu chỉnh lại. Tách raw crossing khỏi cận sau calibration. |
| **C3. Holdout B ảnh hưởng gián tiếp đến cấu hình dùng cho B.** `src/pipelines/train_eval.py:198`, `:275`, `:323`, `:691`, `:748`, `:778`; spec:145 | B dùng lại cấu hình M1, số chiều SVD và PCA đã thắng trên A. Các split độc lập **bên trong B**, song nhãn của một số tin B-test/B-cal đã tham gia fit hoặc chấm ứng viên ở A. Vì vậy tuyên bố chọn hyperparameter hoàn toàn biệt lập với B-test/B-cal không đứng vững. | Lấy giao theo `row_id` giữa hai manifest. B-test có **87** ID thuộc A-train và **17** thuộc A-val. B-cal có **95** thuộc A-train và **21** thuộc A-val. Model B được fit lại trên B-train, winner cuối cùng được chọn trên B-val tại `train_eval.py:800`. Chưa định lượng hướng hoặc độ lớn sai lệch. | Tuning lại trên B-train/B-val, hoặc cố định cấu hình trước cả hai thí nghiệm. Ghi nguồn cấu hình vào log B và kiểm tra giao giữa dữ liệu chọn cấu hình và holdout B. |
| **C4. Bộ artifact đang lưu thuộc nhiều phiên bản và báo sai winner B.** `configs/base.yaml:3`, `:6`; `data/processed/embeddings/cache_meta.json:4`; `models/bundle_a/bundle_meta.json:3`; `outputs/metrics/run_summary.json:3`, `:13`; `reports/experiment_report.md:4` | Config hiện tại là v1.1.0. Toàn bộ 94.854 dòng clean của hợp manifest mang v1.0.2, bundle mang v1.0.2, cache ghi v1.0.0. Báo cáo cũ ghi M5 thắng B dù validation MAE nhỏ nhất thuộc M0. Dùng riêng lệnh build_reports vẫn đọc winner cũ. | Gọi `load_embeddings_for_row_ids([id], cache_dir, "v1.1.0")` nhận `ValueError` đúng như thiết kế. Trong JSON cũ, M0 **0,551245**, M5 **0,565848**. Với cùng dict, `min` trả M0, `_pick_conformal_model` trả M5. Hàm train mới đã tách hai khái niệm ở `train_eval.py:799`. | Tạo run mới từ prepare đến report. Buộc clean, split, cache, bundle và metrics cùng mang run ID, rules/text version và nguồn dữ liệu trước khi cho phép ghép hoặc xuất báo cáo. |
| **C5. Tin sai nhãn vẫn được đưa vào clean.** `src/data/price_area_parser.py:76`, `:92`, `:300`, `:362`, `:390`; spec:107,114 | Masker hiểu nhiều dạng tiền hơn parser. Parser ghi nhận thỏa thuận nhưng reconciler bỏ qua cờ này khi bảng có giá. Regex gộp bỏ sót một số tin nêu nhiều lô với một giá tổng. Các hàng này có thể đi vào cả train và đánh giá. | Với diện tích 60 m², “Bán nhà 5.8..tỷ” và giá bảng 50 tỷ cho `clean_no_text_price`. “Bán nhà giá thỏa thuận” cùng giá bảng cũng cho `clean_no_text_price`. “Bán đất 3 lô giá 6 tỷ”, giá bảng 6 tỷ, cho `is_bundle=False, clean_consistent`. | Chuẩn hóa dấu câu ngay trong parser, xử lý thỏa thuận không có số trước nhánh clean, nhận diện nhiều lô chưa có giá riêng. Phân biệt “liên hệ” ở chân tin với “liên hệ để biết giá”. |

### Important

| ID và file:dòng | Cơ chế và tác động | Bằng chứng hoặc tái hiện tối thiểu | Đề xuất sửa |
|---|---|---|---|
| **I1. Gate strict dừng trên văn bản thật và có thể pass khi thiếu bằng chứng.** `src/data/text_masker.py:22`, `:161`; `src/eval/leakage_checks.py:24`, `:34`, `:58`, `:66`; `src/pipelines/train_eval.py:855` | Mẫu che range chưa bao hết các đơn vị mà bộ dò phần còn dư nhận diện. Mặt khác, thiếu cột hoặc thiếu text được hiểu thành tập so sánh rỗng. | Trên **text đã lưu**, bộ dò HEAD báo A **945/48.000**, B **930/48.000**. Sau khi che lại tiêu đề và mô tả của cùng 94.854 ID bằng HEAD, còn **A 0, B 4**. ID `00_0053205` còn “8.800 - MASK_GIA” trong **phí quản lý**. `8.800 - 12.000 đồng/tháng` tái hiện trực tiếp. Ngược lại, truyền manifest B chỉ có một dòng train và `text_df=pd.DataFrame()` nhận `passed=True, n_texts_checked=0`. | Hoàn thiện range, giữ ngưỡng 0. Trong chế độ strict, kiểm schema, sự hiện diện các split, uniqueness và khớp chính xác tập ID cần kiểm. Ghi riêng số cảnh báo, số tiền còn sót và số thực sự lộ giá chào. |
| **I2. G2 trên demo bỏ cờ diện tích và form bỏ chiều sâu.** `app/inference.py:272`, `:330`, `:336`; `app/streamlit_app.py:88`, `:128`; `src/eval/flagger.py:36` | Reconciler trả riêng status giá và cờ diện tích, service chỉ chuyển status. Form cũng không truyền `house_depth` dù bộ feature và một số preset dùng trường này. | Tin 45 m², mô tả 120 m², giá bảng/mô tả cùng 8,5 tỷ cho `clean_consistent, area_mismatch_flag=True`. Service chỉ trả `clean_consistent`, UI im lặng. Với preset đầu, bỏ chiều sâu 10,5 m như form làm thay đổi giá M4 từ **12.760.061.349** thành **12.771.952.633** VND. | Trả G2 theo cùng hàm phân loại với offline, hiển thị lý do riêng cho diện tích. Cho form nhập và truyền chiều sâu, kiểm tra parity với payload của preset. |
| **I3. Che PII bỏ sót số nhà.** `src/data/text_normalizer.py:26`, `:66`; `src/pipelines/prepare_data.py:355`; `outputs/audits/audit_labels_300.csv:94`; spec:100 | Regex địa chỉ chủ yếu nhận số có dấu gạch chéo. Các exporter gọi đúng hàm nhưng kế thừa điểm mù của nó. | `redact_pii("Nhà số 123 đường Ví Dụ, LH 0901 234 567")` giữ nguyên số nhà và che điện thoại. Tìm mẫu “số + chữ số + đường/phố/ngõ/hẻm” trong các cột đã redacted thấy một ô ở CSV nhãn, ID `02_0320692`. | Bổ sung ngữ cảnh số nhà đơn và test đối chứng số tầng/đường rộng. Xuất lại audit và rà văn bản của comps trước khi chia sẻ. |
| **I4. Dữ liệu cũ có thể được gắn phiên bản mới.** `src/data/download.py:39`, `:65`; `src/pipelines/extract_embeddings.py:29`, `:62`; `src/features/phobert_embedder.py:101`, `:110` | Shard có sẵn được tính lại hash và ghi revision yêu cầu mà không so với nguồn đã xác minh. Bước embedding đọc masked text rồi gắn text_version từ config. Loader chỉ so text_version và ánh xạ ID bằng dict. | Đặt Parquet một dòng vào thư mục tạm, gọi downloader với revision giả: manifest vẫn ghi revision đó và 1 dòng. Ghi cache hai hàng cùng ID `x`, vector 1 và 2: loader trả 2. Trên bộ hiện tại, 10/10 hash khớp **manifest local**, toàn bộ offset thực tế hợp lệ. | Kiểm provenance của shard khi reuse, so phiên bản clean với config trước extract, lưu và xác nhận model SHA/cấu hình embedding, từ chối ID/offset trùng, thiếu hoặc ngoài miền. |
| **I5. Rerun với bucket trống giữ Parquet của lượt trước.** `src/pipelines/prepare_data.py:241`, `:254`, `:553`; `src/pipelines/train_eval.py:850` | Khi tất cả part có 0 dòng, merge chỉ tạo file rỗng nếu đích chưa tồn tại. Lượt sau có thể đưa noisy cũ vào audit mới. | Ghi một Parquet đích một dòng, truyền một part rỗng cho `_merge_shard_parquets([empty], dest)`. Hàm trả **0**, đọc lại đích vẫn có **1** dòng. | Luôn xuất đích của lượt hiện tại, kể cả rỗng, với schema ổn định. Dùng file tạm rồi thay đích sau khi ghi thành công. |
| **I6. Kaggle Run All có thể đóng gói sau một stage thất bại.** `notebooks/kaggle_runner.ipynb:22`, `:31`, `:41`, `:71`, `:81`; `src/pipelines/build_reports.py:26` | Các cell chạy pytest, CLI và tar bằng IPython `!`, và cuối notebook in “Done!” vô điều kiện. Notebook clone đầu nhánh và cài riêng một phần dependency. Đầu ra cũ hoặc thiếu có thể bị đóng gói. | Trong IPython thật, cell `!cmd /c exit 7` cho `success=True`, `_exit_code=7` và cell sau chạy. Report builder coi một số file metrics thiếu là DataFrame rỗng. `nbformat.validate` vẫn pass vì chỉ kiểm cấu trúc. | Dùng `subprocess.run(..., check=True)` với `sys.executable` cho mọi stage, kiểm artifact bắt buộc trước tar, ghi commit thực chạy và môi trường. Chạy Run All trên Kaggle sạch để xác nhận budget/RAM/VRAM. |
| **I7. M0 và dedup có thể tin quá mức vào nhóm nhỏ hoặc tiền tố giống nhau.** `src/models/m0_median.py:46`, `:62`; `src/data/dedup.py:30`, `:64`, `:81`, `:106`; spec:139,162 | M0 áp ngưỡng min-count ở hai cấp đầu nhưng dùng median loại hình kể cả một mẫu. Exact hash chỉ thấy 300 ký tự, MinHash chỉ lấy 50 shingle đầu và union ứng viên LSH ngay. | Với ba tin A giá 1 tỷ và một tin B giá 5 tỷ, cùng diện tích, ngưỡng 3, B được dự đoán 5 tỷ thay vì fallback toàn train 1 tỷ. Hai tin cùng thông số, cùng `"abc "*76` nhưng khác phần còn lại nhận cùng dup ID. Chưa đo tỷ lệ gộp sai trong corpus. | Áp ngưỡng cho cấp loại hình theo spec. Dùng hash văn bản đầy đủ cho exact; với ứng viên near-duplicate, xác nhận Jaccard theo tiêu chuẩn đã công bố trước khi union. |
| **I8. M6 có thể khôi phục nhầm checkpoint trên CPU.** `src/models/m6_finetune.py:115`, `:121`; `src/pipelines/train_eval.py:357`; spec:184 | `v.cpu()` của state dict vốn ở CPU dùng chung storage với tham số. Epoch sau có thể ghi đè “best”. Khi bật, M6 chạy trước E3–E6; vòng lặp bước optimizer mỗi batch ở `m6_finetune.py:97`. | Lưu `{k:v.cpu() for ...}` từ một `torch.nn.Linear` trên CPU, tăng weight tại chỗ: tensor đã lưu đổi theo. M6 mặc định tắt trong `configs/kaggle_full.yaml:20` và `configs/local_smoke.yaml:23`. | Clone tensor best trên CPU và test epoch tốt nhất nằm giữa. Nếu triển khai M6 theo spec, chạy như bước tùy chọn sau E1–E6 và thêm accumulation đã nêu. |
| **I9. Đối chứng stale-cal chưa cô lập toàn bộ bước học.** `src/pipelines/train_eval.py:129`, `:198`, `:545`, `:556`, `:585` | Hai nhánh stale/fresh thật sự dùng chung model và test. Tuy nhiên vocabulary rare-category và cấu hình M1 đã học/chọn bằng cả A-train, bao gồm thời gian dành làm stale-cal. | So các đối số: `tab_ext.fit_transform(tr)` và tuning toàn `tr` chạy trước khi tạo `early_mask`. Chỉ bước fit lại CatBoost lấy các hàng early. Artifact cũ ở `reports/experiment_report.md:51` chỉ có dòng stale. | Tách early/stale trước mọi bước fit hoặc tuning của đối chứng. Fit bộ feature và chọn cấu hình chỉ từ phần được phép, sau đó giữ nguyên model cho cả hai calibration. |

### Minor / Cải thiện

| ID và file:dòng | Cơ chế, bằng chứng và ảnh hưởng | Đề xuất sửa |
|---|---|---|
| **M1. Trường hợp finite-sample không đủ mẫu.** `src/eval/conformal.py:19`; `tests/test_eval_conformal_flagger.py:284` | Với `n=1, alpha=0.1`, hạng yêu cầu là 2, hàm clip và trả score duy nhất. Trường hợp này không có ngưỡng hữu hạn mang bảo đảm 90% phân phối tự do. Với cal hiện có hàng nghìn mẫu, không chạm biên này. | Xác thực alpha, score hữu hạn và trả ngưỡng vô hạn hoặc lỗi rõ ràng khi hạng vượt n. |
| **M2. Phạm vi thời gian và hyperparameter trong log chưa đầy đủ.** `src/models/catboost_utils.py:97`; `src/models/m5_phobert_catboost.py:61`, `:79`; `src/pipelines/train_eval.py:276`, `:324`; `outputs/metrics/tuning_logs.csv:16` | 18/18 dòng tuning có thời gian (M1/M2/M3/M4/M5 = 8/3/3/2/2). Dòng M4/M5 có k, MAE và thời gian nhưng thiếu loss/depth/lr kế thừa; thời gian M5 chỉ tính phần CatBoost, không gồm scaler/PCA hay quantile heads. `best_iteration=0` cũng bị `or iterations` ghi thành toàn bộ ngân sách. M2 cố định bigram ở `train_eval.py:138`, trong khi spec:167 nêu cả unigram. | Ghi effective config, seed và phạm vi timer. Dùng kiểm tra `is None` cho best iteration. Thử hai n-gram hoặc cập nhật phạm vi thực nghiệm. |
| **M3. Tài liệu và hình vẽ dễ bị đọc sai.** `src/pipelines/prepare_data.py:471`; `reports/data_dictionary.md:6`; `src/eval/plots.py:98`; `reports/experiment_report.md:50`, `:62` | Dictionary ghi `price: float64`, trong khi schema Arrow clean thực tế là `large_string`. Hình coverage đặt một đường 90% cho cả hàng C2 alpha 0,20 có mục tiêu 80%. Bản report cũ ghi 0,034 ms/tin chung, JSON chỉ có batch predict. | Đồng nhất schema hoặc sửa mô tả trong generator. Vẽ mục tiêu theo alpha, ghi rõ batch hay end-to-end, tái sinh Markdown. |
| **M4. Bộ lấy mẫu vẫn dùng chung vị trí trong các tầng cùng cỡ.** `src/data/splits.py:148`, `:159`, `:174` | Khi giảm mẫu có shuffle cuối. Nhưng hai tầng 5 phần tử cùng thứ tự nội bộ, lấy 6 dòng tổng ở seed 42, cả hai đều chọn các vị trí `{1,2,4}`. Khi ngân sách đủ lớn hàm trả nguyên thứ tự. Điều này quan trọng nếu mỗi tầng đều được sắp theo thời gian. | Dùng cùng một đối tượng RNG tiến trạng thái giữa các lần lấy mẫu. Ghi rõ khi nào cần shuffle cả trường hợp lấy toàn bộ. |
| **M5. Giá trị thiếu của pandas và cấu hình pooling mở rộng.** `src/data/text_normalizer.py:37`, `:44`; `src/pipelines/extract_embeddings.py:56`; `src/pipelines/train_eval.py:468`; `app/inference.py:210` | `normalize_text(pd.NA)` ném `TypeError: boolean value of NA is ambiguous`. Khi bật `chunk_pooling`, extract nhận cờ nhưng E3 title/unmasked và service dựng embedder theo mặc định false. Các profile đang xét dùng false tại `configs/base.yaml:78`. | Xử lý sentinel thiếu scalar trước phép kiểm boolean. Truyền cùng cấu hình encoder đã lưu ở cả ba đường và kiểm parity khi bật pooling. |

## 3. Bảng kiểm tra rò rỉ và toán học

Pass và Fail áp dụng cho từng thuộc tính trong bảng. Unverified chỉ phần chưa thực thi hoặc chưa đủ bằng chứng để suy rộng.

| Phạm vi | Pass | Fail | Unverified |
|---|---|---|---|
| `src/data` | Parser chỉ nhận text và diện tích, đối chiếu giá ở hàm riêng (`price_area_parser.py:287`, `:352`). Exact/MinHash đọc allowlist text masked, địa lý, loại hình và thuộc tính vật lý, không đưa trực tiếp `price/y_true` vào chữ ký dù DataFrame chứa chúng (`dedup.py:63`, `:81`). A sort theo ngày rồi row_id, giữ earliest, tháng thực tế train 06–10/2025, val 11–12/2025, cal 01–02/2026, test 03/2026 (`splits.py:19`, `:32`). B giữ một đại diện rồi gán nguyên quận vào một split. Đo trên cả hai manifest thấy giao row ID/dup bằng 0, B giao district cũng bằng 0 (`splits.py:61`, `:124`). | C1, C5, I1, I3–I5, I7, M4–M5. Cặp near-duplicate lọt ngoài block chưa được kiểm bằng điều kiện zero-overlap của mã cụm. | Precision/recall của dedup và parser trên tập đã gán nhãn thủ công. |
| `src/features` | `__RARE__` fit từ train (`tabular_features.py:69`, `:90`), TF-IDF/SVD chỉ fit train (`tfidf_features.py:39`, `:64`), các split khác transform tại `train_eval.py:129`, `:142`. PhoBERT thật theo thứ tự PyVi → thay mask → tokenizer; auto device, no-grad, AMP chỉ trên CUDA (`phobert_embedder.py:54`, `:160`, `:174`, `:188`). Cache thật fp16 **94.854 × 768**, ID/offset duy nhất, offset liên tiếp, tập ID bằng đúng hợp hai manifest. | I4, M5. Version/SHA nằm trong JSON riêng, không nằm trên index hai cột như spec:179 (`phobert_embedder.py:78`, `:81`). | Đồng nhất embedding ở mức bit giữa các GPU/môi trường. |
| `src/models` | M2/M3 chọn alpha theo MAE val, M3 L2-normalize (`m2_tfidf_ridge.py:36`, `m3_phobert_ridge.py:28`). M5 fit scaler/PCA trên emb train và transform val (`m5_phobert_catboost.py:37`, `:61`). M1/M4/M5 có hai head 0,05/0,95 dùng loss quantile cho early stopping; CatBoost train CPU với random seed và eval_set val (`configs/base.yaml:88`, `catboost_utils.py:40`, `:46`, `:132`, `:146`; `train_eval.py:201`, `:288`, `:331`). Ba seed chạy điểm, head quantile chỉ ở seed chính. | M0 ở cấp loại hình, M6 ở chế độ CPU, phạm vi tuning log (I7, I8, M2). | Chạy đủ M6 nhiều epoch và tái lập bitwise cả lượt huấn luyện. Seed Python/NumPy/Torch được đặt tại `src/config.py:70`; đây chưa phải phép so sánh hai run độc lập. |
| `src/eval` | C1 dùng trị tuyệt đối và order statistic, C2 dùng max hai phía trên cal, C3 chia tercile theo **prediction cal**, fallback loại hình rồi global khi thiếu mẫu (`conformal.py:19`, `:54`, `:77`, `:91`, `:115`). Raw crossing ghi **trước** sắp cận ở C0/C2/C3 (`:45`, `:72`), metrics dùng cờ này (`metrics.py:99`). Metadata merge chỉ lấy cột còn thiếu, group đủ `n >= min_eval_group_size` mới vào worst coverage (`metrics.py:85`, `:126`, `:142`). Bootstrap ghép cặp theo row ID, lấy lại có hoàn lại **nguyên quận**, chia theo tổng số dòng đã lấy. Ví dụ hai quận với lỗi [1,1] và [10] cho điểm 4, CI [1,10] (`metrics.py:159`, `:169`, `:182`). | C2, I1, M1, M3. | Độ chính xác thực địa của cờ lệch giá và ý nghĩa nhân quả của từ khóa. |
| `src/pipelines` | A chọn winner toàn bộ theo MAE val và winner conformal trong M1/M4/M5. Cả A/B thực sự truyền `strict_text=True` với ngưỡng mặc định 0 (`train_eval.py:390`, `:800`, `:854`; `configs/base.yaml:102`). Trong E3, cả sáu biến thể người dùng nêu dùng chung các index train/val/test phân tầng tỉnh × loại hình lấy một lần tại `train_eval.py:408`–416. M1 dùng các vị trí tại dòng 420, M4 masked/unmasked ở 429/459, M5 masked/title/unmasked ở 464/484/494. M5 masked dùng cache tạo với cùng `format_phobert_input` (`phobert_embedder.py:269`); title-only bỏ mô tả ở 477, unmasked đổi nguồn text ở 487. M4 dùng TF-IDF, M1 dùng bảng cùng cờ từ khóa/độ dài masked text (`tabular_features.py:113`). | C3–C4, I4–I5, I9. Cùng index không đảm bảo loại bỏ mọi khác biệt tiền xử lý. Ví dụ rare vocab ở E3 dùng toàn A-train trước khi cắt slice (`train_eval.py:129`, `:420`). | Chi phí và kết quả khi chạy nguyên profile `kaggle_full` hoặc `local_smoke` trên tất cả dữ liệu ở HEAD. |
| `app/` | Đối chiếu **16 tin test** (8 ID đầu mỗi shard 08/09) từ raw → `_process_batch` → extractor/SVD/bundle, so với `predict_single` trên cùng hàng. Chênh tuyệt đối lớn nhất của cả point, lo và hi trên thang log đều **0**. Thay riêng giá hỏi không đổi point. 30.000 ID của `train_comps.parquet` trùng đúng 30.000 ID A-train. Fallback alias hoạt động khi ép đường dẫn Linux không tồn tại rồi mở bundle trên Windows. `prepare_data.py:68`, `:105`; `train_eval.py:623`; `app/inference.py:181`, `:220`, `:254`, `:285`, `:297`, `:309`. | I2, I3, M5. Retrieval ưu tiên quận nếu có ít nhất 3 mẫu, rồi tỉnh/loại hình, xếp theo khoảng cách log diện tích (`inference.py:225`, `:234`); chưa có thành phần khoảng cách đặc trưng khác như spec:220. File aliases fallback lấy từ checkout nên không đảm bảo cùng nội dung nếu deploy khác version. | Windows → Linux bằng bundle thật; parity định lượng của mọi bundle M1/M5. Lượt tích hợp riêng đã tạo và gọi được bundle M5. |
| `notebooks/`, cấu hình, dependency | YAML kế thừa qua `src/config.py:26`. Notebook đặt HF_HOME trước các subprocess (`kaggle_runner.ipynb:29`); downloader import Hugging Face lười tại `download.py:40`, transformer import lười tại `phobert_embedder.py:158`. CatBoost, PyVi, datasketch và tabulate đã khai báo (`pyproject.toml:17`, `:20`, `:21`, `:26`), nbformat ở dev (`:32`). SHAP dùng API CatBoost `get_feature_importance(type="ShapValues")` (`catboost_utils.py:183`), nên đường này không cần import gói `shap`. Không thấy `groupby().apply()` trong mã src/app đang kiểm. | I6. Danh sách cài ở notebook dựa một phần vào image Kaggle, không cài theo khai báo đầy đủ của project. Bản ghi môi trường đã test là Windows tại `requirements-tested.txt:1`. | Thực thi trên Kaggle Linux T4/P100, bộ package sẵn có trong image và tài nguyên đỉnh của lượt full. |

### Kết quả chi tiết về masking, parser và audit E6

| Đầu vào đã thử | Mask | Parser/đối chiếu |
|---|---|---|
| `4. tỷ`, `4,Tỷ 780`, `500tr`, `3 củ`, `5.800.000.000 đồng`, `8,x tỷ` | Một token mỗi cụm | Lần lượt 4 tỷ, 4,78 tỷ, 500 triệu, 3 triệu, 5,8 tỷ, xấp xỉ 8,5 tỷ. `src/data/text_masker.py:28`, `:37`, `:47`, `:52`; `src/data/price_area_parser.py:165`, `:189`, `:198`. |
| `110tr/m2`, `500k/m2`, `5.8..tỷ`, `6xxtr` | Đều được che | Trường hợp đầu cho 110 triệu/m² và 5,5 tỷ với 50 m². Ba trường hợp sau không cho giá; riêng `find_leaked_money("500k/m2")` khi đưa trực tiếp chuỗi chưa che cũng trả rỗng. `text_masker.py:70`, `:109`; `price_area_parser.py:83`, `:95`, `:323`. |
| `Quận 6`, `1181 Đ Lê Văn Lương`, `6, Đồng Mai`, `1 trệt`, `2 Củ Chi` | Giữ nguyên cả năm | Không suy ra giá. Đây là kết quả mong muốn cho đầu vào model; xuất audit cần áp dụng thêm chính sách PII. `text_masker.py:47`, `:60`; `price_area_parser.py:96`. |
| `Bán nhà 5 tỷ HĐT 20tr/th`, `bán cả cặp 2 căn giá 5 tỷ`, `Không rõ` | Giá bán và tiền thuê đều được che | Tin đầu vẫn là sale, giá bán 5 tỷ. Với giá bảng, tin thứ hai và thứ ba vào multi_or_unknown. `price_area_parser.py:145`, `:299`, `:312`, `:366`. |

**211/211 tin lộ cụm giá vẫn được parser/reconciler của HEAD xếp clean** khi chạy lại trên text gốc với giá và diện tích bảng. Theo manifest cũ, A có 63 train, 20 val, 24 cal, **15 test**. B có 52 train, 13 val, 16 cal, **11 test**. Mỗi scenario dùng 48.000 hàng; phép đếm hợp loại trùng giữa hai scenario. Bốn cảnh báo sau khi remask ở I1 là phép đo khác: bộ dò đã **bỏ qua** 211 tin trên. Phân loại tại `src/data/price_area_parser.py:352`, đường ghi text tại `src/pipelines/prepare_data.py:114`, gate tại `src/eval/leakage_checks.py:59`.

**E6 đạt cân bằng khi có đủ ứng viên.** Thuật toán dành quota cho below/above/G2 trước khi bù từ pool còn lại (`src/eval/flagger.py:83`, `:96`, `:106`). CSV đang lưu có 300 có cờ và 300 không cờ, gồm 100 below, 100 above, 100 G2 (`outputs/metrics/run_summary.json:25`). Trên lượt rút gọn chỉ có 3 below và 1 above được xuất, nên G2 chiếm đa số sau khi bù. Không nên suy tỷ lệ bất thường toàn thị trường từ bảng lấy mẫu cân bằng này. Các trường human review trong 300 nhãn, 100 masking và 600 flag hiện đều trống (`src/pipelines/prepare_data.py:360`, `:378`; `src/eval/flagger.py:139`).

**Các exporter có gọi redaction.** Label audit và mask audit ở `prepare_data.py:355`, `:374`; flag audit dùng allowlist cột ở `flagger.py:127`, `:130`; error buckets redacts cả masked text và bỏ cột thô ở `error_analysis.py:55`, `:61`; comps ở `train_eval.py:629`. Cả bốn CSV được kiểm trường văn bản. Điểm sót đã quan sát ở I3 đủ để bác bỏ kết luận “đã che hết PII”. File Parquet prediction C2 còn mang `text_masked` từ `train_eval.py:521`, `:536`; **mask giá không đồng nghĩa với redact PII** khi chia sẻ cả thư mục outputs.

**Đối chiếu bốn báo cáo.** Data report được sinh từ funnel/manifest (`prepare_data.py:397`, `:463`) và ghi v1.0.2 ở `reports/data_report.md:4`; dictionary sinh từ bảng mô tả cố định (`prepare_data.py:465`, `:483`), có sai khác schema ở M3. Experiment report đọc các CSV/JSON có sẵn (`build_reports.py:26`, `:39`); bản lưu thiếu dòng fresh đối chứng, thêm hai hàng sparse E3 trong generator pipeline hiện hành tại `train_eval.py:451`–452 chưa có trong bảng `reports/experiment_report.md:29`–34. Outline 13 slide ở `reports/slide_outline.md:1` tương ứng danh sách trong `build_reports.py:108`, `:124`; mục “kết quả kiểm định mẫu audit E6” ở slide 10 chưa có đánh giá người đọc đi kèm trong CSV.

## 4. Khoảng trống test

| Ưu tiên | Test đang bảo vệ | Ca cần bổ sung |
|---|---|---|
| Leakage và nhãn | Test regex có ca giữ địa chỉ, lặp dấu câu và x/xx (`tests/test_text_masker.py:53`, `:91`); có phân biệt bán kèm thuê (`tests/test_price_area_parser.py:67`). | Giá ngầm định theo ngữ cảnh, VND nguyên liền `đ`, range phí, 3 lô, và thỏa thuận khi bảng có giá. Test hiện tại truyền **None** ở `tests/test_price_area_parser.py:96`, nên luôn qua nhánh thiếu giá. Kiểm gate với manifest thiếu split/cột/ID. |
| CQR, subgroup, bootstrap | Có test công thức metric thật, order statistic, raw crossing, merge subgroup (`tests/test_eval_conformal_flagger.py:35`, `:168`, `:274`, `:287`). | Khi Q âm và test hẹp, evaluator/flagger/demo phải xử lý nhất quán. Test C3 hiện chỉ đòi **có ít nhất một** fallback (`:101`); cần so đúng Q cho từng nhánh tier → type → global. Thêm ca bootstrap có quận lớn/nhỏ với kết quả tính tay, và `n+1` ngoài hạng hữu hạn. |
| Mô hình và kịch bản | Test có train CatBoost/Ridge thật (`tests/test_tabular_and_baselines.py:109`, `tests/test_tfidf_models.py:11`). Tách cụm/tháng có assertion trên từng ID (`tests/test_dedup_and_splits.py:67`). | Test M0 chỉ kiểm độ dài/hữu hạn (`tests/test_tabular_and_baselines.py:105`). Thêm giá trị kỳ vọng ở cả bốn cấp, hai văn bản giống tiền tố nhưng khác đuôi, nguồn tuning B và dữ liệu được phép dùng cho stale-cal. |
| PhoBERT, cache và inference | Cache có test đảo thứ tự ID thật và ID thiếu (`tests/test_phobert_and_fusion.py:49`, `:55`). Test demo dùng bundle M1 với Q dương (`tests/test_demo_and_pipeline.py:36`). | Test segmentation kiểm chuỗi `<mask>` (`tests/test_phobert_and_fusion.py:27`) nhưng chưa kiểm ID tokenizer. Bổ sung cache sai SHA/offset, đường M4/M5 và G2 diện tích, form đầy đủ, bundle chuyển hệ điều hành, M6 giữ best epoch CPU. |
| Tích hợp, báo cáo, notebook | TV1 có test end-to-end với Parquet tạm (`tests/test_prepare_data_pipeline.py:279`). Test notebook kiểm cấu trúc và chuỗi tên stage (`tests/test_demo_and_pipeline.py:87`, `:93`). | Theo coverage **chỉ của pytest**, `train_eval.py` **7%**, `m6_finetune.py` **8%**, `phobert_embedder.py` **44%**; `extract_embeddings.py`, `build_reports.py`, `plots.py`, `streamlit_app.py` **0%**. Thêm một run nhỏ có lỗi stage phải dừng, rerun bucket trống, artifacts khác run, và kiểm report phản ánh đúng winner/alpha. Phép chạy tích hợp và AppTest ngoài pytest ở mục 1 không được cộng vào tỷ lệ này. |

## 5. Tái kiểm độc lập

Chạy từ gốc repo trên PowerShell. Các ví dụ Python bên dưới có thể dán vào stdin của cùng interpreter. Lệnh pytest giữ nguyên cache repo.

```powershell
$env:HF_HOME = 'E:\hf_cache'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
.\.venv\Scripts\python.exe -m pytest tests/ -v -p no:cacheprovider
```

**Tìm lại 211 bản ghi và chạy cùng bộ dò của HEAD trên hai phiên bản văn bản.** Một lần dùng text masked đang lưu, một lần che lại tên/mô tả từ đầu. Phép thử độc lập tìm từ “giá” và một số thập phân, quy đổi sang VND rồi so với nhãn chỉ ở bước kiểm toán.

```python
from collections import Counter
from pathlib import Path
import re
import pandas as pd
import pyarrow.parquet as pq
from src.data.text_masker import find_leaked_money, mask_money_text

root = Path("data/processed")
man = {
    s: pd.read_parquet(root / "splits" / f"scenario_{s}_manifest.parquet")
         .set_index("row_id")["split"].to_dict()
    for s in ("a", "b")
}
ids = set(man["a"]) | set(man["b"])
price_cue = re.compile(r"\bgiá\s*(?:bán|chào)?\s*[:=]?\s*(\d{1,3}[.,]\d{1,3})\b", re.I)
counts = Counter()
cols = ["row_id", "price", "name_norm", "description_norm", "text_masked"]
for batch in pq.ParquetFile(root / "clean.parquet").iter_batches(columns=cols, batch_size=50000):
    data = batch.to_pandas()
    for row in data[data.row_id.isin(ids)].itertuples(index=False):
        masked = mask_money_text(row.name_norm) + ". " + mask_money_text(row.description_norm)
        hit_stored = bool(find_leaked_money(row.text_masked))
        hit_fresh = bool(find_leaked_money(masked))
        exposed = not hit_fresh and any(
            abs(float(m[1].replace(",", ".")) * 1e9 - float(row.price)) < 1
            for m in price_cue.finditer(masked)
        )
        counts["exposed_union"] += exposed
        for s, splits in man.items():
            if row.row_id in splits:
                counts[s, "stored_hits"] += hit_stored
                counts[s, "fresh_hits"] += hit_fresh
                counts[s, splits[row.row_id], "exposed"] += exposed
print(counts)
```

**Tái hiện CQR và đường đi tới G1.**

```python
import pandas as pd
from src.eval.conformal import fit_apply_conformal
from src.eval.metrics import evaluate_predictions_df
from src.eval.flagger import flag_mispriced_listings

cal = pd.DataFrame(dict(row_id=range(10), y_true=5., y_pred=5., q_lo=0., q_hi=10.))
test = cal.iloc[:1].assign(q_lo=4., q_hi=6.)
out = fit_apply_conformal(cal, test, "C2", alpha=.1)
print(out[["q_lo", "q_hi", "conformal_q", "crossed_raw"]])
print(evaluate_predictions_df(out, out, train_df=cal)["overall"]["coverage"])
print(flag_mispriced_listings(out)["flag_g1"].tolist())
```

**Tái hiện ảnh hưởng chéo giữa hai scenario.**

```python
import pandas as pd
from pathlib import Path
base = Path("data/processed/splits")
a, b = (pd.read_parquet(base / f"scenario_{s}_manifest.parquet") for s in "ab")
overlap = a[["row_id", "split"]].merge(
    b[["row_id", "split"]], on="row_id", suffixes=("_a", "_b")
)
print(pd.crosstab(overlap.split_a, overlap.split_b))
```

**Thứ tự nghiệm thu đề nghị.** Sửa C1/C5 và xác minh lại cổng strict trên văn bản thật trước khi tốn GPU. Tách tuning B và thống nhất quy ước CQR, sau đó tạo run mới có provenance xuyên suốt, chạy notebook ở chế độ lỗi là dừng và hoàn tất audit thủ công. Khi đó mới đối chiếu lại MAE, coverage và E3 với kết quả lịch sử.

Giới hạn của cuộc kiểm toán: đã chạy Windows CPU/CUDA, replay bundle M4, tạo bundle M5 rút gọn và kiểm tra các artifact ghi trên đĩa. Chưa chạy profile nguyên ngân sách trên Kaggle T4/P100, chưa fine-tune M6 toàn bộ, chưa có bộ nhãn người đánh giá để định lượng precision/recall của cờ và chưa đo MAE sau sửa.
