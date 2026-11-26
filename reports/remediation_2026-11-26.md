# Kiểm chứng sau sửa lỗi kiểm toán ngày 26/11/2026

Bản sửa sử dụng luật dữ liệu và văn bản `v1.2.2`. [Báo cáo kiểm toán ban đầu](cold_audit_2026-11-26.md) ghi các ca tái hiện tại commit `101c079ab78dd00e864b3bbdf9ea2307dcfa98f6`.

## Kết quả đo trên dữ liệu thật

| Phép kiểm | Kết quả |
|---|---|
| Test tự động trên Windows | 75/75 test đạt trong 74,69 giây dưới coverage.py 7.17.5. Với `src,app`, đã chạy 2.284/3.362 câu lệnh (67,94%) và 541/940 nhánh (57,55%). Độ phủ gộp là 65,67%. AppTest và hai lượt tích hợp dữ liệu thật được chạy riêng. |
| Che lại văn bản của 94.854 ID khác nhau trong hai manifest lịch sử | Trong văn bản đã lưu trên đĩa, 255 ID chứa số thập phân sau từ “giá” trùng giá bảng theo đơn vị tỷ. Khi báo cáo ban đầu che lại từ nguồn bằng mã ở commit kiểm toán, 211 ID vẫn lộ và bộ dò khi đó bỏ sót cả 211. Che lại từ nguồn bằng v1.2.2 còn 0 ID trùng theo phép dò này. Cổng strict cũng đạt trên cả A và B, mỗi manifest 48.000 dòng. Mục 5 của báo cáo ban đầu ghi tiêu chí và mã tái hiện. |
| Quét toàn bộ tập sạch trung gian v1.2.0 | Bộ dò và phép đối chiếu giá bảng tìm được 50 ID cần rà soát trên 2.967.486 tin sạch từ 10 shard. Trong đó, 7 tin có số trùng giá bảng. Có cả cảnh báo nhầm như số quận đứng trước giá. Che lại từ văn bản gốc của 50 ID bằng v1.2.1 cho 0 cảnh báo còn lại. |
| Quét toàn bộ tập sạch trung gian v1.2.1 trên Windows | Xử lý lại 3.500.744 tin gốc bằng v1.2.1 cho 2.967.493 tin sạch. Đọc lại đủ 2.967.493 dòng từ Parquet mới. Có **0** dòng bị bộ dò tiền đánh dấu và **0** dòng còn số sau cụm “giá” trùng giá bảng theo phép kiểm ngoại tuyến. Quy tắc giữ số địa chỉ ở phiên bản này cũng bỏ qua một số cận dưới của khoảng tiền. Dòng tiếp theo ghi phép dò riêng cho trường hợp đó. |
| Phép dò riêng trên v1.2.1 và che lại bằng v1.2.2 | Tìm được 182 ID có số đứng sau “căn”, “lô”, “tầng” hoặc “diện tích” và ngay trước dấu nối với một giá đã che. Trong số đó có cận dưới tiền thuê như `thuê nguyên căn 15 - 20 triệu`. Che lại từ tiêu đề và mô tả gốc của 182 ID bằng v1.2.2 còn hai chuỗi khớp phép tìm. Kiểm trực tiếp cả hai (`07_0189338`, `08_0083668`) xác nhận đó là số tầng viết liền `tầng3`. |
| Chạy từ dữ liệu thô đến báo cáo trên Windows có CUDA | Lấy 500 dòng đầu từ mỗi shard trong 10 shard. Trong 5.000 dòng, 4.235 dòng vào tập sạch, 156 dòng nhiễu, 67 dòng cho thuê, 400 dòng thiếu giá hoặc thỏa thuận và 142 dòng gộp hoặc chưa rõ. Mỗi kịch bản dùng 250/70/70/70 dòng train/validation/calibration/test. PhoBERT tạo 863 embedding cho hợp hai manifest. |
| Gate trước khi huấn luyện | Cả A và B kiểm đủ 460 dòng mỗi kịch bản. Số dòng rò giá và số ID hoặc cụm trùng giữa các phần chia đều bằng 0. B có 0 quận/huyện chung giữa các phần chia. |
| Chọn mô hình | M0 đạt MAE_log validation thấp nhất ở cả A (0,56257) và B (0,75205). Trong các mô hình có hai phân vị, M4 đứng đầu ở A (0,72932) và M1 đứng đầu ở B (0,82423). |
| Calibration và G1 | Với mục tiêu 90%, C2 chứa giá thực của 64/70 tin test ở A và 69/70 ở B. Ca hồi quy khoảng thô [4, 6] và nhãn 3 cho cận sau hiệu chỉnh [4, 6], coverage 0 và cờ `below_interval`. |
| Đường chạy E3, báo cáo và demo | Có đầu ra cho các đối chứng dùng mô tả đã che, chưa che và chỉ tiêu đề, đối chứng calibration cũ/mới và sáu hình. Streamlit AppTest mở bundle đã tạo, gửi một tin mẫu, nhận ba giá trị theo thứ tự cận dưới ≤ giá điểm ≤ cận trên cho tin này, và đọc được bốn bảng. |
| Đối chiếu đường dự đoán của demo với pipeline | Nạp từng tin thô trong cả 70 ID test của A qua `ValuationInferenceService.predict_single`. Sai lệch tuyệt đối lớn nhất trên thang log bằng 0 cho cả giá điểm và hai cận C2 khi so với `A_M4_test_C2.parquet`. |

Run dùng để đối chiếu có ID `48ff4a8e8aeb42069d663cb596fbfe17` và prepare ID `80481a4b04ec45b9bc6878e5a086f41a`. Hai lượt trước/sau dùng cùng 5.000 dòng thô, seed 42, tối đa 5 vòng CatBoost và 4 chiều SVD/PCA. Bản trước giữ 4.299 tin sạch. Sau khi phân loại và lấy mẫu lại, hai tập test chỉ còn 6 ID chung trên A và 0 trên B. Các số MAE và coverage ở bảng trên mô tả lượt chạy rút gọn. Chênh lệch giữa hai lượt không đủ để suy ra tác động riêng của luật che giá.

Một số ca có thể tìm lại bằng `row_id` trong Parquet: `00_0162162` viết dính hai đơn giá thành `115 triệu / m25 triệu / m2`, `00_0325246` ghi khoảng `500 đến 5.000.000.000`, và `01_0117286` có giá ngầm định ở ngay trước từ “căn”. Đối chứng `01_0161995` cần giữ số quận trong “Quận 12 - chỉ 7 đồng” trong khi vẫn che giá. Các chuỗi này được kiểm trực tiếp trong `tests/test_text_masker.py`.

Có thể mở lại demo của đúng lượt 5.000 dòng trên máy đã kiểm chứng bằng hai lệnh PowerShell sau, chạy từ gốc repo.

```powershell
$env:VN_RE_PROJECT_ROOT = 'E:\CS114-validation\v122_5000'
.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

## Đối chiếu phát hiện và cách sửa

| Phát hiện trong báo cáo gốc | Thay đổi và bằng chứng |
|---|---|
| C1, C5, I1 | Mở rộng parser và bộ che cho giá ngầm định, số tiền liền đơn vị và khoảng giá. Tách tin gộp, giá thỏa thuận và các ca giá lỗi khỏi tập sạch. Gate kiểm đủ bốn split, quan hệ một-một giữa ID và văn bản, cùng phép dò đối chiếu nhãn ngoại tuyến. Xem `tests/test_text_masker.py`, `tests/test_price_area_parser.py`, `tests/test_dedup_and_splits.py`. |
| C2, M1 | C2/C3 dùng score bảo thủ `max(0, lo - y, y - hi)`. Cận thô giao nhau được ghi nhận trước khi sắp xếp. Hạng hữu hạn vượt số mẫu gây lỗi rõ ràng, nhóm C3 nhỏ dùng fallback. Xem `tests/test_conformal_regressions.py`. |
| C3, I9 | B tự chọn cấu hình CatBoost và số chiều SVD/PCA trên B-train/B-validation. Đối chứng calibration cũ tách riêng cả dữ liệu chọn cấu hình lẫn dữ liệu fit theo thời gian. Xem `src/pipelines/train_eval.py` và `tests/test_winner_selection.py`. |
| C4, I4–I6 | Đối chiếu SHA-256 của shard với nguồn đã khóa. Manifest, cache, summary và bundle mang thông tin phiên bản và hàm băm. Bộ sinh báo cáo kiểm tra các đầu vào và tính lại thứ hạng validation. Bucket trống thay được tệp cũ. Notebook dừng ngay khi cài đặt, test hoặc một stage lỗi. Xem `tests/test_provenance.py` và `tests/test_demo_and_pipeline.py`. |
| I2, I3 | Demo nhận chiều sâu và hiển thị G2 khi mô tả lệch diện tích. Các tệp audit che thêm số nhà. Parquet dự đoán C2 bỏ cột văn bản. Bundle mang tệp alias dùng lúc train. |
| I7, I8, M2, M4, M5 | Dedup xác minh Jaccard trên toàn bộ shingle khi LSH tìm được ứng viên. M0 áp ngưỡng cỡ mẫu cho các cấp địa lý. M6 sao chép checkpoint độc lập trên CPU. Tuning ghi cấu hình, thời gian; lấy mẫu phân tầng dùng một bộ sinh số ngẫu nhiên xuyên suốt. Giá trị thiếu của pandas và các tùy chọn pooling có kiểm tra. |
| M3 | Data dictionary mô tả schema xuất thực tế. Đồ thị độ phủ vẽ mục tiêu tương ứng với từng alpha. Báo cáo phân biệt mô hình điểm tốt nhất với mô hình có phân vị dùng để sinh khoảng. |

Các kết quả tự động ở trên chưa đo precision/recall của parser, dedup hoặc cờ trên bộ nhãn do người đánh giá lập. M6 là tùy chọn tắt trong cả hai profile kiểm chứng. Lượt chạy 5.000 dòng dùng profile `validation_5000`; kết quả của nó được giữ riêng với `local_smoke` và `kaggle_full`.

## Kiểm chứng trên Kaggle Linux

Kaggle đã hoàn tất phiên 4 của notebook riêng tư với profile `kaggle_full`, luật v1.2.1 và **75/75 test đạt trong 37,56 giây**. Run ID là `2003672eb5154efaa053a776903ae25e`. Gói tải về có 85 tệp, gồm báo cáo, sáu hình, bộ mô hình và dự đoán. Đã đối chiếu mã lần chạy giữa runtime, summary và bundle, mã chuẩn bị dữ liệu trong manifest và SHA-256 của các tệp có trong danh mục kiểm tra.

| Kịch bản | Mô hình điểm tốt nhất trên validation | MAE_log validation | MAE_log test | Mô hình tạo khoảng C2 | Số tin test trong khoảng ở mục tiêu 90% |
|---|---|---:|---:|---|---:|
| A, chia theo thời gian | M4 | 0,26489 | 0,26965 | M4 | 22.308/25.000 (89,232%) |
| B, quận/huyện chưa gặp | M0 | 0,56743 | 0,66583 | M4 | 24.502/25.000 (98,008%) |

Tính lại thứ hạng validation, MAE_log test và khoảng C2 từ Parquet calibration/test cho kết quả như bảng. Cả hai kịch bản có 0 khoảng đảo sau hiệu chỉnh. A thấp hơn mục tiêu độ phủ 0,768 điểm phần trăm. Độ rộng trung bình trên thang log là 1,221 ở A và 2,689 ở B.

Đã mở bundle Kaggle bằng đúng mã nguồn của phiên 4 trên Windows và gửi tin qua Streamlit. Gọi trực tiếp dịch vụ dự đoán cho 6 tin thô lấy ở đầu, giữa và cuối danh sách ID test của hai shard cho giá điểm và hai cận khớp trên thang log. Khi chạy theo batch từ dữ liệu thô, cả 25.000/25.000 tin test của A đều khớp giá điểm lẫn hai cận đã lưu, sai lệch tuyệt đối lớn nhất bằng 0 cho cả ba đầu ra. Scikit-learn phát cảnh báo phiên bản khi nạp bộ biến đổi từ Kaggle 1.6.1 sang Windows 1.9.1. Phép đối chiếu bao phủ bundle M4 và tập test của lượt này.

SHA-256 gói nguồn phiên 4 là `5b5755c00926f14436cb01db17e458e4a8473887c325802b83fd0cbad33c4bdc`. SHA-256 của tệp kết quả `cs114_2003672eb5154efaa053a776903ae25e.tar.gz` là `05cb7681abb15b3138537d46d3011283d9d6e3be8d4b9bd4c735960766c0175b`. Trên máy kiểm chứng, bộ kết quả và mã nguồn tương ứng nằm trong `E:\CS114-validation\kaggle-v4-2003672eb5154efaa053a776903ae25e`.

Phiên 5 đã được gửi với mã nguồn v1.2.2, SHA-256 `c9a2e657cf08bbe92f1d83ac3849cad431b43e9698d10b44bf6038a878b46606`. Các số liệu trong bảng trên thuộc phiên 4.

## Windows local_smoke trên 10 shard của v1.2.2

Prepare ID `7e6f59b08fa448bab5af93fe2b2e03a6`. Trong 2.967.493 tin sạch có 1.917.240 cụm, trong đó 97.402 cụm chứa nhiều mức giá bảng. A và B đều đạt ngân sách 30.000/6.000/6.000/6.000 dòng train/validation/calibration/test. Cả hai tập qua gate strict với 0 dòng rò giá, 0 ID trùng và 0 cụm trùng giữa các phần chia. B có 0 quận/huyện trùng. Pipeline đã xuất 300 dòng kiểm nhãn và 100 dòng kiểm che giá để đánh giá thủ công.

PhoBERT đã ghi 94.834 embedding thuộc hợp ID của A và B với đúng prepare ID. Trên GPU Windows, riêng bước mã hóa đạt 152,34 tin/giây, mất 622,53 giây và dùng tối đa 803,98 MiB bộ nhớ CUDA theo bộ đếm của PyTorch.

## Kiểm lại khoảng giá trên toàn bộ Parquet v1.2.2

Số tin đầu vào là 3.500.744. Phép quét bộ dò tiền và phép đối chiếu giá bảng đã đọc lại đủ 2.967.493 dòng sạch, với 0 dòng bị đánh dấu. Phép tìm khoảng giá dưới đây dùng biểu thức riêng để kiểm tra điểm mù đã gặp ở v1.2.1.

Phép tìm sau chạy trực tiếp trên cột văn bản sau khi che. Đặt `VN_RE_PROJECT_ROOT` trỏ tới lượt chạy cần kiểm tra. Trên Windows, kết quả của v1.2.2 gồm 2.967.493 dòng đã đọc và hai ID `07_0189338`, `08_0083668`. Đối chiếu hai tin gốc xác nhận cả hai ghi “tầng3”.

```python
import os
from pathlib import Path
import pyarrow.compute as pc
import pyarrow.parquet as pq

root = Path(os.environ["VN_RE_PROJECT_ROOT"])
pattern = r"(?i)(?:căn|lô|tầng|diện tích)\s*\d+(?:[.,]\d+)?\s*(?:[-–—~]|đến|tới)\s*MASK_GIA"
ids = []
rows = 0
with pq.ParquetFile(root / "data/processed/clean.parquet") as table:
    for batch in table.iter_batches(columns=["row_id", "text_masked"]):
        matches = pc.match_substring_regex(batch.column("text_masked"), pattern=pattern)
        ids.extend(batch.filter(matches).column("row_id").to_pylist())
        rows += batch.num_rows
    assert rows == table.metadata.num_rows
print(rows, ids)
```
