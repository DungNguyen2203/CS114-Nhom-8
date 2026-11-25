# Báo cáo Dữ liệu & Kiểm định Nhãn (TV1)

- **Dataset:** `tinixai/vietnam-real-estates` (revision `ca3fedcbb089bf65f7e4cfb03316cce4bd0d779f`)
- **Rules version:** `v1.0.2`
- **Số shard đã xử lý:** 10

## 1. Bảng Chuyển đổi Phễu lọc Dữ liệu

| Bước / Nhóm | Số dòng | Tỷ lệ (%) |
|---|---:|---:|
| Tổng dòng thô | 3,500,744 | 100.00% |
| Tập sạch dùng huấn luyện/đánh giá (`clean`) | 3,003,979 | 85.81% |
| Tập mâu thuẫn nhãn (`noisy_flagged`) | 109,288 | 3.12% |
| Tin cho thuê (`rental`) | 60,364 | 1.72% |
| Thiếu giá / thỏa thuận (`missing_or_negotiable`) | 261,232 | 7.46% |
| Gộp nhiều BĐS / không rõ (`multi_or_unknown`) | 65,881 | 1.88% |
| Tin có dấu hiệu lệch địa giới cũ/mới (07/2025) | 58,715 | 1.68% |

## 2. Phân bố Tập Sạch theo Tháng & Loại hình

### Theo tháng (`published_at`)
| Tháng | Số mẫu sạch |
|---|---:|
| 2025-06 | 233,112 |
| 2025-07 | 320,072 |
| 2025-08 | 308,065 |
| 2025-09 | 299,108 |
| 2025-10 | 354,461 |
| 2025-11 | 347,448 |
| 2025-12 | 377,955 |
| 2026-01 | 238,847 |
| 2026-02 | 160,145 |
| 2026-03 | 364,766 |

### Theo loại hình (`property_type_name`)
| Loại hình | Số mẫu sạch | Tỷ lệ (%) |
|---|---:|---:|
| Nhà | 1,416,878 | 47.17% |
| Đất | 699,563 | 23.29% |
| Căn hộ chung cư | 625,448 | 20.82% |
| Biệt thự/Nhà liền kề | 219,100 | 7.29% |
| Shophouse | 42,990 | 1.43% |

### Top 10 Tỉnh/Thành (tập sạch)
| Tỉnh/Thành | Số mẫu sạch | Tỷ lệ (%) |
|---|---:|---:|
| Hồ Chí Minh | 1,214,904 | 40.44% |
| Hà Nội | 974,105 | 32.43% |
| Đà Nẵng | 148,606 | 4.95% |
| Bình Dương | 122,463 | 4.08% |
| Khánh Hòa | 89,653 | 2.98% |
| Hải Phòng | 59,278 | 1.97% |
| Hưng Yên | 51,222 | 1.71% |
| Đồng Nai | 49,947 | 1.66% |
| Bà Rịa - Vũng Tàu | 41,316 | 1.38% |
| Long An | 37,990 | 1.26% |

## 3. Khử trùng & Splits
- Số nhóm trùng duy nhất trên `clean`: **1,973,555** / 3,003,979 dòng.
- Nhóm trùng có thay đổi giá giữa các lần đăng lại: **93,708**.
- Kịch bản A (mẫu cuối sau budget): {'train': 30000, 'val': 6000, 'cal': 6000, 'test': 6000}
- Kịch bản B (mẫu cuối sau budget): {'train': 30000, 'val': 6000, 'cal': 6000, 'test': 6000}

## 4. Ngưỡng phân vị giá/m² học từ Train (Kịch bản A) & Tỷ lệ ngoài miền
| Loại hình | Split | N | Train Q0.5% (VND/m²) | Train Q99.5% (VND/m²) | Ngoài miền Train |
|---|---|---:|---:|---:|---:|
| Biệt thự/Nhà liền kề | train | 1,994 | 9,183,078 | 678,490,815 | 20 (1.00%) |
| Biệt thự/Nhà liền kề | val | 390 | 9,183,078 | 678,490,815 | 2 (0.51%) |
| Biệt thự/Nhà liền kề | cal | 461 | 9,183,078 | 678,490,815 | 6 (1.30%) |
| Biệt thự/Nhà liền kề | test | 413 | 9,183,078 | 678,490,815 | 4 (0.97%) |
| Căn hộ chung cư | train | 5,752 | 16,977,715 | 363,315,546 | 58 (1.01%) |
| Căn hộ chung cư | val | 1,281 | 16,977,715 | 363,315,546 | 7 (0.55%) |
| Căn hộ chung cư | cal | 1,395 | 16,977,715 | 363,315,546 | 10 (0.72%) |
| Căn hộ chung cư | test | 1,486 | 16,977,715 | 363,315,546 | 13 (0.87%) |
| Nhà | train | 14,489 | 8,665,593 | 960,000,000 | 145 (1.00%) |
| Nhà | val | 2,850 | 8,665,593 | 960,000,000 | 35 (1.23%) |
| Nhà | cal | 2,725 | 8,665,593 | 960,000,000 | 25 (0.92%) |
| Nhà | test | 2,664 | 8,665,593 | 960,000,000 | 26 (0.98%) |
| Shophouse | train | 392 | 16,479,558 | 569,035,566 | 4 (1.02%) |
| Shophouse | val | 77 | 16,479,558 | 569,035,566 | 2 (2.60%) |
| Shophouse | cal | 91 | 16,479,558 | 569,035,566 | 0 (0.00%) |
| Shophouse | test | 80 | 16,479,558 | 569,035,566 | 0 (0.00%) |
| Đất | train | 7,373 | 745,649 | 392,193,333 | 74 (1.00%) |
| Đất | val | 1,402 | 745,649 | 392,193,333 | 17 (1.21%) |
| Đất | cal | 1,328 | 745,649 | 392,193,333 | 18 (1.36%) |
| Đất | test | 1,357 | 745,649 | 392,193,333 | 15 (1.11%) |

## 5. Kiểm tra Tự động Ẩn số tiền & Xuất mẫu Audit
- `audit_labels_300.csv`: 300 dòng (đã che PII).
- `audit_masking_100.csv`: 100 dòng, số dòng còn sót ký hiệu tiền: **0**.
