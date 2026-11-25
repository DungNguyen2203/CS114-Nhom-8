# Data Dictionary — Vietnam Real Estate Valuation

| Cột | Kiểu | Mô tả |
|---|---|---|
| `row_id` | `string` | Định danh duy nhất `SS_RRRRRRR` từ chỉ số shard và số dòng gốc |
| `price` | `float64` | Giá chào bán gốc (VND), chỉ dùng tạo nhãn `y_true = log(price)` |
| `y_true` | `float64` | Biến mục tiêu `log(price)` trên tập sạch |
| `area` | `float64` | Diện tích (m²) |
| `property_type_name` | `string` | 5 loại hình thực tế (`Nhà`, `Đất`, `Căn hộ chung cư`, `Biệt thự/Nhà liền kề`, `Shophouse`) |
| `province_norm`, `district_norm`, `ward_norm` | `string` | Địa giới chuẩn hóa theo `location_aliases_v1.yaml` |
| `district_key` | `string` | Khóa nhóm `province_norm|district_norm` cho Kịch bản B |
| `house_direction_norm`, `balcony_direction_norm` | `string` | Hướng đã chuẩn hóa (`Đông`, `Tây`, ..., `Đa hướng`, `Thiếu`) |
| `*_was_outlier`, `*_is_missing` | `int8` | Cờ ngoại lệ miền và khuyết thiếu dữ liệu bảng |
| `name_masked`, `description_masked`, `text_masked` | `string` | Văn bản đã thay mọi số tiền bằng `MASK_GIA` |
| `label_bucket`, `text_status` | `string` | Kết quả phân loại nhãn và đối chiếu giá/diện tích độc lập |
| `dup_group_id` | `string` | Mã cụm trùng từ text đã ẩn giá + vị trí + thông số vật lý (không dùng `price`) |
