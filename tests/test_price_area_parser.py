from src.data.text_normalizer import normalize_text, redact_pii
from src.data.price_area_parser import parse_listing_text, reconcile_labels


DEFAULT_DOMAIN_CFG = {
    "min_area_m2": 5.0,
    "max_area_m2": 50000.0,
    "min_price_vnd": 50_000_000.0,
    "max_price_vnd": 2_000_000_000_000.0,
    "min_price_per_m2_vnd": 500_000.0,
    "max_price_per_m2_vnd": 3_000_000_000.0,
    "price_mismatch_ratio_strict": 1.35,
    "price_mismatch_ratio_approx": 1.60,
    "order_of_magnitude_ratio": 5.0,
    "area_mismatch_ratio": 1.30,
}


def test_normalize_text_strips_html_and_normalizes_unicode():
    raw = "Bán\u00a0<b>nhà</b>   Hồ\u0020Chi\u0301 Minh &amp; Quận 1\r\n  giá tốt"
    norm = normalize_text(raw)
    assert "<b>" not in norm
    assert "&amp;" not in norm
    assert "Chí Minh" in norm
    assert "  " not in norm


def test_redact_pii_hides_phone_zalo_email_and_house_number():
    raw = "LH: anh Hùng 0912.345.678 hoặc zalo 0987654321, email test@gmail.com, nhà số 123/45 đường Lê Lợi"
    redacted = redact_pii(raw)
    assert "0912" not in redacted
    assert "0987654321" not in redacted
    assert "test@gmail.com" not in redacted
    assert "anh Hùng" not in redacted
    assert "123/45" not in redacted


def test_parse_sale_ty_trieu_and_decimal():
    p1 = parse_listing_text("Bán nhà 9 tỷ 100 triệu", "Diện tích 48m2, 4 tầng")
    assert p1.tx_type == "sale"
    assert abs(p1.sale_price_vnd - 9_100_000_000) < 1e3
    assert p1.area_m2 == 48.0

    p2 = parse_listing_text("Nhà phố 7.45 tỷ", "DT 37m2 ngõ ô tô")
    assert p2.tx_type == "sale"
    assert abs(p2.sale_price_vnd - 7_450_000_000) < 1e3


def test_parse_approx_and_unit_price_per_m2():
    p_approx = parse_listing_text("Bán nhà 8,x tỷ", "Nhà đẹp nhỉnh 8 tỷ")
    assert p_approx.tx_type == "sale"
    assert p_approx.is_approx is True
    assert 8_000_000_000 <= p_approx.sale_price_vnd <= 9_000_000_000

    p_unit = parse_listing_text("Bán đất 85 triệu/m2", "Diện tích 60 m2 đường nhựa", tabular_area=60.0)
    assert p_unit.tx_type == "sale"
    assert abs(p_unit.unit_price_per_m2_vnd - 85_000_000) < 1e3
    assert abs(p_unit.sale_price_vnd - 5_100_000_000) < 1e4


def test_parse_written_vietnamese_words():
    p = parse_listing_text("Bán căn hộ một tỷ rưỡi", "Nhà hai tỷ ba? Không, giá một tỷ rưỡi thôi")
    assert p.tx_type == "sale"
    assert abs(p.sale_price_vnd - 1_500_000_000) < 1e3


def test_sale_with_embedded_rental_yield_not_misclassified_as_rent():
    p = parse_listing_text(
        "Bán căn hộ 72m2, chung cư Sky Garden 3, giá 6.3 tỷ, nhà đang có HDT 20tr/t",
        "Đang cho thuê 20tr/tháng ổn định. Giá bán: 6.3 tỷ thương lượng",
        tabular_area=72.0,
    )
    assert p.tx_type == "sale"
    assert abs(p.sale_price_vnd - 6_300_000_000) < 1e3
    rec = reconcile_labels(6_300_000_000.0, 72.0, p, DEFAULT_DOMAIN_CFG)
    assert rec.bucket == "clean"
    assert rec.status == "clean_consistent"


def test_pure_rental_classified_as_rental():
    p = parse_listing_text("Cho thuê căn hộ 12 triệu/tháng", "Full nội thất, vào ở ngay, dt 50m2")
    assert p.tx_type == "rent"
    rec = reconcile_labels(12_000_000.0, 50.0, p, DEFAULT_DOMAIN_CFG)
    assert rec.bucket == "rental"


def test_order_of_magnitude_mismatch_flagged():
    p = parse_listing_text("Bán nhà 2.050 tỷ", "Nhà mới đẹp 50m2")
    rec = reconcile_labels(20_500_000_000.0, 50.0, p, DEFAULT_DOMAIN_CFG)
    assert rec.bucket == "noisy"
    assert rec.status == "noisy_order_of_magnitude"


def test_negotiable_missing_and_bundle():
    p_neg = parse_listing_text("Bán nhà giá thỏa thuận", "Liên hệ trực tiếp")
    rec_neg = reconcile_labels(None, 60.0, p_neg, DEFAULT_DOMAIN_CFG)
    assert rec_neg.bucket == "missing_or_negotiable"

    p_bundle = parse_listing_text("Bán cả cặp 2 căn nhà liền kề", "Tổng 2 căn giá 18 tỷ")
    assert p_bundle.is_bundle is True
    rec_b = reconcile_labels(18_000_000_000.0, 100.0, p_bundle, DEFAULT_DOMAIN_CFG)
    assert rec_b.bucket == "multi_or_unknown"


def test_area_from_dimensions_ignores_road_width():
    p = parse_listing_text("Bán nhà mặt tiền", "Diện tích: 4m x 12m. Mặt tiền đường 8m thông thoáng")
    assert abs(p.area_m2 - 48.0) < 1e-3
    rec = reconcile_labels(9_000_000_000.0, 48.0, p, DEFAULT_DOMAIN_CFG)
    assert rec.area_mismatch_flag is False


def test_tr_prefix_words_cu_chi_and_bedrooms_not_parsed_as_price():
    # "1 trệt", "1 trong", "6 trong", "1 triển", "2 Củ Chi" must NOT parse as 1M/6M/2M
    p1 = parse_listing_text("Bán nhà Quận 6 trong hẻm", "Kết cấu 1 trệt 2 lầu, giai đoạn 1 triển khai, tỉnh lộ 2 Củ Chi")
    assert p1.sale_price_vnd is None

    # "5 ty 2 ngủ" -> 5.0 tỷ, NOT 5.2 tỷ!
    p2 = parse_listing_text("Bán chung cư 5 ty 2 ngủ", "Căn hộ đẹp 70m2")
    assert abs(p2.sale_price_vnd - 5_000_000_000.0) < 1e3

    # "LN 1.3 tỷ/năm ... Giá bán từ 11 tỷ" -> total_sale is 11 tỷ, not 1.3 tỷ/năm!
    p3 = parse_listing_text("Biệt thự Bãi Kem - LN 1.3 tỷ/năm", "Cam kết lợi nhuận 1.3 tỷ/năm. Giá bán từ 11 tỷ", tabular_area=247.0)
    assert abs(p3.sale_price_vnd - 11_000_000_000.0) < 1e3

    # "41.000.000đ/m2" -> unit price
    p4 = parse_listing_text("Bán đất Hưng Yên 112m2", "Giá 41.000.000đ/m2 đầu tư hợp lý", tabular_area=112.0)
    assert abs(p4.unit_price_per_m2_vnd - 41_000_000.0) < 1e3



def test_sale_listing_with_embedded_tenant_rent_is_not_rental():
    for text in [
        "Bán nhà đang cho thuê 20tr/tháng, sổ hồng chính chủ",
        "Chính chủ bán nhà, hiện đang cho thuê 18 triệu/tháng",
        "Bán nhà HĐT 20tr/th",
    ]:
        assert parse_listing_text(text, "", None).tx_type == "sale", text
    assert parse_listing_text("CHO THUÊ nhà nguyên căn 15tr/tháng", "", None).tx_type == "rent"
    assert parse_listing_text("Cho thuê nhà nguyên căn 15tr/tháng, sổ hồng chính chủ", "", None).tx_type == "rent"
