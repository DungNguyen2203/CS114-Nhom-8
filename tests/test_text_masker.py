from src.data.text_masker import mask_money_text, find_leaked_money


def test_mask_money_masks_all_vietnamese_currency_forms():
    text = (
        "Bán nhà 5 tầng 3PN diện tích 60m2, mặt tiền 4.6m, ngõ 192 ô tô tránh. "
        "Giá chào 7.45 tỷ (hoặc 9 tỷ 100 triệu), giảm 500tr, đơn giá 110tr/m2, "
        "nhỉnh 8,x tỷ, đang có HDT 20tr/t, cọc 300 triệu, một tỷ rưỡi."
    )
    masked = mask_money_text(text)
    assert "MASK_GIA" in masked
    assert "7.45" not in masked
    assert "9 tỷ" not in masked
    assert "100 triệu" not in masked
    assert "500tr" not in masked
    assert "110tr" not in masked
    assert "8,x" not in masked
    assert "20tr" not in masked
    assert "300 triệu" not in masked
    assert "một tỷ rưỡi" not in masked
    # Non-money domain numbers must stay intact!
    assert "5 tầng" in masked
    assert "3PN" in masked
    assert "60m2" in masked
    assert "4.6m" in masked
    assert "ngõ 192" in masked
    assert find_leaked_money(masked) == []


def test_find_leaked_money_catches_residual_amounts():
    assert len(find_leaked_money("Còn sót giá 4.5 ty và 60tr/m2")) >= 2
    assert len(find_leaked_money("Giá 41.000.000đ/m2 và 6xxtr")) >= 2
    assert find_leaked_money("Nhà 4 tầng 50m2 ngõ 10 đường rộng 6m giá MASK_GIA") == []


def test_mask_money_preserves_tret_trong_cuchi_and_masks_dong_and_x_forms():
    text = (
        "Nhà 1 trệt 2 lầu hẻm thẳng 1 trục, Quận 6 trong hẻm, giai đoạn 1 triển khai, "
        "tỉnh lộ 2 Củ Chi. Giá 41.000.000đ/m2, lô đất chỉ 6xxtr, căn hộ 5.4xx tỷ."
    )
    masked = mask_money_text(text)
    assert "1 trệt" in masked
    assert "1 trục" in masked
    assert "6 trong" in masked
    assert "1 triển" in masked
    assert "2 Củ Chi" in masked
    assert "41.000.000" not in masked
    assert "6xxtr" not in masked
    assert "5.4xx" not in masked
    assert find_leaked_money(masked) == []


def test_mask_money_handles_trailing_punctuation_single_x_and_preserves_dong_addresses():
    text = (
        "Nhà đường 6, Đồng Mai, Hà Đông giá chỉ 4. tỷ, căn khác 13.TỶ, 6.tỷ 350, "
        "TC: 8.2.tỷ, 4,Tỷ 780, 40, tỷ, 5.8..tỷ, 4,.tỷ, 500.triệu, HĐT 45.Tr/Tháng, "
        "phí 21000 đ/m2, giá 8.x, 7.x đồng, mỹ đình quanh 6, đồng, KDC 1181 Đ Lê Văn Lương, chợ 104, Đồng Nai."
    )
    masked = mask_money_text(text)
    for leaked in ["4. tỷ", "13.TỶ", "6.tỷ 350", "8.2.tỷ", "4,Tỷ 780", "40, tỷ", "5.8..tỷ", "4,.tỷ", "500.triệu", "45.Tr", "21000 đ/m2", "8.x", "7.x đồng", "6, đồng"]:
        assert leaked not in masked, f"Should have masked {leaked}: {masked}"
    assert "Đồng Mai" in masked
    assert "Đồng Nai" in masked
    assert "1181 Đ Lê Văn Lương" in masked
    assert find_leaked_money(masked) == []



def test_mask_catches_slang_bare_nodiacritic_spoken_and_range_forms():
    cases = {
        "Giá 1.450.000.000": "1.450",
        "GIÁ BÁN: 2.190.000.000 thương lượng": "2.190",
        "giá chỉ :3.5 tỏi": "3.5",
        "9,5 TỎI": "9,5",
        "gia 5ti": "5ti",
        "5,15ti": "5,15",
        "800trieu": "800",
        "hai mươi triệu": "hai mươi",
        "ba trăm năm mươi triệu": "ba trăm",
        "giá 6-7 tỷ": "6-",
        "từ 6 đến 7 tỷ": "6 đến",
        "500 - 700tr": "500",
    }
    for raw, forbidden in cases.items():
        masked = mask_money_text(raw)
        assert "MASK_GIA" in masked, raw
        assert forbidden not in masked, (raw, masked)
        assert find_leaked_money(masked) == [], (raw, masked)


def test_mask_keeps_non_money_numbers_near_new_patterns():
    for raw in [
        "3 ti vi", "ti le xay dung 60%", "diện tích 16.000.000 m2", "LH 091.234.567", "Hotline 0523.018.708",
        "phòng ngủ 2-3 người", "nhà 4-5 tầng", "mặt tiền 4-5m", "đường 12 - 15 m", "2 Củ Chi", "năm 2025 xây mới",
    ]:
        assert mask_money_text(raw) == raw, raw
        assert find_leaked_money(raw) == [], raw


def test_masked_leftover_words_and_checker_is_independent_of_masker():
    assert mask_money_text("3 củ rưỡi") == "MASK_GIA"
    assert mask_money_text("hai tỷ bốn trăm") == "MASK_GIA"
    # The checker must flag forms on its own, not only what the masker's patterns know.
    for raw in ["giá 3.5 tỏi", "5ti", "1.450.000.000", "6- MASK_GIA", "MASK_GIA rưỡi", "hai mươi triệu"]:
        assert find_leaked_money(raw), raw


def test_redact_pii_hides_honorific_names():
    from src.data.text_normalizer import redact_pii

    out = redact_pii("Chính chủ anh Dũng bán, nhà chú Hưng cho thuê, gặp anh Tấn")
    assert "Dũng" not in out and "Hưng" not in out and "Tấn" not in out
    assert "bán" in out


def test_mask_implicit_currency_and_ranges_preserves_physical_numbers():
    cases = {
        "Giá bán: 2,85": "Giá bán: MASK_GIA",
        "Giá: 8.8 - 9.1": "Giá: MASK_GIA",
        "Giá 1,650 Lô 2": "Giá MASK_GIA Lô 2",
        "8500000000đ": "MASK_GIA",
        "Phí quản lý: 8.800 - 16.000 đ/m²": "Phí quản lý: MASK_GIA",
        "5.8-6,x tỷ": "MASK_GIA",
        "dt 4.3m x 15 - 18.x tỷ": "dt 4.3m x 15 - MASK_GIA",
        "70-80 nghìn công nhân": "70-80 nghìn công nhân",
        "Giá 70-80 m2": "Giá 70-80 m2",
        "Giá 12.5 m²": "Giá 12.5 m²",
        "115 triệu / m25 triệu / m2, 310 m2": "MASK_GIA , 310 m2",
        "Tài chính tầm 500 đến 5.000.000.000": "Tài chính tầm MASK_GIA",
        "Ngay tòa án Quận 12 - chỉ 7 đồng": "Ngay tòa án Quận 12 - MASK_GIA",
        "Giá 8.350 LÔ ĐẤT": "Giá MASK_GIA LÔ ĐẤT",
        "giá 2.65 căn 2 phòng ngủ": "giá MASK_GIA căn 2 phòng ngủ",
        "5.2xxVNĐ": "MASK_GIA",
        "1,8x VND": "MASK_GIA",
        "HĐT: 40 triệu/th18 tỷ": "HĐT: MASK_GIA",
        "7....tỷ": "MASK_GIA",
        "MƯỜI MỘT TỶ BA TRĂM": "MASK_GIA",
        "5xx nghìn/m2": "MASK_GIA",
        "10 TỶ 5 TRĂM 60 TRIỆU": "MASK_GIA",
        "ba tỷ hai phòng": "MASK_GIA hai phòng",
        "thuê nguyên căn 15 - 20 triệu/tháng": "thuê nguyên căn MASK_GIA",
        "cho thuê được luôn 2 tầng 15 - 20 triệu": "cho thuê được luôn 2 tầng MASK_GIA",
        "rẻ hơn căn cùng diện tích 300-500 triệu": "rẻ hơn căn cùng diện tích MASK_GIA",
    }
    for raw, expected in cases.items():
        assert mask_money_text(raw) == expected
        assert not find_leaked_money(expected)
    for raw in ("Giá bán: 2,85", "8500000000đ", "500k/m2", "thuê nguyên căn 15 - MASK_GIA"):
        assert find_leaked_money(raw)
