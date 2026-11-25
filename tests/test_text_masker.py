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

