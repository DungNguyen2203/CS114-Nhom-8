import numpy as np
import pandas as pd

from src.data.location_normalizer import LocationNormalizer
from src.data.tabular_cleaner import clean_tabular_domain


def test_location_normalizer_canonicalizes_districts_wards_and_directions():
    norm = LocationNormalizer("configs/location_aliases_v1.yaml")
    df = pd.DataFrame(
        {
            "province_name": ["TP. HCM", "Bình Dương", "Hà Nội"],
            "district_name": ["8", "Dĩ An", "Quận Cầu Giấy"],
            "ward_name": ["5", "P. Dĩ An", "Dịch Vọng"],
            "street_name": ["Tạ Quang Bửu", None, "Xuân Thủy"],
            "project_name": [None, "Sky Park", None],
            "house_direction": ["Đông - Nam", "Đông, Tây Nam", None],
            "balcony_direction": ["Tây - Bắc", "Nam", "Đông - Bắc"],
            "description": [
                "Nhà mặt tiền P.5 Q.8",
                "Mặt tiền chợ P. Dĩ An, TP. HCM sầm uất",
                "Ngõ rộng ô tô",
            ],
        }
    )
    out = norm.normalize_df(df)
    assert out.loc[0, "province_norm"] == "Hồ Chí Minh"
    assert out.loc[0, "district_norm"] == "Quận 8"
    assert out.loc[0, "ward_norm"] == "Phường 5"
    assert out.loc[0, "house_direction_norm"] == "Đông Nam"
    assert out.loc[0, "balcony_direction_norm"] == "Tây Bắc"
    assert out.loc[1, "house_direction_norm"] == "Đa hướng"
    assert out.loc[2, "house_direction_norm"] == "Thiếu"
    assert bool(out.loc[1, "admin_drift_flag"]) is True
    assert bool(out.loc[0, "admin_drift_flag"]) is False
    assert out.loc[0, "district_key"] == "Hồ Chí Minh|Quận 8"


def test_clean_tabular_domain_flags_extreme_outliers_and_missingness():
    df = pd.DataFrame(
        {
            "area": [50.0, 82_000_000.0],
            "frontage_width": [4.5, 9_080_000.0],
            "house_depth": [12.0, np.nan],
            "road_width": [6.0, 103_000.0],
            "floor_count": [4.0, 255.0],
            "bedroom_count": [3.0, 200.0],
            "bathroom_count": [3.0, 255.0],
        }
    )
    bounds = {
        "frontage_width": [1.0, 100.0],
        "house_depth": [1.0, 200.0],
        "road_width": [0.5, 100.0],
        "floor_count": [1.0, 60.0],
        "bedroom_count": [1.0, 50.0],
        "bathroom_count": [1.0, 50.0],
    }
    out = clean_tabular_domain(df, {"tabular_bounds": bounds})
    assert out.loc[0, "frontage_width"] == 4.5
    assert out.loc[0, "frontage_width_was_outlier"] == 0
    assert np.isnan(out.loc[1, "frontage_width"])
    assert out.loc[1, "frontage_width_was_outlier"] == 1
    assert out.loc[1, "frontage_width_is_missing"] == 1
    assert np.isnan(out.loc[1, "floor_count"])
    assert out.loc[1, "floor_count_was_outlier"] == 1
    assert out.loc[1, "house_depth_is_missing"] == 1
