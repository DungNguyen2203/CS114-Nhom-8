import numpy as np
import pandas as pd

from src.data.dedup import assign_duplicate_groups
from src.data.splits import build_scenario_a_split, build_scenario_b_split


def _make_sample_clean_df():
    rows = [
        # Duplicate pair across months (different price, identical masked text & specs)
        ("00_0000001", "2025-06-10", "Hà Nội", "Cầu Giấy", "Hà Nội|Cầu Giấy", "Nhà", 50.0, 4.0, 3.0, 7.0e9, "bán nhà cầu giấy MASK_GIA 50m2 4 tầng"),
        ("02_0000002", "2025-11-15", "Hà Nội", "Cầu Giấy", "Hà Nội|Cầu Giấy", "Nhà", 50.0, 4.0, 3.0, 7.2e9, "bán nhà cầu giấy MASK_GIA 50m2 4 tầng"),
        # Near duplicate in same district
        ("01_0000003", "2025-07-12", "Hà Nội", "Đống Đa", "Hà Nội|Đống Đa", "Nhà", 60.0, 5.0, 4.0, 10.0e9, "bán nhà thái thịnh đống đa MASK_GIA 60m2 5 tầng ô tô đỗ cửa"),
        ("03_0000004", "2026-03-01", "Hà Nội", "Đống Đa", "Hà Nội|Đống Đa", "Nhà", 60.0, 5.0, 4.0, 10.2e9, "bán nhà thái thịnh đống đa MASK_GIA 60m2 5 tầng ô tô đỗ cửa nhé"),
        # Unique listings across months and 10 districts for Scenario B
        ("00_0000005", "2025-08-05", "Hồ Chí Minh", "Quận 1", "Hồ Chí Minh|Quận 1", "Căn hộ chung cư", 70.0, np.nan, 2.0, 8.0e9, "căn hộ quận 1 MASK_GIA 70m2"),
        ("04_0000006", "2025-11-20", "Hồ Chí Minh", "Quận 7", "Hồ Chí Minh|Quận 7", "Căn hộ chung cư", 72.0, np.nan, 2.0, 6.3e9, "căn hộ sky garden quận 7 MASK_GIA 72m2"),
        ("05_0000007", "2026-01-10", "Hồ Chí Minh", "Quận 8", "Hồ Chí Minh|Quận 8", "Nhà", 48.0, 4.0, 3.0, 9.1e9, "nhà mặt tiền tạ quang bửu quận 8 MASK_GIA"),
        ("06_0000008", "2026-02-14", "Đà Nẵng", "Hải Châu", "Đà Nẵng|Hải Châu", "Đất", 100.0, np.nan, np.nan, 5.0e9, "đất hải châu đà nẵng MASK_GIA 100m2"),
        ("07_0000009", "2026-03-18", "Đà Nẵng", "Ngũ Hành Sơn", "Đà Nẵng|Ngũ Hành Sơn", "Đất", 120.0, np.nan, np.nan, 4.5e9, "đất ngũ hành sơn MASK_GIA 120m2"),
        ("08_0000010", "2026-03-22", "Hải Phòng", "Ngô Quyền", "Hải Phòng|Ngô Quyền", "Nhà", 55.0, 3.0, 3.0, 3.5e9, "nhà ngô quyền hải phòng MASK_GIA 55m2"),
        ("08_0000011", "2025-09-10", "Bình Dương", "Dĩ An", "Bình Dương|Dĩ An", "Đất", 90.0, np.nan, np.nan, 3.0e9, "đất dĩ an bình dương MASK_GIA 90m2"),
        ("09_0000012", "2025-10-12", "Khánh Hòa", "Nha Trang", "Khánh Hòa|Nha Trang", "Shophouse", 110.0, 4.0, 4.0, 11.0e9, "shophouse nha trang MASK_GIA 110m2"),
        ("09_0000013", "2025-12-01", "Cần Thơ", "Ninh Kiều", "Cần Thơ|Ninh Kiều", "Nhà", 80.0, 3.0, 3.0, 4.2e9, "nhà ninh kiều cần thơ MASK_GIA 80m2"),
        ("09_0000014", "2026-01-25", "Đồng Nai", "Biên Hòa", "Đồng Nai|Biên Hòa", "Đất", 100.0, np.nan, np.nan, 2.5e9, "đất biên hòa đồng nai MASK_GIA 100m2"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "row_id",
            "published_at",
            "province_norm",
            "district_norm",
            "district_key",
            "property_type_name",
            "area",
            "floor_count",
            "bedroom_count",
            "price",
            "text_masked",
        ],
    )


def test_dedup_groups_exact_and_near_duplicates_ignoring_price():
    df = _make_sample_clean_df()
    groups = assign_duplicate_groups(df, threshold=0.80, num_perm=64)
    assert len(groups) == len(df)
    assert groups.iloc[0] == groups.iloc[1]
    assert groups.iloc[2] == groups.iloc[3]
    assert groups.iloc[0] != groups.iloc[4]


def test_scenario_a_keeps_earliest_duplicate_and_chronological_splits():
    df = _make_sample_clean_df()
    df["dup_group_id"] = assign_duplicate_groups(df, threshold=0.80, num_perm=64)
    df["month"] = df["published_at"].str.slice(0, 7)
    df["y_true"] = np.log(df["price"])
    split_cfg = {
        "train_months": ["2025-06", "2025-07", "2025-08", "2025-09", "2025-10"],
        "val_months": ["2025-11", "2025-12"],
        "cal_months": ["2026-01", "2026-02"],
        "test_months": ["2026-03"],
    }
    man_a = build_scenario_a_split(df, split_cfg)
    # Reposts from later months (02_0000002 in 2025-11, 03_0000004 in 2026-03) must NOT appear!
    assert "00_0000001" in set(man_a["row_id"])
    assert "02_0000002" not in set(man_a["row_id"])
    assert "01_0000003" in set(man_a["row_id"])
    assert "03_0000004" not in set(man_a["row_id"])
    assert man_a["dup_group_id"].is_unique
    assert set(man_a[man_a["split"] == "train"]["month"]).issubset(set(split_cfg["train_months"]))
    assert set(man_a[man_a["split"] == "test"]["month"]).issubset(set(split_cfg["test_months"]))


def test_dedup_uses_full_text_and_confirms_lsh_candidates(monkeypatch, tmp_path):
    from src.data.dedup import deduplicate_parquet

    df = _make_sample_clean_df().iloc[[0, 0, 0]].copy().reset_index(drop=True)
    df["row_id"] = ["a", "b", "c"]
    prefix = "phân lô " * 60
    df["text_masked"] = [prefix + "nhà cạnh chợ", prefix + "biệt thự ven hồ", prefix + "nhà cạnh chợ"]
    # An LSH collision is only a candidate; force one with a very different suffix.
    monkeypatch.setattr("src.data.dedup.MinHashLSH.query", lambda self, mh: [0] if len(self.keys) else [])
    direct = assign_duplicate_groups(df)
    assert direct[0] != direct[1]
    assert direct[0] == direct[2]
    path = tmp_path / "clean.parquet"
    df.to_parquet(path, index=False)
    disk = deduplicate_parquet(path, list(df), 0.85, 64, 3).set_index("row_id")
    assert disk.at["a", "dup_group_id"] == disk.at["c", "dup_group_id"] != disk.at["b", "dup_group_id"]


def test_stratified_sampling_advances_one_generator_for_equal_strata():
    from src.data.splits import stratified_sample_df

    frame = pd.DataFrame({"tier": np.repeat(["a", "b"], 10), "pos": list(range(10)) * 2})
    sample = stratified_sample_df(frame, "tier", 10)
    assert set(sample.loc[sample.tier == "a", "pos"]) != set(sample.loc[sample.tier == "b", "pos"])
    pd.testing.assert_frame_equal(sample, stratified_sample_df(frame, "tier", 10))


def test_scenario_b_has_zero_district_and_dup_overlap():
    df = _make_sample_clean_df()
    df["dup_group_id"] = assign_duplicate_groups(df, threshold=0.80, num_perm=64)
    df["month"] = df["published_at"].str.slice(0, 7)
    df["y_true"] = np.log(df["price"])
    man_b = build_scenario_b_split(df, {"ratios": [0.5, 0.2, 0.15, 0.15]}, seed=42)
    splits = man_b["split"].unique()
    assert set(["train", "val", "cal", "test"]).issubset(set(splits))
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1 :]:
            d1 = set(man_b[man_b["split"] == s1]["district_key"])
            d2 = set(man_b[man_b["split"] == s2]["district_key"])
            g1 = set(man_b[man_b["split"] == s1]["dup_group_id"])
            g2 = set(man_b[man_b["split"] == s2]["dup_group_id"])
            r1 = set(man_b[man_b["split"] == s1]["row_id"])
            r2 = set(man_b[man_b["split"] == s2]["row_id"])
            assert d1.isdisjoint(d2)
            assert g1.isdisjoint(g2)
            assert r1.isdisjoint(r2)


def test_stratified_sample_df_is_shuffled_not_grouped_by_stratum():
    from src.data.splits import stratified_sample_df

    df = pd.DataFrame(
        {
            "row_id": [f"00_{i:04d}" for i in range(120)],
            "province_norm": ["Hà Nội"] * 40 + ["Hồ Chí Minh"] * 40 + ["Đà Nẵng"] * 40,
        }
    )
    sampled = stratified_sample_df(df, "province_norm", n_total=30, seed=42)
    assert len(sampled) == 30
    assert sampled["province_norm"].value_counts().to_dict() == {"Hà Nội": 10, "Hồ Chí Minh": 10, "Đà Nẵng": 10}
    assert sampled.iloc[:10]["province_norm"].nunique() >= 2
