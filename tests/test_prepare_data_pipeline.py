import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import load_config, resolve_project_paths
from src.pipelines.prepare_data import run_prepare_data


def _write_synthetic_shards(raw_dir: Path) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    shard0_rows = [
        {
            "name": "Bán nhà mặt tiền 4 tầng giá 9 tỷ 100 triệu",
            "description": "Nhà 4 tầng KDC Tạ Quang Bửu P5 Q8, dt 4m x 12m, LH 0912345678",
            "property_type_name": "Nhà",
            "province_name": "TP. HCM",
            "district_name": "8",
            "ward_name": "5",
            "street_name": "Tạ Quang Bửu",
            "project_name": None,
            "price": 9_100_000_000.0,
            "area": 48.0,
            "floor_count": 4.0,
            "frontage_width": 4.0,
            "house_depth": 12.0,
            "road_width": 8.0,
            "bedroom_count": 3.0,
            "bathroom_count": 3.0,
            "house_direction": "Đông - Nam",
            "balcony_direction": "Nam",
            "published_at": "2025-06-10T10:00:00",
        },
        {
            "name": "Cho thuê căn hộ 12 triệu/tháng",
            "description": "Căn hộ đẹp 50m2 full nội thất",
            "property_type_name": "Căn hộ chung cư",
            "province_name": "Hà Nội",
            "district_name": "Cầu Giấy",
            "ward_name": "Dịch Vọng",
            "street_name": "Xuân Thủy",
            "project_name": "Indochina",
            "price": 12_000_000.0,
            "area": 50.0,
            "floor_count": None,
            "frontage_width": None,
            "house_depth": None,
            "road_width": None,
            "bedroom_count": 2.0,
            "bathroom_count": 1.0,
            "house_direction": None,
            "balcony_direction": None,
            "published_at": "2025-06-12T10:00:00",
        },
        {
            "name": "Bán nhà 2.050 tỷ đẹp",
            "description": "Diện tích 50m2 3 tầng ngõ ô tô",
            "property_type_name": "Nhà",
            "province_name": "Hà Nội",
            "district_name": "Đống Đa",
            "ward_name": "Thịnh Quang",
            "street_name": "Thái Thịnh",
            "project_name": None,
            "price": 20_500_000_000.0,
            "area": 50.0,
            "floor_count": 3.0,
            "frontage_width": 4.0,
            "house_depth": None,
            "road_width": None,
            "bedroom_count": 3.0,
            "bathroom_count": 2.0,
            "house_direction": None,
            "balcony_direction": None,
            "published_at": "2025-07-01T10:00:00",
        },
        {
            "name": "Bán căn hộ Sky Garden giá 6.3 tỷ, HDT 20tr/t",
            "description": "Diện tích 72m2, 3PN 2WC, giá bán 6.3 tỷ",
            "property_type_name": "Căn hộ chung cư",
            "province_name": "Hồ Chí Minh",
            "district_name": "7",
            "ward_name": "Tân Phong",
            "street_name": "Phạm Văn Nghị",
            "project_name": "Sky Garden 3",
            "price": 6_300_000_000.0,
            "area": 72.0,
            "floor_count": None,
            "frontage_width": None,
            "house_depth": None,
            "road_width": None,
            "bedroom_count": 3.0,
            "bathroom_count": 2.0,
            "house_direction": "Tây",
            "balcony_direction": "Nam",
            "published_at": "2025-08-15T10:00:00",
        },
        {
            "name": "Bán đất Dĩ An 90m2 giá 3 tỷ",
            "description": "Đất P. Dĩ An, TP. HCM sổ hồng riêng",
            "property_type_name": "Đất",
            "province_name": "Bình Dương",
            "district_name": "Dĩ An",
            "ward_name": "Dĩ An",
            "street_name": "Đoàn Thị Kia",
            "project_name": None,
            "price": 3_000_000_000.0,
            "area": 90.0,
            "floor_count": None,
            "frontage_width": 5.0,
            "house_depth": None,
            "road_width": 6.0,
            "bedroom_count": None,
            "bathroom_count": None,
            "house_direction": "Nam",
            "balcony_direction": None,
            "published_at": "2025-09-11T10:00:00",
        },
        {
            "name": "Bán đất Hòa Vang 100m2 giá 2.9 tỷ",
            "description": "Lô đất đẹp 100m2 đường 5.5m",
            "property_type_name": "Đất",
            "province_name": "Đà Nẵng",
            "district_name": "Hòa Vang",
            "ward_name": "Hòa Liên",
            "street_name": "Nguyễn Tất Thành",
            "project_name": None,
            "price": 2_900_000_000.0,
            "area": 100.0,
            "floor_count": None,
            "frontage_width": 5.0,
            "house_depth": None,
            "road_width": 5.5,
            "bedroom_count": None,
            "bathroom_count": None,
            "house_direction": "Tây Bắc",
            "balcony_direction": None,
            "published_at": "2025-10-03T10:00:00",
        },
    ]
    shard1_rows = [
        # Duplicate of shard0[0] appearing in val month (should be dropped in Scenario A!)
        {**shard0_rows[0], "published_at": "2025-11-12T10:00:00", "price": 9_300_000_000.0},
        {
            "name": "Bán nhà Hoàng Mai 33m2 chào 7.95 tỷ",
            "description": "Nhà 4 tầng lô góc 33m2 ngõ ô tô",
            "property_type_name": "Nhà",
            "province_name": "Hà Nội",
            "district_name": "Hoàng Mai",
            "ward_name": "Định Công",
            "street_name": "Định Công Thượng",
            "project_name": None,
            "price": 7_950_000_000.0,
            "area": 33.0,
            "floor_count": 4.0,
            "frontage_width": 4.6,
            "house_depth": None,
            "road_width": None,
            "bedroom_count": 3.0,
            "bathroom_count": 3.0,
            "house_direction": None,
            "balcony_direction": None,
            "published_at": "2025-11-20T10:00:00",
        },
        {
            "name": "Bán shophouse Nha Trang 120m2 giá 11 tỷ",
            "description": "Shophouse 4 tầng mặt tiền rộng 6m",
            "property_type_name": "Shophouse",
            "province_name": "Khánh Hòa",
            "district_name": "Nha Trang",
            "ward_name": "Lộc Thọ",
            "street_name": "Trần Phú",
            "project_name": None,
            "price": 11_000_000_000.0,
            "area": 120.0,
            "floor_count": 4.0,
            "frontage_width": 6.0,
            "house_depth": None,
            "road_width": 12.0,
            "bedroom_count": 4.0,
            "bathroom_count": 4.0,
            "house_direction": "Đông",
            "balcony_direction": None,
            "published_at": "2025-12-05T10:00:00",
        },
        {
            "name": "Bán biệt thự Hải Châu 200m2 giá 22 tỷ",
            "description": "Biệt thự 3 tầng đường 10m khu vip",
            "property_type_name": "Biệt thự/Nhà liền kề",
            "province_name": "Đà Nẵng",
            "district_name": "Hải Châu",
            "ward_name": "Hải Châu 1",
            "street_name": "Bạch Đằng",
            "project_name": None,
            "price": 22_000_000_000.0,
            "area": 200.0,
            "floor_count": 3.0,
            "frontage_width": 10.0,
            "house_depth": 20.0,
            "road_width": 10.0,
            "bedroom_count": 5.0,
            "bathroom_count": 5.0,
            "house_direction": "Đông Nam",
            "balcony_direction": None,
            "published_at": "2026-01-15T10:00:00",
        },
        {
            "name": "Bán nhà Bình Thạnh 60m2 giá 8.5 tỷ",
            "description": "Nhà 3 tầng hẻm xe hơi",
            "property_type_name": "Nhà",
            "province_name": "Hồ Chí Minh",
            "district_name": "Bình Thạnh",
            "ward_name": "25",
            "street_name": "Điện Biên Phủ",
            "project_name": None,
            "price": 8_500_000_000.0,
            "area": 60.0,
            "floor_count": 3.0,
            "frontage_width": 4.0,
            "house_depth": 15.0,
            "road_width": 5.0,
            "bedroom_count": 3.0,
            "bathroom_count": 3.0,
            "house_direction": None,
            "balcony_direction": None,
            "published_at": "2026-02-18T10:00:00",
        },
        {
            "name": "Bán nhà Tân Bình 48m2 giá 6.5 tỷ",
            "description": "Hẻm xe hơi 48m2 3 tầng chính chủ",
            "property_type_name": "Nhà",
            "province_name": "Hồ Chí Minh",
            "district_name": "Tân Bình",
            "ward_name": "10",
            "street_name": "Trần Văn Quang",
            "project_name": None,
            "price": 6_500_000_000.0,
            "area": 48.0,
            "floor_count": 3.0,
            "frontage_width": 4.0,
            "house_depth": 12.0,
            "road_width": 5.0,
            "bedroom_count": 3.0,
            "bathroom_count": 3.0,
            "house_direction": None,
            "balcony_direction": None,
            "published_at": "2026-03-20T10:00:00",
        },
        {
            "name": "Bán đất Long Biên 70m2 giá 7 tỷ",
            "description": "Đất mặt ngõ 70m2 ô tô vào nhà",
            "property_type_name": "Đất",
            "province_name": "Hà Nội",
            "district_name": "Long Biên",
            "ward_name": "Phúc Đồng",
            "street_name": None,
            "project_name": None,
            "price": 7_000_000_000.0,
            "area": 70.0,
            "floor_count": None,
            "frontage_width": 5.0,
            "house_depth": None,
            "road_width": 5.0,
            "bedroom_count": None,
            "bathroom_count": None,
            "house_direction": None,
            "balcony_direction": None,
            "published_at": "2026-03-22T10:00:00",
        },
    ]
    p0 = raw_dir / "shard_0000.parquet"
    p1 = raw_dir / "shard_0001.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(shard0_rows)), p0)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(shard1_rows)), p1)
    return [p0, p1]


def test_run_prepare_data_end_to_end_synthetic(tmp_path):
    cfg = load_config("configs/local_smoke.yaml")
    cfg["audits"]["label_audit_size"] = 5
    cfg["audits"]["mask_audit_size"] = 5
    paths = resolve_project_paths(cfg, root=tmp_path)
    # copy alias file for isolated tmp_path
    aliases_dst = tmp_path / "configs" / "location_aliases_v1.yaml"
    aliases_dst.parent.mkdir(parents=True, exist_ok=True)
    aliases_dst.write_text(Path("configs/location_aliases_v1.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    paths["aliases_path"] = aliases_dst

    shards = _write_synthetic_shards(paths["raw_dir"])
    summary = run_prepare_data(cfg, paths, raw_shards=shards)

    assert (paths["processed_dir"] / "clean.parquet").exists()
    assert (paths["processed_dir"] / "noisy_flagged.parquet").exists()
    assert (paths["processed_dir"] / "rental_unknown.parquet").exists()
    assert (paths["processed_dir"] / "splits" / "scenario_a_manifest.parquet").exists()
    assert (paths["processed_dir"] / "splits" / "scenario_b_manifest.parquet").exists()
    assert (paths["outputs_dir"] / "audits" / "audit_labels_300.csv").exists()
    assert (paths["outputs_dir"] / "audits" / "audit_masking_100.csv").exists()
    assert (paths["reports_dir"] / "data_report.md").exists()
    assert (paths["reports_dir"] / "data_dictionary.md").exists()

    clean_df = pd.read_parquet(paths["processed_dir"] / "clean.parquet")
    noisy_df = pd.read_parquet(paths["processed_dir"] / "noisy_flagged.parquet")
    rental_df = pd.read_parquet(paths["processed_dir"] / "rental_unknown.parquet")
    assert len(noisy_df) == 1  # 2.050 ty vs 20.5B
    assert len(rental_df) == 1  # 12M/month rental
    assert len(clean_df) == 11

    man_a = pd.read_parquet(paths["processed_dir"] / "splits" / "scenario_a_manifest.parquet")
    assert man_a["dup_group_id"].is_unique
    assert "01_0000000" not in set(man_a["row_id"])  # duplicate from val month dropped

    mask_audit = pd.read_csv(paths["outputs_dir"] / "audits" / "audit_masking_100.csv")
    assert (mask_audit["leaked_count"] == 0).all()
    assert summary["funnel"]["total_raw"] == 13
