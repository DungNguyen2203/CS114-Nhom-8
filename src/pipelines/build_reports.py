"""TV4 report and visualization builder reading strictly from saved outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config, resolve_project_paths
from src.eval.plots import generate_all_figures


def run_build_reports(cfg: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    """Generate figures and compile experiment_report.md & slide_outline.md."""
    outputs_dir = paths["outputs_dir"]
    metrics_dir = outputs_dir / "metrics"
    reports_dir = paths["reports_dir"]
    splits_dir = paths["processed_dir"] / "splits"
    reports_dir.mkdir(parents=True, exist_ok=True)

    figs = generate_all_figures(outputs_dir, splits_dir)

    mc = pd.read_csv(metrics_dir / "model_comparison.csv") if (metrics_dir / "model_comparison.csv").exists() else pd.DataFrame()
    cf = pd.read_csv(metrics_dir / "conformal_summary.csv") if (metrics_dir / "conformal_summary.csv").exists() else pd.DataFrame()
    e3 = pd.read_csv(metrics_dir / "e3_ablation.csv") if (metrics_dir / "e3_ablation.csv").exists() else pd.DataFrame()
    boot = pd.read_csv(metrics_dir / "bootstrap_ci.csv") if (metrics_dir / "bootstrap_ci.csv").exists() else pd.DataFrame()
    emb_stats_p = paths["processed_dir"] / "embeddings" / "embedding_stats.json"
    emb_stats = json.loads(emb_stats_p.read_text(encoding="utf-8")) if emb_stats_p.exists() else {}
    run_sum_p = metrics_dir / "run_summary.json"
    run_sum = json.loads(run_sum_p.read_text(encoding="utf-8")) if run_sum_p.exists() else {}

    lines = [
        "# Báo cáo Thực nghiệm & Đánh giá (TV2–TV4)",
        "",
        f"- **Profile cấu hình:** `{cfg.get('runtime', {}).get('profile_name', 'base')}`",
        f"- **Mô hình thắng trên Validation:** Kịch bản A=`{run_sum.get('winner_a', run_sum.get('winner_id', 'N/A'))}`, Kịch bản B=`{run_sum.get('winner_b', 'N/A')}`",
        f"- **Kiểm tra rò rỉ split:** A=`{run_sum.get('leakage', {}).get('A')}`, B=`{run_sum.get('leakage', {}).get('B')}` (leaks A={run_sum.get('leakage', {}).get('leaks_a', 0)}, B={run_sum.get('leakage', {}).get('leaks_b', 0)})",
        "",
        "## 1. So sánh Mô hình Điểm (Kịch bản A & B — Test, seed chính)",
        "",
    ]
    if len(mc) > 0:
        primary_seed = int(cfg["project"]["primary_seed"])
        sub = mc[(mc["split"] == "test") & (mc["seed"] == primary_seed)]
        lines.append(sub[["scenario", "model_id", "n", "MAE_log", "RMSE_log", "MdAPE_pct", "within_10_pct", "within_20_pct", "WAPE_vnd_pct", "train_time_sec"]].to_markdown(index=False))

    lines.extend(["", "## 2. Thí nghiệm E3 — Đối chứng Rò rỉ Giá & Đóng góp Văn bản", ""])
    if len(e3) > 0:
        lines.append(e3.to_markdown(index=False))

    if len(boot) > 0:
        lines.extend(["", "## 3. Khoảng tin cậy Bootstrap theo Quận/Huyện (95% CI)", "", boot.to_markdown(index=False)])

    lines.extend(["", "## 4. Khoảng dự đoán Conformal (E5 — C0–C3 & Hiệu chỉnh cũ vs mới)", ""])
    if len(cf) > 0:
        cols = [c for c in ["scenario", "model_id", "method", "alpha", "coverage", "mean_width_log", "median_ratio_hi_lo", "interval_score", "crossing_rate_raw", "worst_group_coverage"] if c in cf.columns]
        lines.append(cf[cols].to_markdown(index=False))

    if emb_stats:
        lines.extend(
            [
                "",
                "## 5. Hiệu năng Hệ thống (Tách từ & PhoBERT GPU)",
                f"- Thiết bị: `{emb_stats.get('device')}`, `fp16={emb_stats.get('fp16')}`, `batch_size={emb_stats.get('batch_size')}`",
                f"- Tốc độ tách từ CPU (`pyvi`): {emb_stats.get('segmentation_sec_per_1k', 0):.2f}s / 1.000 văn bản",
                f"- Tốc độ embedding PhoBERT: {emb_stats.get('embed_texts_per_sec', 0):.1f} văn bản/s, Peak VRAM: {emb_stats.get('peak_vram_mb', 0):.1f} MB",
                f"- Tỷ lệ văn bản chạm giới hạn 256 token: {100*emb_stats.get('truncated_ratio', 0):.2f}%",
                f"- Độ trễ suy luận trung bình/tin: {run_sum.get('latency', {}).get('per_row_mean_ms', 0):.3f} ms",
            ]
        )

    lines.extend(
        [
            "",
            "## 6. Giới hạn & Phạm vi Sử dụng",
            "- Mô hình ước lượng **giá chào bán** giai đoạn `06/2025–03/2026`, không thay thế thẩm định giá hay căn cứ giao dịch/cho vay.",
            "- Dữ liệu tập trung mạnh ở Hà Nội & TP.HCM (~71%), địa giới có độ lệch trước/sau 07/2025, và độ phủ conformal có thể lệch ở nhóm hiếm hoặc khi thị trường dịch chuyển theo thời gian.",
        ]
    )
    (reports_dir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    slide_lines = [
        "# Bố cục Slide Thuyết trình (13 Slides)",
        "1. Bài toán định giá chào bán & phát hiện tin lệch giá (Introduction - 10%)",
        "2. Tổng quan Dataset & Rủi ro thực tế (không có nhãn bán/thuê, lệch địa giới 07/2025) (Dataset - 20%)",
        "3. Quy trình chuẩn hóa, tách giá độc lập & phễu lọc dữ liệu (`fig_01_funnel.png`)",
        "4. Chống rò rỉ giá (`MASK_GIA`), khử trùng MinHash & 2 kịch bản chia tập A/B",
        "5. Kiến trúc mô hình M0–M5: Bảng, TF-IDF và PhoBERT đóng băng (Methodology - 25%)",
        "6. Hồi quy phân vị & Hiệu chỉnh Conformal C0–C3",
        "7. Kết quả thực nghiệm E1 & E2: Bảng so sánh & độ đóng góp văn bản (`fig_02_model_comparison.png`)",
        "8. Thí nghiệm E3: Đo lường rò rỉ giá & feature importance (`fig_03`, `fig_06`)",
        "9. Tổng quát hóa khu vực mới (Kịch bản B) & độ phủ Conformal theo nhóm/thời gian (`fig_04`, `fig_05`)",
        "10. Gắn cờ lệch giá G1/G2 & kết quả kiểm định mẫu audit E6",
        "11. Demo ứng dụng Streamlit trực tiếp & tin tham khảo từ train",
        "12. Trả lời RQ1–RQ3, Giới hạn & Hướng phát triển (Conclusion - 10%)",
        "13. Phân công TV1–TV4, Nhật ký mốc nghiệm thu & Quản lý GPU (Time managing - 10%)",
    ]
    (reports_dir / "slide_outline.md").write_text("\n".join(slide_lines) + "\n", encoding="utf-8")
    return {"figures": [str(f) for f in figs]}


def main() -> None:
    parser = argparse.ArgumentParser(description="TV4 report and figure builder")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = resolve_project_paths(cfg)
    out = run_build_reports(cfg, paths)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
