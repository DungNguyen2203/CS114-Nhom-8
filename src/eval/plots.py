"""Visualization generator for TV4 experiment reports and Streamlit evaluation dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def generate_all_figures(outputs_dir: Path, splits_dir: Path) -> list[Path]:
    """Generate publication-ready charts from saved experiment outputs."""
    figs_dir = outputs_dir / "figures"
    metrics_dir = outputs_dir / "metrics"
    figs_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Data funnel chart
    split_manifest_path = splits_dir / "split_manifest.json"
    if split_manifest_path.exists():
        sm = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        funnel = sm.get("funnel", {})
        labels = ["Raw", "Clean", "Rental", "Missing/Neg", "Noisy", "Multi/Unk"]
        vals = [
            funnel.get("total_raw", 0),
            funnel.get("clean", 0),
            funnel.get("rental", 0),
            funnel.get("missing_or_negotiable", 0),
            funnel.get("noisy", 0),
            funnel.get("multi_or_unknown", 0),
        ]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.barh(labels[::-1], vals[::-1], color="#2b6cb0")
        ax.set_title("Phễu lọc & Phân loại dữ liệu (TV1)")
        ax.set_xlabel("Số tin đăng")
        for b in bars:
            w = b.get_width()
            ax.text(w, b.get_y() + b.get_height() / 2, f" {int(w):,}", va="center", fontsize=9)
        fig.tight_layout()
        p1 = figs_dir / "fig_01_funnel.png"
        fig.savefig(p1, dpi=160)
        plt.close(fig)
        saved.append(p1)

    # 2. E1 & E4 model comparison
    summary_path = metrics_dir / "model_comparison.csv"
    if summary_path.exists():
        mc = pd.read_csv(summary_path)
        test_mc = mc[mc["split"] == "test"].copy()
        if len(test_mc) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            sns.barplot(data=test_mc, x="model_id", y="MAE_log", hue="scenario", ax=axes[0])
            axes[0].set_title("MAE trên log(giá) — Test (thấp hơn là tốt hơn)")
            axes[0].set_ylabel("MAE_log")
            sns.barplot(data=test_mc, x="model_id", y="within_10_pct", hue="scenario", ax=axes[1])
            axes[1].set_title("Tỷ lệ dự đoán trong ±10% — Test (cao hơn là tốt hơn)")
            axes[1].set_ylabel("Tỷ lệ (%)")
            fig.tight_layout()
            p2 = figs_dir / "fig_02_model_comparison.png"
            fig.savefig(p2, dpi=160)
            plt.close(fig)
            saved.append(p2)

    # 3. E3 Leakage & Ablation
    e3_path = metrics_dir / "e3_ablation.csv"
    if e3_path.exists():
        e3 = pd.read_csv(e3_path)
        if len(e3) > 0:
            fig, ax = plt.subplots(figsize=(10.5, 4.8))
            sns.barplot(data=e3, x="variant", y="MAE_log", color="#dd6b20", ax=ax)
            ax.set_title("E3: Đối chứng rò rỉ giá & đóng góp mô tả (Scenario A Test)")
            ax.set_ylabel("MAE_log")
            ax.tick_params(axis="x", rotation=22)
            for tick in ax.get_xticklabels():
                tick.set_horizontalalignment("right")
            fig.tight_layout()
            p3 = figs_dir / "fig_03_e3_leakage_ablation.png"
            fig.savefig(p3, dpi=160)
            plt.close(fig)
            saved.append(p3)

    # 4. E5 Conformal methods
    conf_path = metrics_dir / "conformal_summary.csv"
    if conf_path.exists():
        cf = pd.read_csv(conf_path)
        if len(cf) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            sns.barplot(data=cf, x="method", y="coverage", hue="scenario", ax=axes[0])
            axes[0].axhline(0.90, color="red", linestyle="--", label="Mục tiêu 90%")
            axes[0].set_ylim(0, 1.05)
            axes[0].set_title("Độ phủ thực tế (Conformal C0–C3 & Stale Cal)")
            sns.barplot(data=cf, x="method", y="mean_width_log", hue="scenario", ax=axes[1])
            axes[1].set_title("Độ rộng trung bình trên thang log")
            fig.tight_layout()
            p4 = figs_dir / "fig_04_conformal_methods.png"
            fig.savefig(p4, dpi=160)
            plt.close(fig)
            saved.append(p4)

    # 5. Coverage by month and region (C2 Scenario A)
    sub_path = metrics_dir / "conformal_subgroups_a_c2.csv"
    if sub_path.exists():
        sb = pd.read_csv(sub_path)
        reg = sb[sb["group_dim"] == "region_bucket"]
        ptype = sb[sb["group_dim"] == "property_type_name"]
        if len(reg) > 0 and len(ptype) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            sns.barplot(data=reg, x="group_val", y="coverage", color="#3182ce", ax=axes[0])
            axes[0].axhline(0.90, color="red", linestyle="--")
            axes[0].set_ylim(0, 1.05)
            axes[0].set_title("Độ phủ C2 theo Khu vực (HN / TP.HCM / Tỉnh khác)")
            sns.barplot(data=ptype, x="group_val", y="coverage", color="#38a169", ax=axes[1])
            axes[1].axhline(0.90, color="red", linestyle="--")
            axes[1].set_ylim(0, 1.05)
            axes[1].set_title("Độ phủ C2 theo Loại hình BĐS")
            axes[1].tick_params(axis="x", rotation=20)
            fig.tight_layout()
            p5 = figs_dir / "fig_05_coverage_subgroups.png"
            fig.savefig(p5, dpi=160)
            plt.close(fig)
            saved.append(p5)

    # 6. Feature importance (top 15)
    imp_path = metrics_dir / "feature_importance_val.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path).head(15)
        if len(imp) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=imp, y="feature", x="mean_abs_shap", color="#5a67d8", ax=ax)
            ax.set_title("Top 15 Đặc trưng ảnh hưởng (Mean |SHAP| trên Validation)")
            ax.set_xlabel("Mean |SHAP| (log price)")
            fig.tight_layout()
            p6 = figs_dir / "fig_06_feature_importance.png"
            fig.savefig(p6, dpi=160)
            plt.close(fig)
            saved.append(p6)

    return saved
