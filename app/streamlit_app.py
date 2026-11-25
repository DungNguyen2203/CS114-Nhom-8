"""TV4 Streamlit demo application: property valuation, 90% conformal interval, comps, and eval dashboard."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference import SAMPLE_DEMO_LISTINGS, ValuationInferenceService


st.set_page_config(page_title="Định giá BĐS Việt Nam & Conformal", layout="wide")


@st.cache_resource
def _load_service(bundle_path: str) -> ValuationInferenceService:
    return ValuationInferenceService.load(bundle_path)


def _fmt_ty(vnd: float) -> str:
    return f"{vnd / 1e9:,.2f} tỷ VND ({vnd:,.0f} đ)"


def main() -> None:
    st.title("Định giá Giá Chào Bán Bất Động Sản Việt Nam & Khoảng Tin Cậy Conformal 90%")
    st.caption(
        "Phạm vi: Ước lượng **giá chào bán** dựa trên tin đăng giai đoạn 06/2025–03/2026 (`tinixai/vietnam-real-estates`). "
        "Không thay thế thẩm định giá chuyên nghiệp và không dùng làm căn cứ mua bán/cho vay."
    )

    bundle_dir = ROOT / "models" / "bundle_a"
    outputs_dir = ROOT / "outputs"

    tab_demo, tab_eval = st.tabs(["1. Định giá & Kiểm tra tin", "2. Bảng Kết quả & Độ phủ (E1–E6)"])

    with tab_demo:
        if not (bundle_dir / "bundle_meta.json").exists():
            st.warning("Chưa tìm thấy model bundle tại `models/bundle_a`. Hãy chạy pipeline `train_eval` trước.")
        else:
            svc = _load_service(str(bundle_dir))
            sample_names = ["(Nhập thủ công)"] + [s["label"] for s in SAMPLE_DEMO_LISTINGS]
            chosen = st.selectbox("Chọn tin mẫu minh họa (đã ẩn thông tin liên hệ):", sample_names)
            preset = next((s for s in SAMPLE_DEMO_LISTINGS if s["label"] == chosen), {})

            with st.form("valuation_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    ptype = st.selectbox(
                        "Loại hình",
                        ["Nhà", "Đất", "Căn hộ chung cư", "Biệt thự/Nhà liền kề", "Shophouse"],
                        index=["Nhà", "Đất", "Căn hộ chung cư", "Biệt thự/Nhà liền kề", "Shophouse"].index(
                            preset.get("property_type_name", "Nhà")
                        ),
                    )
                    province = st.text_input("Tỉnh / Thành phố", value=preset.get("province_name", "Hà Nội"))
                    district = st.text_input("Quận / Huyện", value=preset.get("district_name", "Cầu Giấy"))
                    ward = st.text_input("Phường / Xã", value=preset.get("ward_name", ""))
                    street = st.text_input("Tên đường", value=preset.get("street_name", ""))
                with c2:
                    area = st.number_input("Diện tích (m²)", min_value=5.0, max_value=50000.0, value=float(preset.get("area") or 50.0))
                    floor_count = st.number_input("Số tầng (0 = không rõ)", min_value=0.0, max_value=60.0, value=float(preset.get("floor_count") or 0.0))
                    bedroom_count = st.number_input("Số phòng ngủ (0 = không rõ)", min_value=0.0, max_value=50.0, value=float(preset.get("bedroom_count") or 0.0))
                    bathroom_count = st.number_input("Số phòng tắm (0 = không rõ)", min_value=0.0, max_value=50.0, value=float(preset.get("bathroom_count") or 0.0))
                    frontage = st.number_input("Mặt tiền (m, 0 = không rõ)", min_value=0.0, max_value=100.0, value=float(preset.get("frontage_width") or 0.0))
                with c3:
                    road_width = st.number_input("Đường rộng (m, 0 = không rõ)", min_value=0.0, max_value=100.0, value=float(preset.get("road_width") or 0.0))
                    house_dir = st.text_input("Hướng nhà", value=str(preset.get("house_direction") or ""))
                    balcony_dir = st.text_input("Hướng ban công", value=str(preset.get("balcony_direction") or ""))
                    project_name = st.text_input("Dự án (nếu có)", value=str(preset.get("project_name") or ""))
                    asking_ty = st.number_input(
                        "Giá chào bán kiểm tra (tỷ VND, 0 = bỏ qua)",
                        min_value=0.0,
                        max_value=2000.0,
                        value=float(preset.get("asking_price_vnd", 0.0) or 0.0) / 1e9,
                    )

                name_txt = st.text_input("Tiêu đề tin đăng", value=preset.get("name", ""))
                desc_txt = st.text_area("Mô tả tin đăng", value=preset.get("description", ""), height=120)
                submitted = st.form_submit_button("Ước lượng giá & Kiểm tra khoảng 90%")

            if submitted:
                payload = {
                    "property_type_name": ptype,
                    "province_name": province,
                    "district_name": district,
                    "ward_name": ward or None,
                    "street_name": street or None,
                    "project_name": project_name or None,
                    "area": area,
                    "floor_count": floor_count if floor_count > 0 else None,
                    "bedroom_count": bedroom_count if bedroom_count > 0 else None,
                    "bathroom_count": bathroom_count if bathroom_count > 0 else None,
                    "frontage_width": frontage if frontage > 0 else None,
                    "road_width": road_width if road_width > 0 else None,
                    "house_direction": house_dir or None,
                    "balcony_direction": balcony_dir or None,
                    "name": name_txt,
                    "description": desc_txt,
                    "asking_price_vnd": asking_ty * 1e9 if asking_ty > 0 else None,
                }
                res = svc.predict_single(payload)

                if res["low_data_warning"]:
                    st.warning(
                        f"Khu vực `{district}, {province}` chỉ có {res['district_train_count']} mẫu trong train (< 30). "
                        "Kết quả chỉ mang tính tham khảo hạn chế."
                    )

                m1, m2, m3 = st.columns(3)
                m1.metric("Giá chào ước lượng", _fmt_ty(res["pred_price_vnd"]))
                m2.metric("Cận dưới Conformal 90%", _fmt_ty(res["q_lo_vnd"]))
                m3.metric("Cận trên Conformal 90%", _fmt_ty(res["q_hi_vnd"]))
                st.caption(f"Phiên bản mô hình: `{res['model_version']}` • Độ trễ: `{res['inference_ms']:.1f} ms`")

                if res["price_position"] == "below_interval":
                    st.error("Cảnh báo G1: Giá chào nhập vào **THẤP HƠN** khoảng dự đoán 90% (cần kiểm tra lỗi nhập liệu hoặc yếu tố đặc thù).")
                elif res["price_position"] == "above_interval":
                    st.warning("Cảnh báo G1: Giá chào nhập vào **CAO HƠN** khoảng dự đoán 90% (có thể do nội thất/vị trí đặc biệt hoặc lệch giá).")
                elif res["price_position"] == "in_interval":
                    st.success("Giá chào nhập vào **NẰM TRONG** khoảng dự đoán Conformal 90%.")

                if res["text_consistency"] not in {"not_checked", "clean_consistent", "clean_no_text_price"}:
                    st.error(f"Cảnh báo G2 (đối chiếu văn bản): `{res['text_consistency']}` — giá chào mâu thuẫn với nội dung mô tả.")

                with st.expander("Văn bản sau khi ẩn mọi số tiền (đầu vào thực tế cho NLP)"):
                    st.code(res["masked_text"])

                st.subheader("Tin tham khảo tương tự (từ tập Train đã làm sạch & ẩn PII)")
                if res["similar_listings"]:
                    st.dataframe(pd.DataFrame(res["similar_listings"]), width="stretch")
                else:
                    st.info("Không tìm thấy tin tham khảo cùng loại hình trong kho train.")

    with tab_eval:
        metrics_dir = outputs_dir / "metrics"
        figs_dir = outputs_dir / "figures"
        mc_p = metrics_dir / "model_comparison.csv"
        cf_p = metrics_dir / "conformal_summary.csv"
        e3_p = metrics_dir / "e3_ablation.csv"

        if mc_p.exists():
            st.subheader("Bảng so sánh mô hình (E1 & E4 — Test)")
            mc = pd.read_csv(mc_p)
            st.dataframe(mc[mc["split"] == "test"], width="stretch")
        if cf_p.exists():
            st.subheader("Độ phủ & Độ rộng Conformal (E5 — C0–C3)")
            st.dataframe(pd.read_csv(cf_p), width="stretch")
        if e3_p.exists():
            st.subheader("Đối chứng Rò rỉ giá & Đóng góp văn bản (E3)")
            st.dataframe(pd.read_csv(e3_p), width="stretch")

        if figs_dir.exists():
            st.subheader("Biểu đồ Thực nghiệm")
            for img in sorted(figs_dir.glob("*.png")):
                st.image(str(img), caption=img.name)


if __name__ == "__main__":
    main()
