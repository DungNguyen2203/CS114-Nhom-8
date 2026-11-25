from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.phobert_embedder import (
    format_phobert_input,
    segment_vietnamese_texts,
    load_embeddings_for_row_ids,
    save_embedding_cache,
)
from src.features.tabular_features import TabularFeatureExtractor
from src.models.m3_phobert_ridge import M3PhoBERTRidgeModel
from src.models.m5_phobert_catboost import M5PhoBERTCatBoostModel
from tests.test_tabular_and_baselines import _make_tabular_splits


def test_segment_vietnamese_texts_preserves_mask_token():
    texts = [
        "Nhà mặt tiền quận Cầu Giấy giá MASK_GIA diện tích 60m2",
        "Căn hộ chung cư MASK_GIA có nội thất đẹp",
    ]
    seg_texts, elapsed = segment_vietnamese_texts(texts)
    assert len(seg_texts) == 2
    assert elapsed >= 0.0
    for s in seg_texts:
        assert "<mask>" in s
        assert "MASK_GIA" not in s


def test_format_phobert_input_prioritizes_core_fields():
    s = format_phobert_input(
        property_type="Nhà",
        province="Hà Nội",
        district="Cầu Giấy",
        name_masked="Bán nhà đẹp MASK_GIA",
        description_masked="Ngõ ô tô 5 tầng",
    )
    assert s.startswith("Nhà.")
    assert "Cầu Giấy, Hà Nội" in s
    assert "Bán nhà đẹp MASK_GIA" in s


def test_embedding_cache_alignment_by_row_id_and_missing_key(tmp_path):
    row_ids = ["00_0000010", "00_0000002", "00_0000005"]
    embs = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    save_embedding_cache(tmp_path, row_ids, embs, text_version="v1", model_sha="abc")

    # Request in scrambled order -> must return aligned by row_id, not positional order!
    loaded = load_embeddings_for_row_ids(["00_0000005", "00_0000010"], tmp_path)
    assert loaded.shape == (2, 2)
    assert np.allclose(loaded[0], [3.0, 4.0], atol=1e-2)
    assert np.allclose(loaded[1], [1.0, 0.0], atol=1e-2)

    with pytest.raises(KeyError):
        load_embeddings_for_row_ids(["99_9999999"], tmp_path)


def test_m3_and_m5_fit_and_predict_with_train_only_pca():
    train_df, val_df = _make_tabular_splits(n_train=100, n_val=30)
    rng = np.random.default_rng(7)
    emb_tr = rng.normal(size=(len(train_df), 32)).astype(np.float32)
    emb_va = rng.normal(size=(len(val_df), 32)).astype(np.float32)

    m3 = M3PhoBERTRidgeModel(alphas=[1.0, 10.0])
    best_a, _ = m3.tune_and_fit(emb_tr, train_df["y_true"].to_numpy(), emb_va, val_df["y_true"].to_numpy())
    assert best_a in (1.0, 10.0)
    assert m3.predict(emb_va).shape == (len(val_df),)

    tab = TabularFeatureExtractor(min_freq=5)
    X_tr_tab, cat_cols = tab.fit_transform(train_df)
    X_va_tab, _ = tab.transform(val_df)

    m5 = M5PhoBERTCatBoostModel(pca_components=8, iterations=50, early_stopping_rounds=15, depth=4, learning_rate=0.1, loss_function="RMSE")
    m5.fit(X_tr_tab, emb_tr, train_df["y_true"].to_numpy(), X_va_tab, emb_va, val_df["y_true"].to_numpy(), cat_cols, seed=42)
    preds = m5.predict(X_va_tab, emb_va)
    assert preds.shape == (len(val_df),)
    assert np.isfinite(preds).all()
