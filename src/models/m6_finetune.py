"""M6 optional PhoBERT end-to-end fine-tuning regressor (for Kaggle T4/P100 or RTX 3090)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.features.phobert_embedder import segment_vietnamese_texts


class M6FineTuneRegressor:
    """Optional end-to-end PhoBERT + linear regression head (disabled unless E1-E6 complete)."""

    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        revision: str = "01daacda68afe13d83023d16ec647239e344a1e6",
        max_length: int = 256,
        lr: float = 1.5e-5,
        epochs: int = 2,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.max_length = int(max_length)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.tokenizer = None
        self.encoder = None
        self.head = None
        self.device = None

    def fit(
        self,
        tr_texts: Sequence[str],
        y_tr: np.ndarray,
        va_texts: Sequence[str],
        y_va: np.ndarray,
        checkpoint_dir: Path | None = None,
    ) -> dict[str, Any]:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        self.encoder = AutoModel.from_pretrained(self.model_name, revision=self.revision).to(self.device)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1).to(self.device)

        seg_tr, _ = segment_vietnamese_texts(tr_texts)
        seg_va, _ = segment_vietnamese_texts(va_texts)

        enc_tr = self.tokenizer(seg_tr, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt")
        enc_va = self.tokenizer(seg_va, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt")

        tr_loader = DataLoader(
            TensorDataset(enc_tr["input_ids"], enc_tr["attention_mask"], torch.tensor(y_tr, dtype=torch.float32)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        va_loader = DataLoader(
            TensorDataset(enc_va["input_ids"], enc_va["attention_mask"], torch.tensor(y_va, dtype=torch.float32)),
            batch_size=self.batch_size * 2,
            shuffle=False,
        )

        opt = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.head.parameters()), lr=self.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        loss_fn = nn.L1Loss()

        best_val_mae = float("inf")
        best_state = None
        for epoch in range(self.epochs):
            self.encoder.train()
            self.head.train()
            for ids, mask, target in tr_loader:
                ids, mask, target = ids.to(self.device), mask.to(self.device), target.to(self.device)
                opt.zero_grad()
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                    out = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
                    m = mask.unsqueeze(-1).to(out.dtype)
                    pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-6)
                    pred = self.head(pooled).squeeze(-1)
                    loss = loss_fn(pred, target)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            self.encoder.eval()
            self.head.eval()
            preds_va = []
            with torch.no_grad():
                for ids, mask, _ in va_loader:
                    ids, mask = ids.to(self.device), mask.to(self.device)
                    out = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
                    m = mask.unsqueeze(-1).to(out.dtype)
                    pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-6)
                    preds_va.append(self.head(pooled).squeeze(-1).cpu().numpy())
            val_mae = float(np.mean(np.abs(np.concatenate(preds_va) - y_va)))
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {
                    "encoder": {k: v.cpu() for k, v in self.encoder.state_dict().items()},
                    "head": {k: v.cpu() for k, v in self.head.state_dict().items()},
                }

        if best_state is not None:
            self.encoder.load_state_dict(best_state["encoder"])
            self.head.load_state_dict(best_state["head"])
            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_dir / "m6_best.pt")
        return {"best_val_mae": best_val_mae}

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        if self.encoder is None or self.head is None or self.tokenizer is None or self.device is None:
            raise RuntimeError("M6 model not fitted.")
        seg, _ = segment_vietnamese_texts(texts)
        enc = self.tokenizer(seg, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt")
        self.encoder.eval()
        self.head.eval()
        out_preds = []
        with torch.no_grad():
            for i in range(0, len(seg), self.batch_size * 2):
                ids = enc["input_ids"][i : i + self.batch_size * 2].to(self.device)
                mask = enc["attention_mask"][i : i + self.batch_size * 2].to(self.device)
                hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
                m = mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)
                out_preds.append(self.head(pooled).squeeze(-1).cpu().numpy())
        return np.concatenate(out_preds).astype(np.float64)
