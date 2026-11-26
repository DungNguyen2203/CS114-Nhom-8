from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from src.models.m6_finetune import M6FineTuneRegressor


def test_cpu_finetune_restores_best_epoch_and_flushes_partial_accumulation(monkeypatch):
    class Encoder(torch.nn.Module):
        config = SimpleNamespace(hidden_size=1)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1))

    def tokenize(texts, **kwargs):
        ids = torch.ones(len(texts), 1, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": ids}

    steps = []
    original_step = torch.optim.AdamW.step

    def counted_step(optimizer, *args, **kwargs):
        steps.append(1)
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenize)
    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *args, **kwargs: Encoder())
    monkeypatch.setattr(torch.optim.AdamW, "step", counted_step)
    model = M6FineTuneRegressor(lr=0.1, batch_size=2, gradient_accumulation_steps=2, epochs=3)
    # Opposite targets make the first epoch best and expose aliased CPU state dicts.
    result = model.fit(["train"] * 5, np.full(5, 25.), ["val"], np.array([-25.]))
    assert len(steps) == 6
    assert abs(model.predict(["val"])[0] + 25.) == pytest.approx(result["best_val_mae"])
