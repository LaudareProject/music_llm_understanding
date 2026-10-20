from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiments.common import embed_clamp3


class _Patchilizer:
    pad_token_id = 0

    def __init__(self, values):
        self.values = values

    def encode(self, mtf, add_special_patches):
        assert add_special_patches is True
        return [[value] * 64 for value in self.values]


class _Encoder:
    def __init__(self):
        self.calls = 0

    def __call__(self, input_ids, attention_mask):
        self.calls += 1
        values = input_ids[:, :, 0].to(torch.float32)
        # Mimic arbitrary hidden states at padded positions; the adapter's
        # mask must keep these values out of the segment mean.
        values = torch.where(values == 0, 999.0, values)
        hidden = values.unsqueeze(-1).expand(-1, -1, 2)
        return {"last_hidden_state": hidden}


def _embed(values):
    model = _Encoder()
    result = embed_clamp3._embed_midi_with_components(
        midi=object(),
        model=model,
        patchilizer=_Patchilizer(values),
        midi_loader=lambda **kwargs: "mtf",
        device="cpu",
    )
    return result, model.calls


def test_clamp3_excludes_padding_and_weights_overlapping_context_windows(monkeypatch):
    monkeypatch.setattr(embed_clamp3, "PATCH_LENGTH", 4)

    short_result, short_calls = _embed([1, 2, 3])
    assert short_calls == 1
    assert torch.equal(short_result, torch.tensor([2.0, 2.0]))

    long_result, long_calls = _embed([1, 2, 3, 4, 5])
    assert long_calls == 2
    assert torch.equal(long_result, torch.tensor([3.0, 3.0]))
