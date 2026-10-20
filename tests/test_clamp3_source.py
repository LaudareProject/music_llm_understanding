from __future__ import annotations

import builtins
import importlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mido
import torch

from experiments.common import clamp3_source


def test_official_midi_serialization_and_special_patch_encoding():
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="omitted", time=0))
    track.append(mido.Message("note_on", channel=0, note=60, velocity=90, time=0))
    track.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=120))
    midi.tracks.append(track)

    mtf = clamp3_source.load_midi_mtf(midi)
    assert mtf == (
        "ticks_per_beat 480\n"
        "note_on 0 0 60 90\n"
        "note_off 120 0 60 0\n"
        "end_of_track 0"
    )
    encoded = clamp3_source.M3Patchilizer().encode(mtf, add_special_patches=True)
    assert len(encoded) == 6
    assert encoded[0] == [1] * 64
    assert encoded[-1] == [1] + [2] * 63
    assert encoded[1][:6] == [1, *map(ord, "ticks")]
    assert encoded[1][-1] == 0
    assert encoded[2][:5] == [1, *map(ord, "note")]
    assert encoded[2][63] == 0


def test_project_adapter_does_not_import_aria(monkeypatch):
    original_import = builtins.__import__

    def reject_aria(name, *args, **kwargs):
        if name == "aria" or name.startswith("aria."):
            raise AssertionError(f"unexpected aria import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_aria)
    adapter = importlib.import_module("experiments.common.embed_clamp3")
    monkeypatch.setattr(adapter, "hf_hub_download", lambda **kwargs: "checkpoint")
    model = object()
    patchilizer = object()
    monkeypatch.setattr(adapter, "load_symbolic_encoder", lambda path, device: model)
    monkeypatch.setattr(adapter, "M3Patchilizer", lambda: patchilizer)
    monkeypatch.setattr(adapter, "MODEL", None)
    monkeypatch.setattr(adapter, "PATCHILIZER", None)
    monkeypatch.setattr(adapter, "DEVICE", None)

    loaded_model, loaded_patchilizer = adapter._load_model()
    assert loaded_model is model
    assert loaded_patchilizer is patchilizer



def test_long_symbolic_input_uses_weighted_overlapping_segment_pool(monkeypatch):
    class SegmentModel:
        def __init__(self):
            self.patch_embedding = torch.nn.Linear(1, 1)
            self.calls = 0

        def __call__(self, patches, mask):
            self.calls += 1
            value = 2.0 if self.calls == 1 else 4.0
            return {
                "last_hidden_state": torch.full(
                    (1, clamp3_source.PATCH_LENGTH, 768), value
                )
            }

    class Patchilizer:
        pad_token_id = 0

        def encode(self, mtf, add_special_patches):
            assert add_special_patches is True
            return [[1] * 64 for _ in range(513)]

    monkeypatch.setattr(clamp3_source, "load_midi_mtf", lambda *args, **kwargs: "mtf")
    model = SegmentModel()
    result = clamp3_source.get_midi_embedding(mido.MidiFile(), model, Patchilizer())

    assert model.calls == 2
    assert result.shape == (768,)
    assert torch.equal(result, torch.full((768,), 3.0))

def test_symbolic_checkpoint_key_mapping_strictly_loads_encoder():
    config = clamp3_source.BertConfig(
        vocab_size=1,
        hidden_size=768,
        num_hidden_layers=1,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
    )
    model = clamp3_source.M3PatchEncoder(config)
    state = {
        f"symbolic_model.{key}": value.clone()
        for key, value in model.state_dict().items()
    }
    state["audio_model.unused_weight"] = torch.ones(1)
    mapped = clamp3_source.symbolic_checkpoint_state(state)
    incompatible = model.load_state_dict(mapped, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert "audio_model.unused_weight" not in mapped
