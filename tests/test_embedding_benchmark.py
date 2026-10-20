from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mido
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import embed_beat, feature_backend, runner_data
from experiments.common.comparison_report import generate_comparison_reports
from experiments.common.embed_adversarial_midibert import (
    BEAT_NOTE_FACTOR,
    MAX_BARS,
    POSITION_RESOLUTION,
    _build_vocabularies,
    _encode_notes,
    _pool_window,
)
from experiments.common.embed_midigpt import (
    _normalize_time_signatures,
    _time_signature_for,
    _weighted_window_pool,
)
from experiments.common.runner import _check_backend
from experiments.common.tsv import write_tsv


def test_embedding_cache_isolated_and_rows_remain_aligned(tmp_path, monkeypatch):
    rows = [
        {"query_id": "second", "dataset": "demo", "path": "second.mid"},
        {"query_id": "first", "dataset": "demo", "path": "first.mid"},
    ]
    midi_paths = {"second": tmp_path / "second.mid", "first": tmp_path / "first.mid"}
    values_by_backend = {
        "clamp3_c2": {"second": [2.0, 20.0], "first": [1.0, 10.0]},
        "adversarial_midibert": {"second": [4.0, 40.0], "first": [3.0, 30.0]},
    }
    for backend, values in values_by_backend.items():
        monkeypatch.setenv(feature_backend.FEATURE_BACKEND_ENV, backend)
        monkeypatch.setitem(
            feature_backend._LOADED_ENCODERS,
            backend,
            lambda path, values=values: values[Path(path).stem],
        )
        matrix = feature_backend.matrix_for_rows(rows, midi_paths, tmp_path, tmp_path)
        assert matrix.tolist() == list(values.values())
        assert np.isfinite(matrix).all()
        assert all(
            feature_backend.embedding_cache_path_for_row(row, tmp_path, backend).exists()
            for row in rows
        )
    assert feature_backend.embedding_cache_path_for_row(rows[0], tmp_path, "clamp3_c2") != (
        feature_backend.embedding_cache_path_for_row(rows[0], tmp_path, "adversarial_midibert")
    )


def test_midi_preparation_does_not_skip_aria_cached_rows(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    aria_cache = repo_root / "data" / "demo" / "aria_embeddings" / "song.npy"
    aria_cache.parent.mkdir(parents=True)
    np.save(aria_cache, np.asarray([1.0]))
    tsv_path = tmp_path / "rows.tsv"
    write_tsv(
        tsv_path,
        [{"query_id": "song", "dataset": "demo", "path": "song.mid", "input_kind": "midi"}],
        ["query_id", "dataset", "path", "input_kind"],
    )
    calls = []
    expected_midi = tmp_path / "prepared.mid"
    monkeypatch.setattr(
        runner_data,
        "_prepare_midi_for_row",
        lambda row, root, midi_dir, bin_dir: calls.append(row["query_id"]) or expected_midi,
    )

    rows, midi_paths = runner_data.prepare_rows_and_midi(
        "test", tsv_path, repo_root, tmp_path / "output"
    )

    assert calls == ["song"]
    assert [row["query_id"] for row in rows] == ["song"]
    assert midi_paths == {"song": expected_midi}


def test_midi_bert_octuple_mapping_and_masked_mean():
    vocab = _build_vocabularies()
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.extend(
        [
            mido.Message("program_change", program=0, time=0),
            mido.Message("note_on", note=60, velocity=64, time=0),
            mido.Message("note_off", note=60, velocity=0, time=480),
        ]
    )
    midi.tracks.append(track)
    tokens = _encode_notes(midi, vocab)
    assert len(tokens) == 3
    assert tokens[0][0] == vocab["Bar"]["Bar <SOS>"]
    assert tokens[1][2] == vocab["Instrument"]["Instrument 0"]
    assert tokens[1][3] == vocab["Pitch"]["Pitch 60"]
    assert tokens[-1][0] == vocab["Bar"]["Bar <EOS>"]

    hidden = torch.tensor([[[1.0, 3.0], [100.0, 100.0], [5.0, 7.0]]])
    pooled, count = _pool_window(hidden, torch.tensor([True, False, True]))
    assert count == 2
    assert pooled.tolist() == [3.0, 5.0]


def test_midi_bert_closes_short_bar_on_mid_bar_time_signature_change():
    vocab = _build_vocabularies()
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.extend(
        [
            mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0),
            mido.Message("program_change", program=0, time=0),
            mido.Message("note_on", note=60, velocity=64, time=0),
            mido.Message("note_off", note=60, velocity=0, time=480),
            mido.MetaMessage("time_signature", numerator=3, denominator=4, time=0),
            mido.Message("note_on", note=62, velocity=64, time=0),
            mido.Message("note_off", note=62, velocity=0, time=1440),
        ]
    )
    midi.tracks.append(track)

    tokens = _encode_notes(midi, vocab)

    assert len(tokens) == 4
    assert tokens[0][0] == vocab["Bar"]["Bar <SOS>"]
    first, second = tokens[1], tokens[2]
    assert first[0] == vocab["Bar"]["Bar 0"]
    assert first[1] == vocab["Position"]["Position 0/64"]
    assert first[3] == vocab["Pitch"]["Pitch 60"]
    assert first[6] == vocab["TimeSig"]["TimeSig 4/4"]
    assert second[0] == vocab["Bar"]["Bar 1"]
    assert second[1] == vocab["Position"]["Position 0/64"]
    assert second[3] == vocab["Pitch"]["Pitch 62"]
    assert second[6] == vocab["TimeSig"]["TimeSig 3/4"]
    assert tokens[-1][0] == vocab["Bar"]["Bar <EOS>"]


def test_midi_bert_drops_notes_beyond_vocabulary_bars():
    vocab = _build_vocabularies()
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    bar_256_position = (MAX_BARS + 1) * BEAT_NOTE_FACTOR * POSITION_RESOLUTION
    bar_256_tick = bar_256_position * midi.ticks_per_beat // POSITION_RESOLUTION
    track.extend(
        [
            mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0),
            mido.Message("note_on", note=60, velocity=64, time=0),
            mido.Message("note_off", note=60, velocity=0, time=480),
            mido.Message("note_on", note=62, velocity=64, time=bar_256_tick - 480),
            mido.Message("note_off", note=62, velocity=0, time=480),
        ]
    )
    midi.tracks.append(track)

    tokens = _encode_notes(midi, vocab)

    assert len(tokens) == 3
    assert tokens[1][0] == vocab["Bar"]["Bar 0"]
    assert tokens[1][3] == vocab["Pitch"]["Pitch 60"]
    assert tokens[-1][0] == vocab["Bar"]["Bar <EOS>"]


def test_midigpt_pool_weights_windows_by_valid_token_count():
    result = _weighted_window_pool(
        [torch.tensor([2.0, 4.0]), torch.tensor([8.0, 10.0])], [3, 1]
    )
    assert torch.allclose(result, torch.tensor([3.5, 5.5]))
    with pytest.raises(ValueError, match="valid-token counts"):
        _weighted_window_pool([torch.ones(2)], [0])



class _FakeBeatDecoder:
    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, attention_mask, output_hidden_states, use_cache):
        self.calls.append(input_ids[0].tolist())
        assert output_hidden_states is True
        assert use_cache is False
        values = input_ids.to(torch.float32)
        hidden = torch.stack((values, values * 2), dim=-1)
        return SimpleNamespace(hidden_states=(hidden,))


class _FakeBeatModel:
    def __init__(self):
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.model = _FakeBeatDecoder()

    def parameters(self):
        return iter((self.anchor,))


def test_beat_excludes_header_eos_and_padding_across_context_windows():
    config = SimpleNamespace(
        max_position_embeddings=5,
        bos_token_id=1,
        eos_token_id=10,
        pad_token_id=0,
        hidden_size=2,
    )
    model = _FakeBeatModel()

    vector = embed_beat._encode_tokens(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0], model, config
    )

    assert vector.tolist() == [6.5, 13.0]
    assert model.model.calls == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    assert max(map(len, model.model.calls)) <= config.max_position_embeddings


def test_beat_uses_official_conversion_save_and_tokenizer_flow():
    config = SimpleNamespace(
        max_position_embeddings=5,
        bos_token_id=1,
        eos_token_id=10,
        pad_token_id=0,
        hidden_size=2,
    )
    model = _FakeBeatModel()
    calls = []

    class Converter:
        def convert(self, midi_path):
            calls.append(("convert", midi_path))
            return ["segment"], {"tempo": 120}, ["measure"], ["pedal"]

    class Tokenizer:
        def encode_file(self, npz_path):
            assert Path(npz_path).is_file()
            calls.append(("tokenize", Path(npz_path).suffix))
            return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def save_npz(segments, metadata, measure_info, pedal, output_path):
        assert (segments, metadata, measure_info, pedal) == (
            ["segment"], {"tempo": 120}, ["measure"], ["pedal"]
        )
        Path(output_path).write_bytes(b"fake npz")
        calls.append(("save", Path(output_path).suffix))

    vector = embed_beat._encode_midi_with_components(
        "song.mid", model, Tokenizer(), Converter(), config, save_npz
    )

    assert vector.tolist() == [6.5, 13.0]
    assert calls[0] == ("convert", "song.mid")
    assert calls[1:] == [("save", ".npz"), ("tokenize", ".npz")]


def test_beat_rescales_high_resolution_midi_without_mutating_source(tmp_path):
    source_path = tmp_path / "long-score.mid"
    midi = mido.MidiFile(ticks_per_beat=10_080)
    track = mido.MidiTrack()
    track.extend(
        [
            mido.Message("note_on", note=60, velocity=90, time=0),
            mido.Message("note_off", note=60, velocity=0, time=12_482_820),
        ]
    )
    midi.tracks.append(track)
    midi.save(source_path)

    scaled_path = embed_beat._rescale_midi_tick_resolution(
        source_path, tmp_path / "scaled.mid"
    )
    source = mido.MidiFile(str(source_path))
    scaled = mido.MidiFile(str(scaled_path))
    original_end_tick = sum(message.time for message in source.tracks[0])
    scaled_end_tick = sum(message.time for message in scaled.tracks[0])

    assert source.ticks_per_beat == 10_080
    assert scaled.ticks_per_beat < source.ticks_per_beat
    assert scaled_end_tick + 1 <= embed_beat.PRETTY_MIDI_MAX_TICK
    assert abs(
        original_end_tick / source.ticks_per_beat
        - scaled_end_tick / scaled.ticks_per_beat
    ) <= 0.5 / scaled.ticks_per_beat


def test_beat_rejects_empty_content_and_invalid_pool_weights():
    config = SimpleNamespace(
        max_position_embeddings=8,
        bos_token_id=1,
        eos_token_id=10,
        pad_token_id=0,
        hidden_size=2,
    )
    with pytest.raises(ValueError, match="no valid content tokens"):
        embed_beat._encode_tokens([1, 2, 3, 10], _FakeBeatModel(), config)
    with pytest.raises(ValueError, match="positive valid-token counts"):
        embed_beat._weighted_window_pool([torch.ones(2)], [0])
    with pytest.raises(ValueError, match="matching nonempty windows"):
        embed_beat._weighted_window_pool([torch.ones(2)], [])


def test_beat_backend_is_registered():
    assert feature_backend.BACKEND_ENCODERS["beat"] == (
        "experiments.common.embed_beat",
        "encode_midi",
    )

def test_midigpt_normalizes_unsupported_time_signatures():
    supported = {(1, 16), (1, 8), (2, 4), (3, 8), (3, 4), (4, 4), (5, 4), (9, 8)}

    assert _time_signature_for(12, 16, supported) == (3, 4)
    assert _time_signature_for(11, 16, supported) == (3, 4)
    assert _time_signature_for(19, 16, supported) == (5, 4)
    assert _time_signature_for(1, 32, supported) == (1, 16)

    supported_bar = SimpleNamespace(ts_numerator=4, ts_denominator=4, beat_length=4.0)
    changed_bar = SimpleNamespace(ts_numerator=12, ts_denominator=16, beat_length=3.0)
    score = SimpleNamespace(
        tracks=[SimpleNamespace(bars=[supported_bar, changed_bar])]
    )

    assert _normalize_time_signatures(score, supported) == 1
    assert (supported_bar.ts_numerator, supported_bar.ts_denominator) == (4, 4)
    assert supported_bar.beat_length == 4.0
    assert (changed_bar.ts_numerator, changed_bar.ts_denominator) == (3, 4)
    assert changed_bar.beat_length == 3.0


def test_backend_override_has_priority_over_environment_and_settings(monkeypatch):
    settings = type("Settings", (), {"DEFAULT_FEATURE_BACKEND": "aria"})
    monkeypatch.setenv(feature_backend.FEATURE_BACKEND_ENV, "musif")

    _check_backend(settings, "demo", {}, "clamp3_c2")
    assert feature_backend.feature_backend_name() == "clamp3_c2"
    monkeypatch.setenv(feature_backend.FEATURE_BACKEND_ENV, "musif")
    _check_backend(settings, "demo", {})
    assert feature_backend.feature_backend_name() == "musif"


def _write_fake_run(repo_root, experiment, backend, metrics):
    output_root = repo_root / "outputs" / feature_backend.BACKEND_OUTPUT_DIRS[backend] / experiment
    run_dir = output_root / f"run_{backend}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"experiment": experiment, "feature_backend": backend}),
        encoding="utf-8",
    )
    write_tsv(run_dir / "metrics.tsv", metrics, list(metrics[0]))


def test_comparison_report_generation_uses_complete_exact_runs(tmp_path):
    backends = ("aria", "clamp3_c2")
    for backend, value in zip(backends, (0.4, 0.6)):
        _write_fake_run(
            tmp_path,
            "arousal_axis",
            backend,
            [{"metric": "spearman", "mean": str(value), "ci95": "0.1", "folds": "5", "rows_kept": "20"}],
        )
        _write_fake_run(
            tmp_path,
            "sentiment_prediction",
            backend,
            [
                {"metric": "class_accuracy", "value": str(value), "support": "20"},
                {"metric": "balanced_class_accuracy", "value": str(value - 0.1), "support": "20"},
            ],
        )

    report_dir = generate_comparison_reports(
        tmp_path,
        experiments=("arousal_axis", "sentiment_prediction"),
        backends=backends,
    )

    assert (report_dir / "arousal_axis.tsv").is_file()
    assert (report_dir / "arousal_axis.png").is_file()
    assert (report_dir / "sentiment_prediction.tsv").is_file()
    assert (report_dir / "sentiment_prediction.png").is_file()
    report_rows = (report_dir / "arousal_axis.tsv").read_text(encoding="utf-8")
    assert "run_aria" in report_rows and "run_clamp3_c2" in report_rows


def test_comparison_reports_refuse_incomplete_matrix_without_outputs(tmp_path):
    _write_fake_run(
        tmp_path,
        "arousal_axis",
        "aria",
        [{"metric": "spearman", "mean": "0.4", "ci95": "0.1", "folds": "5", "rows_kept": "20"}],
    )

    with pytest.raises(FileNotFoundError, match="complete comparison report"):
        generate_comparison_reports(
            tmp_path,
            experiments=("arousal_axis",),
            backends=("aria", "clamp3_c2"),
        )

    assert not (tmp_path / "outputs" / "model_comparison").exists()
