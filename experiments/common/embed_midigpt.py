"""MIDI-GPT embedding adapter.

Uses the official MIDI-GPT tokenizer and the ``yellow_medium`` checkpoint,
which exposes a closed time-signature vocabulary (for example ``4/4``,
``3/4``, ``9/16``). The official encoder raises ``Unknown time signature`` for
any bar outside that vocabulary.

Deviation: an unsupported bar signature is first reduced by its greatest
common divisor (``12/16`` -> ``3/4``); if the reduced signature is still not in
the checkpoint vocabulary, the nearest supported bar length in quarter notes
is used (ties prefer the smaller denominator, then the smaller numerator).
``beat_length`` follows the mapped signature. Only inputs the official
encoder rejects are affected.
"""

from __future__ import annotations

import json
import os
import copy
import math

import numpy as np
import torch


def _supported_time_signatures(config_json):
    supported = set()
    for value in config_json.get("time_signatures") or []:
        numerator, denominator = str(value).split("/")
        supported.add((int(numerator), int(denominator)))
    if not supported:
        raise ValueError("MIDI-GPT checkpoint has no supported time signatures")
    return supported


def _time_signature_for(numerator, denominator, supported):
    common = math.gcd(numerator, denominator)
    reduced = (numerator // common, denominator // common)
    if reduced in supported:
        return reduced
    length = numerator * 4 / denominator
    return min(
        supported,
        key=lambda signature: (
            abs(signature[0] * 4 / signature[1] - length),
            signature[1],
            signature[0],
        ),
    )


def _normalize_time_signatures(score, supported):
    changed = 0
    for track in score.tracks:
        for bar in track.bars:
            signature = (bar.ts_numerator, bar.ts_denominator)
            if signature in supported:
                continue
            numerator, denominator = _time_signature_for(*signature, supported)
            bar.ts_numerator = numerator
            bar.ts_denominator = denominator
            bar.beat_length = numerator * 4 / denominator
            changed += 1
    return changed


def _score_windows(score, window_bars):
    from midigpt._types import Bar, Score, Track

    total_bars = max((len(track.bars) for track in score.tracks), default=0)
    windows = []
    for offset in range(0, total_bars, window_bars):
        end = offset + window_bars
        tracks = []
        for track in score.tracks:
            bars = copy.deepcopy(track.bars[offset:end])
            if not bars:
                continue
            if len(bars) < window_bars:
                template = bars[-1]
                bars.extend(
                    Bar(
                        ts_numerator=template.ts_numerator,
                        ts_denominator=template.ts_denominator,
                        beat_length=template.beat_length,
                    )
                    for _ in range(window_bars - len(bars))
                )
            tracks.append(
                Track(
                    bars=bars,
                    instrument=track.instrument,
                    track_type=track.track_type,
                    attributes=dict(track.attributes),
                )
            )
        if tracks:
            windows.append(Score(tracks=tracks, resolution=score.resolution, tempo=score.tempo))
    return windows


def _tokens_per_window(tokenizer, score, window_bars):
    from midigpt import _core

    opts = _core.EncodeOptions()
    opts.window_bars = window_bars
    return [tokenizer.encode(window, opts=opts) for window in _score_windows(score, window_bars)]


def encode_midi(midi_path):
    from midigpt import Score

    model, tokenizer = _load_model()
    score = Score.from_midi(str(midi_path))
    config = json.loads(tokenizer._vocab.config().to_json())

    bar_choices = sorted(set(config.get("num_bars_map") or []))
    if not bar_choices:
        raise ValueError("MIDI-GPT checkpoint has no supported bar-window sizes")
    _normalize_time_signatures(score, _supported_time_signatures(config))
    window_bars = max(bar_choices)
    token_windows = _tokens_per_window(tokenizer, score, window_bars)
    if not token_windows or any(not tokens for tokens in token_windows):
        raise ValueError(f"MIDI-GPT tokenizer produced no valid tokens for {midi_path}")

    model_dim = model.cfg.n_embd
    max_context = model.max_context()
    if max_context < 1:
        raise ValueError("MIDI-GPT checkpoint reports an invalid context length")
    parameter = next(model.parameters())
    device = parameter.device
    pooled_windows = []
    window_sizes = []

    for tokens in token_windows:
        for offset in range(0, len(tokens), max_context):
            window = tokens[offset : offset + max_context]
            input_ids = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
            layer_hidden = []
            with torch.inference_mode():
                model.forward_with_hooks(
                    input_ids,
                    past_kv=None,
                    hooks={"hidden": lambda layer, hidden: layer_hidden.append(hidden)},
                )
                if not layer_hidden:
                    raise RuntimeError("MIDI-GPT returned no hidden states")
                hidden = model.transformer.ln_f(layer_hidden[-1])
                valid_hidden = hidden[0]
                pooled_windows.append(valid_hidden.mean(dim=0))
                window_sizes.append(valid_hidden.shape[0])

    embedding = _weighted_window_pool(pooled_windows, window_sizes)
    vector = np.asarray(embedding.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
    if vector.size != model_dim or not np.isfinite(vector).all():
        raise ValueError(f"MIDI-GPT produced an invalid embedding for {midi_path}")
    return vector



MODEL_NAME = "yellow_medium"
MODEL = None
TOKENIZER = None


def _load_model():
    global MODEL, TOKENIZER
    if MODEL is None:
        try:
            from midigpt.inference import InferenceEngine
        except ImportError as exc:
            raise ImportError(
                "Install the benchmark dependencies including midigpt[inference] "
                "to use the MIDI-GPT backend"
            ) from exc
        device = os.environ.get("MIDIGPT_DEVICE", "auto")
        engine = InferenceEngine.from_pretrained(MODEL_NAME, device=device)
        MODEL = engine._model
        TOKENIZER = engine._tokenizer
    return MODEL, TOKENIZER


def _weighted_window_pool(window_means, valid_token_counts):
    if not window_means or len(window_means) != len(valid_token_counts):
        raise ValueError("MIDI-GPT needs matching nonempty windows and valid-token counts")
    if any(count <= 0 for count in valid_token_counts):
        raise ValueError("MIDI-GPT windows must have positive valid-token counts")
    weights = torch.as_tensor(
        valid_token_counts, dtype=window_means[0].dtype, device=window_means[0].device
    )
    return (torch.stack(window_means) * weights[:, None]).sum(dim=0) / weights.sum()


