"""BEAT piano MIDI embedding adapter.

Uses the official BEAT-code converter, tokenizer, and checkpoint loader. The
embedding is the valid-token mean of final-layer hidden states; long sequences
are encoded as non-overlapping chunks to stay within the backbone context.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
import mido


BEAT_CHECKPOINT_ENV = "BEAT_CHECKPOINT_PATH"
BEAT_CODE_ENV = "BEAT_CODE_PATH"
BEAT_DEVICE_ENV = "BEAT_DEVICE"
MODEL = None
TOKENIZER = None
CONVERTER = None
MODEL_CONFIG = None
SAVE_NPZ = None
DEVICE = None
PRETTY_MIDI_MAX_TICK = 10_000_000


def _beat_code_root() -> Path:
    configured = os.environ.get(BEAT_CODE_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "BEAT-code"


def _load_model():
    global MODEL, TOKENIZER, CONVERTER, MODEL_CONFIG, SAVE_NPZ, DEVICE
    if MODEL is not None:
        return MODEL, TOKENIZER, CONVERTER, MODEL_CONFIG, SAVE_NPZ

    code_root = _beat_code_root()
    required = (
        code_root / "config.py",
        code_root / "piano" / "inference.py",
        code_root / "piano" / "tokenizer.py",
        code_root / "data_prep" / "midi2pianonpz.py",
    )
    if not all(path.is_file() for path in required):
        raise ImportError(
            "BEAT-code source not found. Clone https://github.com/Lekai-Qian/BEAT-code "
            "to BEAT-code/ or set BEAT_CODE_PATH to its checkout."
        )
    root = str(code_root)
    if root not in sys.path:
        sys.path.insert(0, root)

    try:
        config_module = importlib.import_module("config")
        inference = importlib.import_module("piano.inference")
        tokenizer_module = importlib.import_module("piano.tokenizer")
        converter_module = importlib.import_module("data_prep.midi2pianonpz")
    except ImportError as exc:
        raise ImportError(
            "Unable to import BEAT-code. Install its runtime requirements in the "
            "benchmark environment and verify BEAT_CODE_PATH."
        ) from exc

    model_config = config_module.BackboneModelConfig()
    checkpoint = Path(
        os.environ.get(
            BEAT_CHECKPOINT_ENV,
            str(Path(__file__).resolve().parents[2] / "checkpoints" / "piano" / "backbone.pt"),
        )
    ).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"BEAT piano checkpoint not found at {checkpoint}; set {BEAT_CHECKPOINT_ENV} "
            "to checkpoints/piano/backbone.pt"
        )

    device = os.environ.get(
        BEAT_DEVICE_ENV, "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = inference.load_model(str(checkpoint), model_config, device=device)
    model.eval()
    MODEL = model
    TOKENIZER = tokenizer_module.PianoTokenizer()
    CONVERTER = converter_module.MIDIToPianoNPZ()
    SAVE_NPZ = converter_module.save_npz
    MODEL_CONFIG = model_config
    DEVICE = device
    return MODEL, TOKENIZER, CONVERTER, MODEL_CONFIG, SAVE_NPZ


def _rescale_midi_tick_resolution(midi_path, output_path):
    """Keep absolute MIDI timing while reducing high PPQ files below PrettyMIDI's tick cap."""
    midi = mido.MidiFile(str(midi_path))
    absolute_ticks_by_track = []
    largest_tick = 0
    for track in midi.tracks:
        absolute_tick = 0
        absolute_ticks = []
        for message in track:
            absolute_tick += message.time
            absolute_ticks.append(absolute_tick)
        absolute_ticks_by_track.append(absolute_ticks)
        largest_tick = max(largest_tick, absolute_tick)

    largest_tick += 1
    if largest_tick <= PRETTY_MIDI_MAX_TICK:
        return Path(midi_path)
    old_resolution = int(midi.ticks_per_beat)
    if old_resolution <= 1:
        raise ValueError(
            f"Cannot rescale MIDI tick range {largest_tick} with PPQ {old_resolution}"
        )
    new_resolution = old_resolution * (PRETTY_MIDI_MAX_TICK - 2) // largest_tick
    if new_resolution < 1:
        raise ValueError(
            f"Cannot rescale MIDI tick range {largest_tick} with PPQ {old_resolution}"
        )

    scale = new_resolution / old_resolution
    for track, absolute_ticks in zip(midi.tracks, absolute_ticks_by_track):
        previous_tick = 0
        for message, absolute_tick in zip(track, absolute_ticks):
            scaled_tick = round(absolute_tick * scale)
            message.time = scaled_tick - previous_tick
            previous_tick = scaled_tick
    midi.ticks_per_beat = new_resolution
    output_path = Path(output_path)
    midi.save(str(output_path))
    return output_path


def _tokens_for_midi(midi_path, converter, tokenizer, save_npz):
    """Use upstream preprocessing; rescale only inputs that exceed its tick allocation guard."""
    with tempfile.TemporaryDirectory(prefix="beat-midi-") as temp_dir:
        try:
            converted = converter.convert(str(midi_path))
        except ValueError as exc:
            if "largest tick of" not in str(exc):
                raise
            safe_midi_path = _rescale_midi_tick_resolution(
                midi_path, Path(temp_dir) / "scaled.mid"
            )
            converted = converter.convert(str(safe_midi_path))
        if not isinstance(converted, tuple) or len(converted) != 4:
            raise ValueError("BEAT MIDI converter must return segments, metadata, measure_info, pedal")
        segments, metadata, measure_info, pedal = converted
        npz_path = Path(temp_dir) / "piece.npz"
        save_npz(segments, metadata, measure_info, pedal, str(npz_path))
        tokens = tokenizer.encode_file(str(npz_path))
    if not tokens:
        raise ValueError(f"BEAT tokenizer produced no tokens for {midi_path}")
    return [int(token) for token in tokens]


def _weighted_window_pool(window_means, token_counts):
    if not window_means or len(window_means) != len(token_counts):
        raise ValueError("BEAT needs matching nonempty windows and valid-token counts")
    if any(count <= 0 for count in token_counts):
        raise ValueError("BEAT windows must have positive valid-token counts")
    weights = torch.as_tensor(
        token_counts, dtype=window_means[0].dtype, device=window_means[0].device
    )
    return (torch.stack(window_means) * weights[:, None]).sum(dim=0) / weights.sum()


def _encode_tokens(tokens, model, model_config, device=None):
    """Run fixed-size non-overlapping windows and mean-pool content states."""
    if not tokens:
        raise ValueError("BEAT tokenizer produced no tokens")
    max_context = int(model_config.max_position_embeddings)
    if max_context <= 0:
        raise ValueError("BEAT checkpoint reports an invalid context length")
    max_context = min(max_context, 2048)
    if len(tokens) <= 3:
        raise ValueError("BEAT sequence has no content tokens after BOS/TS/TEM header")

    bos_id = int(model_config.bos_token_id)
    eos_id = int(model_config.eos_token_id)
    pad_id = int(model_config.pad_token_id)
    if tokens[0] != bos_id:
        raise ValueError("BEAT piano sequence must begin with its BOS token")
    if device is None:
        device = next(model.parameters()).device

    means = []
    counts = []
    with torch.inference_mode():
        for start in range(0, len(tokens), max_context):
            chunk = tokens[start : start + max_context]
            valid_positions = [
                offset
                for offset, token in enumerate(chunk, start=start)
                if offset >= 3 and token not in (eos_id, pad_id)
            ]
            if not valid_positions:
                continue
            input_ids = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            output = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = getattr(output, "hidden_states", None)
            if not hidden_states:
                raise RuntimeError("BEAT returned no final hidden states")
            hidden = hidden_states[-1]
            if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != len(chunk):
                raise RuntimeError("BEAT returned hidden states with an unexpected shape")
            selected = [position - start for position in valid_positions]
            means.append(hidden[0, selected].mean(dim=0))
            counts.append(len(selected))

    if not means:
        raise ValueError("BEAT sequence has no valid content tokens")
    embedding = _weighted_window_pool(means, counts)
    vector = np.asarray(embedding.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
    expected_dim = int(model_config.hidden_size)
    if vector.size != expected_dim or not np.isfinite(vector).all():
        raise ValueError("BEAT produced an invalid embedding")
    return vector


def _encode_midi_with_components(
    midi_path, model, tokenizer, converter, model_config, save_npz, device=None
):
    tokens = _tokens_for_midi(midi_path, converter, tokenizer, save_npz)
    return _encode_tokens(tokens, model, model_config, device=device)


def encode_midi(midi_path):
    model, tokenizer, converter, model_config, save_npz = _load_model()
    return _encode_midi_with_components(
        midi_path, model, tokenizer, converter, model_config, save_npz, device=DEVICE
    )
