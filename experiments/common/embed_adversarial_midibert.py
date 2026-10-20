"""Adversarial-MidiBERT embedding adapter.

Octuple conversion follows the official muzic/MusicBERT ``preprocess.py`` and
RS2002/PianoBart ``Data/data_generation/convert.py``: positions are quantized
at 16 steps per beat, bar boundaries are derived from the active time
signature, and only the official 256-bar vocabulary is accepted.

Deviation: the official converter raises when a time signature changes while
a bar is still open (``cnt == measure_length`` assertion). Such inputs also
occur for scores with an anacrusis, where MIDI carries no bar-line marker.
For those inputs only, this adapter closes the current short bar at the change
position and starts the new bar at position 0 with the new signature. All
other inputs produce exactly the official encoding. Notes beyond the official
256-bar vocabulary (``Bar 0``-``Bar 255``) are dropped, matching the
converter's vocabulary limit.
"""

from __future__ import annotations

import math
import os

import mido
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertSelfAttention


REPO_ID = "RS2002/Adversarial-MidiBERT"
MAX_SEQUENCE_LENGTH = 1024
POSITION_RESOLUTION = 16
BEAT_NOTE_FACTOR = 4
VELOCITY_QUANTIZATION = 4
TEMPO_QUANTIZATION = 12
MIN_TEMPO_BPM = 16.0
MAX_TEMPO_BPM = 256.0
DURATION_MAX_BEATS = 8
MAX_TIMESIG_DENOMINATOR_POWER = 6
MAX_NOTES_PER_BAR = 2
MAX_INSTRUMENT = 127
MAX_PITCH = 127
MAX_BARS = 255
MODEL = None
VOCABULARIES = None
DEVICE = None


def _build_vocabularies():
    vocab = {name: {} for name in (
        "Bar", "Position", "Instrument", "Pitch", "Duration", "Velocity", "TimeSig", "Tempo"
    )}
    for name, values in (
        ("Bar", (f"Bar {i}" for i in range(MAX_BARS + 1))),
        ("Position", (f"Position {i}/64" for i in range(MAX_NOTES_PER_BAR * BEAT_NOTE_FACTOR * POSITION_RESOLUTION))),
        ("Instrument", (f"Instrument {i}" for i in range(MAX_INSTRUMENT + 1))),
        ("Pitch", _pitch_vocab()),
        ("Duration", (f"Duration {i}" for i in range(DURATION_MAX_BEATS * POSITION_RESOLUTION))),
        ("Velocity", (f"Velocity {4 * i + 2}" for i in range(32))),
        ("TimeSig", _timesig_vocab()),
        ("Tempo", _tempo_vocab()),
    ):
        vocab[name].update((value, index) for index, value in enumerate(values))
        special_start = len(vocab[name])
        for offset, special in enumerate(("PAD", "MASK", "SOS", "EOS", "CLS", "SEP")):
            vocab[name][f"{name} <{special}>"] = special_start + offset
    vocab["Instrument"]["Instrument percussion"] = 128
    # The official vocabulary assigns the percussion ID after programs 0-127.
    vocab["Instrument"]["Instrument <PAD>"] = 129
    vocab["Instrument"]["Instrument <MASK>"] = 130
    vocab["Instrument"]["Instrument <SOS>"] = 131
    vocab["Instrument"]["Instrument <EOS>"] = 132
    vocab["Instrument"]["Instrument <CLS>"] = 133
    vocab["Instrument"]["Instrument <SEP>"] = 134
    return vocab
def _pitch_vocab():
    yield from (f"Pitch {i}" for i in range(MAX_PITCH + 1))
    yield from (f"Pitch percussion {i}" for i in range(MAX_PITCH + 1))


def _timesig_vocab():
    return (
        f"TimeSig {numerator}/{2 ** denominator_power}"
        for denominator_power in range(MAX_TIMESIG_DENOMINATOR_POWER + 1)
        for numerator in range(1, (2 ** denominator_power) * MAX_NOTES_PER_BAR + 1)
    )


def _tempo_vocab():
    return (
        f"Tempo {2 ** (index / TEMPO_QUANTIZATION) * MIN_TEMPO_BPM}"
        for index in range(49)
    )


def _time_signature_reduce(numerator, denominator):
    while (
        denominator > 2 ** MAX_TIMESIG_DENOMINATOR_POWER
        and denominator % 2 == 0
        and numerator % 2 == 0
    ):
        numerator //= 2
        denominator //= 2
    while numerator > MAX_NOTES_PER_BAR * denominator:
        for factor in range(2, numerator + 1):
            if numerator % factor == 0:
                numerator //= factor
                break
    return numerator, denominator


def _tempo_index(bpm):
    bpm = min(max(float(bpm), MIN_TEMPO_BPM), MAX_TEMPO_BPM)
    return round(math.log2(bpm / MIN_TEMPO_BPM) * TEMPO_QUANTIZATION)


def _time_position(ticks, ticks_per_beat):
    return round(ticks * POSITION_RESOLUTION / ticks_per_beat)


def _midi_events(midi):
    notes = []
    time_signatures = [(0, 4, 4)]
    tempos = [(0, 120.0)]
    for track in midi.tracks:
        absolute_tick = 0
        active = {}
        programs = {}
        for message in track:
            absolute_tick += message.time
            if message.type == "time_signature":
                time_signatures.append(
                    (absolute_tick, message.numerator, message.denominator)
                )
            elif message.type == "set_tempo":
                tempos.append((absolute_tick, 60_000_000 / message.tempo))
            elif message.type == "program_change":
                programs[message.channel] = message.program
            elif not message.is_meta and message.type == "note_on" and message.velocity:
                key = (message.channel, message.note)
                active.setdefault(key, []).append(
                    (
                        absolute_tick,
                        message.velocity,
                        message.channel,
                        programs.get(message.channel, 0),
                    )
                )
            elif not message.is_meta and message.type in ("note_off", "note_on"):
                key = (message.channel, message.note)
                starts = active.get(key)
                if starts:
                    start, velocity, channel, program = starts.pop(0)
                    if absolute_tick > start:
                        notes.append(
                            (
                                start,
                                absolute_tick,
                                message.note,
                                velocity,
                                channel == 9,
                                program,
                            )
                        )
    return notes, sorted(time_signatures), sorted(tempos)


def _metadata_at(events, tick, default):
    value = default
    for event_tick, *event_value in events:
        if event_tick > tick:
            break
        value = event_value
    return value


def _encode_notes(midi, vocab):
    notes, time_signatures, tempos = _midi_events(midi)
    if not notes:
        raise ValueError("MIDI file contains no paired note-on/note-off events")
    positions = [
        (_time_position(start, midi.ticks_per_beat), _time_position(end, midi.ticks_per_beat),
         pitch, velocity, is_drum, program, start)
        for start, end, pitch, velocity, is_drum, program in notes
    ]
    max_position = min(max(note[0] for note in positions) + 1, 2 ** 16)
    signature_at_position = []
    tempo_at_position = []
    for position in range(max_position):
        tick = round(position * midi.ticks_per_beat / POSITION_RESOLUTION)
        numerator, denominator = _metadata_at(time_signatures, tick, (4, 4))
        numerator, denominator = _time_signature_reduce(numerator, denominator)
        signature_at_position.append((numerator, denominator))
        tempo_at_position.append(_metadata_at(tempos, tick, (120.0,))[0])

    bars = [0] * max_position
    within_bar = [0] * max_position
    current_bar = 0
    offset = 0
    current_signature = signature_at_position[0]
    measure_length = current_signature[0] * BEAT_NOTE_FACTOR * POSITION_RESOLUTION // current_signature[1]
    for position in range(max_position):
        signature = signature_at_position[position]
        if signature != current_signature:
            # The official converter rejects mid-bar changes. Close the short
            # bar at the change and start the new signature at position 0.
            if offset != 0:
                current_bar += 1
                offset = 0
            current_signature = signature
            measure_length = current_signature[0] * BEAT_NOTE_FACTOR * POSITION_RESOLUTION // current_signature[1]
        bars[position] = current_bar
        within_bar[position] = offset
        offset += 1
        if offset == measure_length:
            current_bar += 1
            offset = 0

    encoded = []
    for position, end_position, pitch, velocity, is_drum, program, start_tick in positions:
        if position >= max_position or bars[position] > MAX_BARS:
            # The official vocabulary has no Bar token above MAX_BARS.
            continue
        numerator, denominator = signature_at_position[position]
        tempo_index = _tempo_index(tempo_at_position[position])
        duration = max(0, end_position - position)
        duration_index = min(duration, DURATION_MAX_BEATS * POSITION_RESOLUTION - 1)
        values = (
            f"Bar {bars[position]}",
            f"Position {within_bar[position]}/64",
            "Instrument percussion" if is_drum else f"Instrument {program}",
            f"Pitch percussion {pitch}" if is_drum else f"Pitch {pitch}",
            f"Duration {duration_index}",
            f"Velocity {min(velocity, 127) // VELOCITY_QUANTIZATION * VELOCITY_QUANTIZATION + 2}",
            f"TimeSig {numerator}/{denominator}",
            f"Tempo {2 ** (tempo_index / TEMPO_QUANTIZATION) * MIN_TEMPO_BPM}",
        )
        try:
            encoded.append([vocab[name][value] for name, value in zip(vocab, values)])
        except KeyError as exc:
            raise ValueError(f"MIDI event is outside the official MidiBERT vocabulary: {exc}") from exc
    encoded.sort()
    if not encoded:
        raise ValueError("MIDI file produced no valid Octuple events")
    sos = [vocab[name][f"{name} <SOS>"] for name in vocab]
    eos = [vocab[name][f"{name} <EOS>"] for name in vocab]
    return [sos, *encoded, eos]


def _load_model():
    global MODEL, VOCABULARIES, DEVICE
    if MODEL is None:
        requested_device = os.environ.get(
            "ADVERSARIAL_MIDIBERT_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        device = torch.device(requested_device)
        vocab = _build_vocabularies()
        config = BertConfig(
            max_position_embeddings=MAX_SEQUENCE_LENGTH,
            position_embedding_type="relative_key_query",
            hidden_size=768,
        )
        model = _MidiBert(config, vocab).to(device)
        weights_path = hf_hub_download(
            repo_id=REPO_ID, filename="model.safetensors"
        )
        state_dict = load_file(weights_path, device=str(device))
        if state_dict and all(key.startswith("model.") for key in state_dict):
            state_dict = {
                key.removeprefix("model."): value for key, value in state_dict.items()
            }
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        MODEL, VOCABULARIES, DEVICE = model, vocab, device
    return MODEL, VOCABULARIES, DEVICE


class _AttributeEmbedding(torch.nn.Module):
    def __init__(self, count, dimension=256):
        super().__init__()
        self.lut = torch.nn.Embedding(count, dimension)
        self.dimension = dimension

    def forward(self, ids):
        return self.lut(ids) * math.sqrt(self.dimension)


class _RelativeBertSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.distance_embedding = torch.nn.Embedding(
            2 * config.max_position_embeddings - 1, self.attention_head_size
        )

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        head_shape = (batch_size, sequence_length, self.num_attention_heads, self.attention_head_size)
        query = self.query(hidden_states).view(head_shape).transpose(1, 2)
        key = self.key(hidden_states).view(head_shape).transpose(1, 2)
        value = self.value(hidden_states).view(head_shape).transpose(1, 2)
        positions = torch.arange(sequence_length, device=hidden_states.device)
        relative_positions = positions[:, None] - positions[None, :]
        relative_positions = relative_positions + self.config.max_position_embeddings - 1
        positional_embeddings = self.distance_embedding(relative_positions).to(query.dtype)
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores + torch.einsum("bhld,lrd->bhlr", query, positional_embeddings)
        scores = scores + torch.einsum("bhrd,lrd->bhlr", key, positional_embeddings)
        scores = scores * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = self.dropout(probabilities)
        context = torch.matmul(probabilities, value)
        context = context.transpose(1, 2).reshape(batch_size, sequence_length, -1).contiguous()
        return context, probabilities


class _MidiBert(torch.nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.classes = tuple(vocab)
        self.bert = BertModel(config)
        for layer in self.bert.encoder.layer:
            layer.attention.self = _RelativeBertSelfAttention(config)
        self.word_emb = torch.nn.ModuleList(
            _AttributeEmbedding(len(vocab[name])) for name in self.classes
        )
        self.in_linear = torch.nn.Linear(8 * 256, config.hidden_size)
        self.attention_linear = torch.nn.Sequential(
            torch.nn.Linear(8 * 256, 4 * 256),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * 256, 4 * 256),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * 256, 8 * 256),
            torch.nn.Sigmoid(),
        )
    def forward(self, input_ids, attn_mask):
        embeddings = torch.cat(
            [layer(input_ids[..., index]) for index, layer in enumerate(self.word_emb)],
            dim=-1,
        )
        return self.bert(
            inputs_embeds=self.in_linear(embeddings),
            attention_mask=attn_mask,
            output_hidden_states=False,
            return_dict=True,
        )


def _pool_window(hidden, valid_mask):
    weights = valid_mask.to(hidden.dtype).view(1, -1, 1)
    count = weights.sum(dim=1)
    if count.item() == 0:
        return None, 0
    return (hidden * weights).sum(dim=1).squeeze(0) / count.squeeze(0), int(count.item())


def encode_midi(midi_path):
    model, vocab, device = _load_model()
    midi = mido.MidiFile(str(midi_path))
    encoded = _encode_notes(midi, vocab)
    pad_id = vocab["Bar"]["Bar <PAD>"]
    pooled_windows = []
    window_sizes = []
    for offset in range(0, len(encoded), MAX_SEQUENCE_LENGTH):
        window = encoded[offset : offset + MAX_SEQUENCE_LENGTH]
        input_ids = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
        valid_mask = input_ids[0, :, 0] != pad_id
        attention_mask = valid_mask.to(torch.long).unsqueeze(0)
        with torch.inference_mode():
            output = model(input_ids, attention_mask)
            pooled, valid_count = _pool_window(output.last_hidden_state, valid_mask)
        if pooled is not None:
            pooled_windows.append(pooled)
            window_sizes.append(valid_count)
    if not pooled_windows:
        raise ValueError(f"MidiBERT produced no valid token embeddings for {midi_path}")
    weights = torch.tensor(window_sizes, dtype=pooled_windows[0].dtype, device=device)
    vector = (torch.stack(pooled_windows) * weights[:, None]).sum(dim=0) / weights.sum()
    result = np.asarray(vector.cpu().numpy(), dtype=np.float32).reshape(-1)
    if result.size != 768 or not np.isfinite(result).all():
        raise ValueError(f"Adversarial-MidiBERT produced an invalid embedding for {midi_path}")
    return result
