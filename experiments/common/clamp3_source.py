"""Standalone CLaMP3 C2 symbolic inference, separated from the aria submodule.

Adapted from EleutherAI/Aria's `aria/eval/m3/emb.py`, `utils.py`, and
`config.py`; reduced to the symbolic inference path and updated for isolated
checkpoint loading. The adapted source is Apache-2.0 licensed; see
`LICENSE-CLaMP3.txt`.
"""
from __future__ import annotations

import os
import random
import re

import mido
import torch
from torch import nn
from transformers import BertConfig, BertModel
from unidecode import unidecode

PATCH_SIZE = 64
PATCH_LENGTH = 512
HIDDEN_SIZE = 768
NUM_LAYERS = 12


class M3Patchilizer:
    """CLaMP3's character-level MTF-to-patch encoder."""

    def __init__(self):
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = "(" + "|".join(map(re.escape, self.delimiters)) + ")"
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.mask_token_id = 3

    def split_bars(self, body):
        bars = re.split(self.regexPattern, "".join(body))
        bars = list(filter(None, bars))
        if bars[0] in self.delimiters:
            bars[1] = bars[0] + bars[1]
            bars = bars[1:]
        bars = [bars[i * 2] + bars[i * 2 + 1] for i in range(len(bars) // 2)]
        return bars

    def bar2patch(self, bar, patch_size=PATCH_SIZE):
        patch = [self.bos_token_id] + [ord(char) for char in bar] + [self.eos_token_id]
        patch = patch[:patch_size]
        patch += [self.pad_token_id] * (patch_size - len(patch))
        return patch

    def encode(
        self,
        item,
        patch_size=PATCH_SIZE,
        add_special_patches=False,
        truncate=False,
        random_truncate=False,
    ):
        item = unidecode(item.replace("L:1/8\n", ""))
        lines = list(filter(None, re.findall(r".*?\n|.*$", item)))
        patches = []

        if lines[0].split(" ")[0] == "ticks_per_beat":
            patch = ""
            for line in lines:
                if patch.startswith(line.split(" ")[0]) and (
                    len(patch) + len(" ".join(line.split(" ")[1:])) <= patch_size - 2
                ):
                    patch = patch[:-1] + "\t" + " ".join(line.split(" ")[1:])
                else:
                    if patch:
                        patches.append(patch)
                    patch = line
            if patch != "":
                patches.append(patch)
        else:
            for line in lines:
                if len(line) > 1 and (
                    (line[0].isalpha() and line[1] == ":") or line.startswith("%%")
                ):
                    patches.append(line)
                else:
                    bars = self.split_bars(line)
                    if bars:
                        bars[-1] += "\n"
                        patches.extend(bars)

        if add_special_patches:
            patches = [chr(self.bos_token_id) * patch_size] + patches + [
                chr(self.eos_token_id) * patch_size
            ]
        if len(patches) > PATCH_LENGTH and truncate:
            choices = ["head", "tail", "middle"]
            choice = random.choice(choices)
            if choice == "head" or random_truncate == False:
                patches = patches[:PATCH_LENGTH]
            elif choice == "tail":
                patches = patches[-PATCH_LENGTH:]
            else:
                start = random.randint(1, len(patches) - PATCH_LENGTH)
                patches = patches[start : start + PATCH_LENGTH]
        return [self.bar2patch(patch, patch_size) for patch in patches]


def load_midi_mtf(mid: mido.MidiFile, m3_compatible: bool = True) -> str:
    """Serialize a Mido MIDI exactly as the official CLaMP3 MTF loader."""
    messages = ["ticks_per_beat " + str(mid.ticks_per_beat)]
    merged = mido.merge_tracks(mid.tracks)
    ignored_meta = {
        "text", "copyright", "track_name", "instrument_name", "lyrics",
        "marker", "cue_marker", "device_name",
    }
    for msg in merged:
        if m3_compatible and msg.is_meta and msg.type in ignored_meta:
            continue
        messages.append(
            " ".join(str(value) for value in msg.dict().values())
            .encode("unicode_escape")
            .decode("utf-8")
        )
    return "\n".join(messages)


class M3PatchEncoder(nn.Module):
    """Checkpoint-compatible symbolic patch encoder from the CLaMP3 C2 model."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.patch_embedding = nn.Linear(PATCH_SIZE * 128, HIDDEN_SIZE)
        self.base = BertModel(config=config)

    def forward(self, input_patches, input_masks):
        patch_vectors = torch.nn.functional.one_hot(input_patches, num_classes=128)
        patch_vectors = patch_vectors.reshape(
            len(patch_vectors), -1, PATCH_SIZE * 128
        ).type(torch.FloatTensor)
        patch_vectors = self.patch_embedding(patch_vectors.to(self.patch_embedding.weight.device))
        return self.base(inputs_embeds=patch_vectors, attention_mask=input_masks)


def symbolic_checkpoint_state(state):
    prefix = "symbolic_model."
    return {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }

def load_symbolic_encoder(checkpoint_path: str, device: str = "cuda") -> M3PatchEncoder:
    config = BertConfig(
        vocab_size=1,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=HIDDEN_SIZE // 64,
        intermediate_size=HIDDEN_SIZE * 4,
        max_position_embeddings=PATCH_LENGTH,
    )
    encoder = M3PatchEncoder(config)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    symbolic_state = symbolic_checkpoint_state(state)
    # Strict loading proves that the official full-model checkpoint contains every
    # expected symbolic encoder tensor with exactly matching names and shapes.
    encoder.load_state_dict(symbolic_state, strict=True)
    return encoder.to(device).eval()


def get_midi_embedding(mid, model: M3PatchEncoder, patchilizer: M3Patchilizer):
    device = model.patch_embedding.weight.device
    mtf = load_midi_mtf(mid, m3_compatible=True)
    patches = patchilizer.encode(mtf, add_special_patches=True)
    token_tensor = torch.tensor(patches, dtype=torch.long, device=device)

    segments = []
    segment_weights = []
    for start in range(0, token_tensor.size(0), PATCH_LENGTH):
        segment = token_tensor[start : start + PATCH_LENGTH]
        segments.append(segment)
        segment_weights.append(segment.size(0))
    if token_tensor.size(0) > PATCH_LENGTH:
        segments[-1] = token_tensor[-PATCH_LENGTH:]
        segment_weights[-1] = segments[-1].size(0)

    features = []
    for segment in segments:
        length = segment.size(0)
        if length < PATCH_LENGTH:
            padding = torch.full(
                (PATCH_LENGTH - length, token_tensor.size(1)),
                patchilizer.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            segment = torch.cat((segment, padding), dim=0)
        mask = torch.cat(
            (
                torch.ones(length, device=device),
                torch.zeros(PATCH_LENGTH - length, device=device),
            )
        ).unsqueeze(0)
        output = model(segment.unsqueeze(0), mask)["last_hidden_state"]
        features.append((output * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True))

    features = torch.stack([feature.squeeze(0) for feature in features], dim=0)
    weights = torch.tensor(segment_weights, dtype=torch.float, device=device).view(-1, 1)
    return ((features * weights).sum(dim=0) / weights.sum()).view(-1)
