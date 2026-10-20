"""CLaMP3 C2 adapter backed by the official M3 code in the Aria submodule."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import mido
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import BertConfig


REPO_ID = "sander-wood/clamp3"
CHECKPOINT_FILENAME = (
    "weights_clamp3_c2_h_size_768_t_model_FacebookAI_xlm-roberta-base_"
    "t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_"
    "s_layers_12_p_size_64_p_length_512.pth"
)
PATCH_LENGTH = 512
MODEL = None
PATCHILIZER = None
MIDI_LOADER = None
DEVICE = None


def _load_model():
    global MODEL, PATCHILIZER, MIDI_LOADER, DEVICE
    if MODEL is not None:
        return MODEL, PATCHILIZER, MIDI_LOADER

    aria_path = Path(__file__).resolve().parents[2] / "aria"
    if not (aria_path / "aria" / "eval" / "m3" / "emb.py").is_file():
        raise ImportError(
            "The Aria submodule is missing; initialize submodules with "
            "`git submodule update --init --recursive`."
        )
    if str(aria_path) not in sys.path:
        sys.path.insert(0, str(aria_path))

    # Reuse upstream M3 tokenization and encoder classes. Instantiate only the
    # symbolic patch encoder: the full CLaMP3 model would also load text/audio
    # encoders that this MIDI-only benchmark never calls.
    from aria.eval.m3.config import M3_HIDDEN_SIZE, PATCH_NUM_LAYERS
    from aria.eval.m3.emb import load_midi
    from aria.eval.m3.utils import M3PatchEncoder, M3Patchilizer

    device = os.environ.get(
        "CLAMP3_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint_path = hf_hub_download(
        repo_id=REPO_ID, filename=CHECKPOINT_FILENAME
    )
    config = BertConfig(
        vocab_size=1,
        hidden_size=M3_HIDDEN_SIZE,
        num_hidden_layers=PATCH_NUM_LAYERS,
        num_attention_heads=M3_HIDDEN_SIZE // 64,
        intermediate_size=M3_HIDDEN_SIZE * 4,
        max_position_embeddings=PATCH_LENGTH,
    )
    model = M3PatchEncoder(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # The C2 checkpoint also contains text/audio weights; load only the
    # checkpoint-compatible symbolic encoder and fail on any tensor mismatch.
    symbolic_state = _symbolic_checkpoint_state(checkpoint["model"])
    model.load_state_dict(symbolic_state, strict=True)
    MODEL = model.to(device).eval()
    PATCHILIZER = M3Patchilizer()
    MIDI_LOADER = load_midi
    DEVICE = device
    return MODEL, PATCHILIZER, MIDI_LOADER


def _symbolic_checkpoint_state(state):
    """Select only the symbolic encoder tensors from the multimodal checkpoint."""
    return {
        key.removeprefix("symbolic_model."): value
        for key, value in state.items()
        if key.startswith("symbolic_model.")
    }


def _embed_midi_with_components(midi, model, patchilizer, midi_loader, device):
    """Encode official MTF patches and merge context windows into one vector.

    CLaMP3's encoder accepts 512 patches at a time. The final window overlaps
    earlier patches for long pieces, matching Aria's inference helper; window
    means are weighted by patch count to produce one fixed-size benchmark vector.
    """
    mtf = midi_loader(mid=midi, m3_compatible=True)
    patches = patchilizer.encode(mtf, add_special_patches=True)
    if not patches:
        raise ValueError("CLaMP3 produced no MTF patches")

    token_tensor = torch.tensor(patches, dtype=torch.long, device=device)
    segments = [
        token_tensor[start : start + PATCH_LENGTH]
        for start in range(0, token_tensor.size(0), PATCH_LENGTH)
    ]
    segment_weights = [segment.size(0) for segment in segments]
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
        hidden = model(segment.unsqueeze(0), mask)["last_hidden_state"]
        features.append(
            (hidden * mask.unsqueeze(-1)).sum(dim=1)
            / mask.sum(dim=1, keepdim=True)
        )

    # Weight segment means by their encoded patch counts, not by window count.
    # This returns one vector while keeping a short final segment from weighing
    # as much as a full 512-patch segment.
    features = torch.stack([feature.squeeze(0) for feature in features])
    weights = torch.tensor(
        segment_weights, dtype=features.dtype, device=device
    ).view(-1, 1)
    return ((features * weights).sum(dim=0) / weights.sum()).view(-1)


def encode_midi(midi_path):
    model, patchilizer, midi_loader = _load_model()
    midi = mido.MidiFile(str(midi_path))
    with torch.inference_mode():
        embedding = _embed_midi_with_components(
            midi, model, patchilizer, midi_loader, DEVICE
        )
    vector = np.asarray(embedding.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
    if vector.size != 768 or not np.isfinite(vector).all():
        raise ValueError(f"CLaMP3 C2 produced an invalid embedding for {midi_path}")
    return vector
