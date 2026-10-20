from __future__ import annotations

import os

import mido
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from experiments.common.clamp3_source import (
    M3Patchilizer,
    get_midi_embedding,
    load_symbolic_encoder,
)


REPO_ID = "sander-wood/clamp3"
CHECKPOINT_FILENAME = (
    "weights_clamp3_c2_h_size_768_t_model_FacebookAI_xlm-roberta-base_"
    "t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_"
    "s_layers_12_p_size_64_p_length_512.pth"
)
MODEL = None
PATCHILIZER = None
DEVICE = None


def _load_model():
    global MODEL, PATCHILIZER, DEVICE
    if MODEL is not None:
        return MODEL, PATCHILIZER

    device = os.environ.get(
        "CLAMP3_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint_path = hf_hub_download(
        repo_id=REPO_ID, filename=CHECKPOINT_FILENAME
    )
    MODEL = load_symbolic_encoder(checkpoint_path, device=device)
    PATCHILIZER = M3Patchilizer()
    DEVICE = device
    return MODEL, PATCHILIZER


def encode_midi(midi_path):
    model, patchilizer = _load_model()
    midi = mido.MidiFile(str(midi_path))
    with torch.inference_mode():
        embedding = get_midi_embedding(midi, model, patchilizer)
    vector = np.asarray(embedding.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
    if vector.size != 768 or not np.isfinite(vector).all():
        raise ValueError(f"CLaMP3 C2 produced an invalid embedding for {midi_path}")
    return vector
