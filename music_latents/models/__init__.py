from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np


@dataclass
class ModelResult:
    latent: np.ndarray
    metadata: Dict[str, Any]


class ModelClient:
    model_type: str

    def encode_midi(self, midi_path: Path) -> ModelResult:
        raise NotImplementedError


def create_model_client(
    model_type: str, name: str, config: Dict[str, Any]
) -> ModelClient:
    if model_type == "aria_embedding":
        from music_latents.models.aria_embedding import AriaEmbeddingClient

        return AriaEmbeddingClient(name=name, **config)

    if model_type == "phrase_vae":
        from music_latents.models.phrase_vae import PhraseVAEClient

        return PhraseVAEClient(name=name, **config)

    if model_type == "dummy":
        from music_latents.models.dummy import DummyClient

        return DummyClient(name=name, **config)

    raise ValueError(f"Unsupported model type: {model_type}")
