from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import hashlib

import numpy as np

from music_latents.models import ModelClient, ModelResult


@dataclass
class DummyClient(ModelClient):
    name: str
    seed: str = "dummy"

    model_type: str = "dummy"

    def encode_midi(self, midi_path: Path) -> ModelResult:
        payload = midi_path.read_bytes() + self.seed.encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        latent = np.frombuffer(digest[:16], dtype=np.uint8).astype(np.float32)
        latent = latent.reshape(1, -1)

        metadata: Dict[str, Any] = {
            "model_type": self.model_type,
            "model_name": self.name,
            "seed": self.seed,
            "latent_shape": list(latent.shape),
        }
        return ModelResult(latent=latent, metadata=metadata)
