from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from music_latents.models.aria_embedding import AriaEmbeddingClient
from music_latents.models.phrase_vae import PhraseVAEClient


def _skip_if_disabled() -> None:
    if os.getenv("RUN_MODEL_LOAD_TESTS") != "1":
        pytest.skip("Set RUN_MODEL_LOAD_TESTS=1 to run model loading tests.")


def test_load_aria_embedding_model() -> None:
    _skip_if_disabled()
    client = AriaEmbeddingClient(
        name="aria_embedding",
        repo_id="loubb/aria-medium-embedding",
        checkpoint_filename="model.safetensors",
        model_config_name="medium-emb",
        device="cpu",
    )
    model = client._load_model()
    assert model is not None


def test_load_phrase_vae_model() -> None:
    _skip_if_disabled()
    client = PhraseVAEClient(
        name="phrase_vae",
        repo_id="LongshenOu/phrase-vae",
        device="cpu",
    )
    model = client._load_model()
    assert model is not None
