from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from music_latents.models import ModelClient, ModelResult


@dataclass
class PhraseVAEClient(ModelClient):
    name: str
    repo_id: Optional[str] = None
    checkpoint_filename: Optional[str] = None
    lightning_ckpt_path: Optional[str] = None
    device: str = "cpu"
    torch_dtype: Optional[str] = None
    cache_dir: Optional[str] = None

    model_type: str = "phrase_vae"

    def __post_init__(self) -> None:
        self._model = None

    def encode_midi(self, midi_path: Path) -> ModelResult:
        model = self._load_model()
        remiz_str = self._midi_to_remiz(midi_path)

        latents = model.encode_batch([remiz_str])
        latent = latents.detach().cpu().numpy()
        metadata: Dict[str, Any] = {
            "model_type": self.model_type,
            "model_name": self.name,
            "repo_id": self.repo_id,
            "checkpoint_filename": self.checkpoint_filename,
            "lightning_ckpt_path": self.lightning_ckpt_path,
            "device": self.device,
            "latent_shape": list(latent.shape),
        }
        return ModelResult(latent=latent, metadata=metadata)

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "torch is required to use the PhraseVAE model."
            ) from exc

        phrase_vae = self._resolve_phrase_vae_class()

        if self.lightning_ckpt_path:
            model = phrase_vae.from_lightning_ckpt(
                self.lightning_ckpt_path,
                device=self.device,
                torch_dtype=getattr(torch, self.torch_dtype)
                if self.torch_dtype
                else None,
            )
        elif self.repo_id:
            model = self._load_from_hub(phrase_vae)
        else:
            raise ValueError(
                "PhraseVAE requires either 'lightning_ckpt_path' or 'repo_id'."
            )

        model.eval()
        self._model = model
        return model

    def _load_from_hub(self, phrase_vae):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "torch is required to load PhraseVAE checkpoints."
            ) from exc

        if not self.checkpoint_filename:
            model = phrase_vae.from_pretrained(
                self.repo_id,
                cache_dir=self.cache_dir,
            )
        else:
            checkpoint_path = self._download_checkpoint(
                self.repo_id, self.checkpoint_filename
            )
            model = phrase_vae()
            state_dict = self._load_state_dict(checkpoint_path)
            model.load_state_dict(state_dict, strict=True)

        if self.torch_dtype:
            model = model.to(dtype=getattr(torch, self.torch_dtype))
        model = model.to(self.device)
        return model

    def _download_checkpoint(self, repo_id: str, filename: str) -> str:
        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "huggingface_hub is required to download PhraseVAE checkpoints."
            ) from exc

        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=self.cache_dir,
        )

    def _load_state_dict(self, checkpoint_path: str):
        if checkpoint_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "safetensors is required to load safetensors PhraseVAE checkpoints."
                ) from exc
            return load_file(checkpoint_path)

        import torch

        return torch.load(checkpoint_path, map_location="cpu")

    def _resolve_phrase_vae_class(self):
        try:
            from music_latents.models.phrase_vae_local import LocalPhraseVAE
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PhraseLDM_code and its dependencies are required to use PhraseVAE."
            ) from exc
        return LocalPhraseVAE

    def _midi_to_remiz(self, midi_path: Path) -> str:
        try:
            self._ensure_remi_z_time_signature_dict()
            from remi_z import MultiTrack
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "remi_z is required to convert MIDI to REMI-z strings for PhraseVAE."
            ) from exc

        mt = MultiTrack.from_midi(str(midi_path))
        tokens = mt.to_remiz_seq()
        return " ".join(tokens)

    def _ensure_remi_z_time_signature_dict(self) -> None:
        import importlib
        import os

        remi_z = importlib.import_module("remi_z")
        ts_path = os.path.join(
            os.path.dirname(remi_z.__file__), "dict_time_signature.yaml"
        )
        if os.path.exists(ts_path):
            return

        bundled_path = Path(__file__).with_name("default_time_signature.yaml")
        if bundled_path.exists():
            content = bundled_path.read_text(encoding="utf-8")
        else:
            content = _build_default_time_signature_yaml()
        with open(ts_path, "w", encoding="utf-8") as handle:
            handle.write(content)


def _build_default_time_signature_yaml() -> str:
    buckets = [
        (1, [1, 2]),
        (2, [1, 2, 3, 4]),
        (4, list(range(1, 9))),
        (8, list(range(1, 17))),
        (16, list(range(1, 33))),
        (32, list(range(1, 65))),
        (64, list(range(1, 129))),
    ]
    lines: list[str] = []
    idx = 0
    for denominator, numerators in buckets:
        for numerator in numerators:
            lines.append(f"s-{idx}: ({numerator}, {denominator})")
            idx += 1
    return "\n".join(lines) + "\n"
