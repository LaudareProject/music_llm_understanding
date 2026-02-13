from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from music_latents.models import ModelClient, ModelResult


@dataclass
class AriaEmbeddingClient(ModelClient):
    name: str
    repo_id: str
    checkpoint_filename: str = "model.safetensors"
    model_config_name: str = "medium-emb"
    device: str = "cpu"
    cache_dir: Optional[str] = None
    torch_dtype: Optional[str] = None

    model_type: str = "aria_embedding"

    def __post_init__(self) -> None:
        self._model = None

    def encode_midi(self, midi_path: Path) -> ModelResult:
        model = self._load_model()

        embedding = self._get_global_embedding(
            model=model,
            midi_path=midi_path,
            device=self.device,
        )

        latent = embedding.detach().cpu().numpy()
        metadata: Dict[str, Any] = {
            "model_type": self.model_type,
            "model_name": self.name,
            "repo_id": self.repo_id,
            "checkpoint_filename": self.checkpoint_filename,
            "model_config_name": self.model_config_name,
            "device": self.device,
            "latent_shape": list(latent.shape),
        }
        return ModelResult(latent=latent, metadata=metadata)

    def _load_model(self):
        if self._model is not None:
            return self._model

        self._ensure_aria_on_path()
        checkpoint_path = self._download_checkpoint()

        try:
            import torch
            from aria.embedding import get_global_embedding_from_midi
            from aria.model import TransformerEMB, ModelConfig
            from aria.config import load_model_config
            from ariautils.tokenizer import AbsTokenizer
            from safetensors.torch import load_file
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "aria and its dependencies are required to use the Aria embedding model."
            ) from exc

        model_config = ModelConfig(**load_model_config(name=self.model_config_name))
        model_config.set_vocab_size(AbsTokenizer().vocab_size)
        model = TransformerEMB(model_config)
        state_dict = load_file(filename=checkpoint_path)
        model.load_state_dict(state_dict=state_dict, strict=True)
        model.eval()

        if self.torch_dtype:
            model = model.to(dtype=getattr(torch, self.torch_dtype))
        model = model.to(self.device)

        self._get_global_embedding = get_global_embedding_from_midi
        self._model = model
        return model

    def _ensure_aria_on_path(self) -> None:
        import sys
        from pathlib import Path as SysPath

        repo_root = SysPath(__file__).resolve().parents[2]
        aria_path = repo_root / "aria"
        if str(aria_path) not in sys.path:
            sys.path.insert(0, str(aria_path))

    def _download_checkpoint(self) -> str:
        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "huggingface_hub is required to download Aria checkpoints."
            ) from exc

        return hf_hub_download(
            repo_id=self.repo_id,
            filename=self.checkpoint_filename,
            cache_dir=self.cache_dir,
        )
