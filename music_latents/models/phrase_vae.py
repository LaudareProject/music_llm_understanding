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

        content = _DEFAULT_TIME_SIGNATURE_YAML
        with open(ts_path, "w", encoding="utf-8") as handle:
            handle.write(content)


_DEFAULT_TIME_SIGNATURE_YAML = """s-0: (1, 1)
s-1: (2, 1)
s-2: (1, 2)
s-3: (2, 2)
s-4: (3, 2)
s-5: (4, 2)
s-6: (1, 4)
s-7: (2, 4)
s-8: (3, 4)
s-9: (4, 4)
s-10: (5, 4)
s-11: (6, 4)
s-12: (7, 4)
s-13: (8, 4)
s-14: (1, 8)
s-15: (2, 8)
s-16: (3, 8)
s-17: (4, 8)
s-18: (5, 8)
s-19: (6, 8)
s-20: (7, 8)
s-21: (8, 8)
s-22: (9, 8)
s-23: (10, 8)
s-24: (11, 8)
s-25: (12, 8)
s-26: (13, 8)
s-27: (14, 8)
s-28: (15, 8)
s-29: (16, 8)
s-30: (1, 16)
s-31: (2, 16)
s-32: (3, 16)
s-33: (4, 16)
s-34: (5, 16)
s-35: (6, 16)
s-36: (7, 16)
s-37: (8, 16)
s-38: (9, 16)
s-39: (10, 16)
s-40: (11, 16)
s-41: (12, 16)
s-42: (13, 16)
s-43: (14, 16)
s-44: (15, 16)
s-45: (16, 16)
s-46: (17, 16)
s-47: (18, 16)
s-48: (19, 16)
s-49: (20, 16)
s-50: (21, 16)
s-51: (22, 16)
s-52: (23, 16)
s-53: (24, 16)
s-54: (25, 16)
s-55: (26, 16)
s-56: (27, 16)
s-57: (28, 16)
s-58: (29, 16)
s-59: (30, 16)
s-60: (31, 16)
s-61: (32, 16)
s-62: (1, 32)
s-63: (2, 32)
s-64: (3, 32)
s-65: (4, 32)
s-66: (5, 32)
s-67: (6, 32)
s-68: (7, 32)
s-69: (8, 32)
s-70: (9, 32)
s-71: (10, 32)
s-72: (11, 32)
s-73: (12, 32)
s-74: (13, 32)
s-75: (14, 32)
s-76: (15, 32)
s-77: (16, 32)
s-78: (17, 32)
s-79: (18, 32)
s-80: (19, 32)
s-81: (20, 32)
s-82: (21, 32)
s-83: (22, 32)
s-84: (23, 32)
s-85: (24, 32)
s-86: (25, 32)
s-87: (26, 32)
s-88: (27, 32)
s-89: (28, 32)
s-90: (29, 32)
s-91: (30, 32)
s-92: (31, 32)
s-93: (32, 32)
s-94: (33, 32)
s-95: (34, 32)
s-96: (35, 32)
s-97: (36, 32)
s-98: (37, 32)
s-99: (38, 32)
s-100: (39, 32)
s-101: (40, 32)
s-102: (41, 32)
s-103: (42, 32)
s-104: (43, 32)
s-105: (44, 32)
s-106: (45, 32)
s-107: (46, 32)
s-108: (47, 32)
s-109: (48, 32)
s-110: (49, 32)
s-111: (50, 32)
s-112: (51, 32)
s-113: (52, 32)
s-114: (53, 32)
s-115: (54, 32)
s-116: (55, 32)
s-117: (56, 32)
s-118: (57, 32)
s-119: (58, 32)
s-120: (59, 32)
s-121: (60, 32)
s-122: (61, 32)
s-123: (62, 32)
s-124: (63, 32)
s-125: (64, 32)
s-126: (1, 64)
s-127: (2, 64)
s-128: (3, 64)
s-129: (4, 64)
s-130: (5, 64)
s-131: (6, 64)
s-132: (7, 64)
s-133: (8, 64)
s-134: (9, 64)
s-135: (10, 64)
s-136: (11, 64)
s-137: (12, 64)
s-138: (13, 64)
s-139: (14, 64)
s-140: (15, 64)
s-141: (16, 64)
s-142: (17, 64)
s-143: (18, 64)
s-144: (19, 64)
s-145: (20, 64)
s-146: (21, 64)
s-147: (22, 64)
s-148: (23, 64)
s-149: (24, 64)
s-150: (25, 64)
s-151: (26, 64)
s-152: (27, 64)
s-153: (28, 64)
s-154: (29, 64)
s-155: (30, 64)
s-156: (31, 64)
s-157: (32, 64)
s-158: (33, 64)
s-159: (34, 64)
s-160: (35, 64)
s-161: (36, 64)
s-162: (37, 64)
s-163: (38, 64)
s-164: (39, 64)
s-165: (40, 64)
s-166: (41, 64)
s-167: (42, 64)
s-168: (43, 64)
s-169: (44, 64)
s-170: (45, 64)
s-171: (46, 64)
s-172: (47, 64)
s-173: (48, 64)
s-174: (49, 64)
s-175: (50, 64)
s-176: (51, 64)
s-177: (52, 64)
s-178: (53, 64)
s-179: (54, 64)
s-180: (55, 64)
s-181: (56, 64)
s-182: (57, 64)
s-183: (58, 64)
s-184: (59, 64)
s-185: (60, 64)
s-186: (61, 64)
s-187: (62, 64)
s-188: (63, 64)
s-189: (64, 64)
s-190: (65, 64)
s-191: (66, 64)
s-192: (67, 64)
s-193: (68, 64)
s-194: (69, 64)
s-195: (70, 64)
s-196: (71, 64)
s-197: (72, 64)
s-198: (73, 64)
s-199: (74, 64)
s-200: (75, 64)
s-201: (76, 64)
s-202: (77, 64)
s-203: (78, 64)
s-204: (79, 64)
s-205: (80, 64)
s-206: (81, 64)
s-207: (82, 64)
s-208: (83, 64)
s-209: (84, 64)
s-210: (85, 64)
s-211: (86, 64)
s-212: (87, 64)
s-213: (88, 64)
s-214: (89, 64)
s-215: (90, 64)
s-216: (91, 64)
s-217: (92, 64)
s-218: (93, 64)
s-219: (94, 64)
s-220: (95, 64)
s-221: (96, 64)
s-222: (97, 64)
s-223: (98, 64)
s-224: (99, 64)
s-225: (100, 64)
s-226: (101, 64)
s-227: (102, 64)
s-228: (103, 64)
s-229: (104, 64)
s-230: (105, 64)
s-231: (106, 64)
s-232: (107, 64)
s-233: (108, 64)
s-234: (109, 64)
s-235: (110, 64)
s-236: (111, 64)
s-237: (112, 64)
s-238: (113, 64)
s-239: (114, 64)
s-240: (115, 64)
s-241: (116, 64)
s-242: (117, 64)
s-243: (118, 64)
s-244: (119, 64)
s-245: (120, 64)
s-246: (121, 64)
s-247: (122, 64)
s-248: (123, 64)
s-249: (124, 64)
s-250: (125, 64)
s-251: (126, 64)
s-252: (127, 64)
s-253: (128, 64)
"""
