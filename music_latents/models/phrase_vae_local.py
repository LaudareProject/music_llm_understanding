from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
from huggingface_hub import PyTorchModelHubMixin


def _load_s2svae3_class():
    repo_root = Path(__file__).resolve().parents[2]
    phrase_ldm_path = repo_root / "PhraseLDM_code"
    if str(phrase_ldm_path) not in sys.path:
        sys.path.insert(0, str(phrase_ldm_path))
    from models.phrase_vae import S2SVAE3

    return S2SVAE3


class LocalPhraseVAE(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        tokenizer_repo_id: str = "LongshenOu/phrase-vae-tokenizer",
        t5_config: dict | None = None,
        compress_style: str = "first_n_tokens",
        n_compress_tokens: int = 4,
        bottleneck_dim: int = 64,
        vocab_size: int = 1000,
    ) -> None:
        super().__init__()

        if t5_config is None:
            t5_config = {
                "d_model": 512,
                "d_ff": 1024,
                "num_layers": 3,
                "num_heads": 6,
                "vocab_size": vocab_size,
                "decoder_start_token_id": 1,
            }

        self.tokenizer_repo_id = tokenizer_repo_id
        self.t5_config = t5_config
        self.compress_style = compress_style
        self.n_compress_tokens = n_compress_tokens
        self.bottleneck_dim = bottleneck_dim

        s2svae3 = _load_s2svae3_class()
        self.model = s2svae3(
            tokenizer_path=tokenizer_repo_id,
            t5_config=t5_config,
            compress_style=compress_style,
            n_compress_tokens=n_compress_tokens,
            bottleneck_dim=bottleneck_dim,
        )

        self._tk = None
        self.eval()

    @property
    def tk(self):
        if self._tk is None:
            self._tk = PreTrainedTokenizerFast.from_pretrained(self.tokenizer_repo_id)
        return self._tk

    @classmethod
    def from_lightning_ckpt(cls, ckpt_fp: str, device="cpu", torch_dtype=None):
        ckpt = torch.load(ckpt_fp, map_location="cpu")
        hp = ckpt.get("hyper_parameters", {})
        tokenizer_repo_id = hp.get(
            "tokenizer_repo_id", "LongshenOu/phrase-vae-tokenizer"
        )
        t5_config = hp.get(
            "t5_config",
            {
                "d_model": 512,
                "d_ff": 1024,
                "num_layers": 3,
                "num_heads": 6,
                "vocab_size": 1000,
                "decoder_start_token_id": 1,
            },
        )
        compress_style = hp.get("compress_style", "first_n_tokens")
        n_compress_tokens = hp.get("n_compress_tokens", 4)
        bottleneck_dim = hp.get("bottleneck_dim", 64)

        obj = cls(
            tokenizer_repo_id=tokenizer_repo_id,
            t5_config=t5_config,
            compress_style=compress_style,
            n_compress_tokens=n_compress_tokens,
            bottleneck_dim=bottleneck_dim,
        )

        sd_new = {}
        for k, v in ckpt["state_dict"].items():
            if k.startswith("model."):
                sd_new[k[len("model.") :]] = v

        obj.model.load_state_dict(sd_new, strict=True)
        obj.eval()
        if torch_dtype is not None:
            obj = obj.to(dtype=torch_dtype)
        obj = obj.to(device)
        return obj

    def encode_batch(self, remiz_strs, do_sample=False):
        bar_strs = []
        for bar_str in remiz_strs:
            inp_seq = bar_str.split()
            inp_seq = ["[BOS]"] + inp_seq + ["[EOS]"]
            inp_seq = [f"[C{i}]" for i in range(4)] + inp_seq
            bar_strs.append(" ".join(inp_seq))

        inputs = self.tk(
            bar_strs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        input_ids = inputs.input_ids
        attn_mask = inputs.attention_mask

        with torch.no_grad():
            encoder_out = self.model.encode(input_ids, attn_mask, sample=do_sample)
            latents = encoder_out["latent"]

        return latents
