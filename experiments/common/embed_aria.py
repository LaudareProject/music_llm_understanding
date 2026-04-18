from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile

import mido
import numpy as np


MODEL = None
GET_GLOBAL_EMBEDDING = None
REPO_ID = "loubb/aria-medium-embedding"
CHECKPOINT_FILENAME = "model.safetensors"
MODEL_CONFIG_NAME = "medium-emb"
DEVICE = "cuda"


def load_model():
    global MODEL, GET_GLOBAL_EMBEDDING
    if MODEL is not None:
        return MODEL, GET_GLOBAL_EMBEDDING

    repo_root = Path(__file__).resolve().parents[2]
    aria_path = repo_root / "aria"
    if str(aria_path) not in sys.path:
        sys.path.insert(0, str(aria_path))

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from aria.config import load_model_config
    from aria.embedding import get_global_embedding_from_midi
    from aria.model import ModelConfig, TransformerEMB
    from ariautils.tokenizer import AbsTokenizer

    checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename=CHECKPOINT_FILENAME)
    model_config = ModelConfig(**load_model_config(name=MODEL_CONFIG_NAME))
    model_config.set_vocab_size(AbsTokenizer().vocab_size)
    model = TransformerEMB(model_config)
    model.load_state_dict(load_file(filename=checkpoint_path), strict=True)
    model.eval()
    model = model.to(DEVICE)

    MODEL = model
    GET_GLOBAL_EMBEDDING = get_global_embedding_from_midi
    return MODEL, GET_GLOBAL_EMBEDDING


def needs_piano_normalization(midi_path):
    midi = mido.MidiFile(str(midi_path))
    for track in midi.tracks:
        for message in track:
            if message.is_meta:
                continue
            if hasattr(message, "channel") and int(message.channel) != 0:
                return True
            if message.type == "program_change" and int(message.program) != 0:
                return True
    return False


def write_single_piano_midi(source_path, output_path):
    source_midi = mido.MidiFile(str(source_path))
    normalized_midi = mido.MidiFile(
        type=source_midi.type,
        ticks_per_beat=source_midi.ticks_per_beat,
        charset=getattr(source_midi, "charset", "latin1"),
    )
    for source_track in source_midi.tracks:
        track = mido.MidiTrack()
        for message in source_track:
            if message.is_meta:
                track.append(message.copy())
                continue
            updates = {}
            if hasattr(message, "channel"):
                updates["channel"] = 0
            if message.type == "program_change":
                updates["program"] = 0
            track.append(message.copy(**updates))
        normalized_midi.tracks.append(track)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_midi.save(str(output_path))


@contextmanager
def aria_ready_midi(midi_path):
    if not needs_piano_normalization(midi_path):
        yield midi_path
        return
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as handle:
        normalized_path = Path(handle.name)
    try:
        write_single_piano_midi(midi_path, normalized_path)
        yield normalized_path
    finally:
        normalized_path.unlink(missing_ok=True)


def encode_midi(midi_path):
    model, get_global_embedding = load_model()
    with aria_ready_midi(midi_path) as ready_path:
        embedding = get_global_embedding(
            model=model, midi_path=ready_path, device=DEVICE
        )
    return np.asarray(embedding.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
