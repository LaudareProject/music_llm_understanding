from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Optional

from music_latents.config import QuerySpec


@dataclass(frozen=True)
class InputInfo:
    input_kind: str
    source: str
    stored_path: Optional[Path]
    midi_path: Path


def prepare_midi_input(query: QuerySpec, base_dir: Path, query_dir: Path) -> InputInfo:
    input_dir = query_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    if query.input_kind == "abc":
        abc_path = input_dir / "input.abc"
        abc_path.write_text(query.value, encoding="utf-8")
        midi_path = input_dir / "input.mid"
        score = _parse_music21_from_string(query.value, fmt="abc")
        _write_music21_midi(score, midi_path)
        return InputInfo(
            input_kind=query.input_kind,
            source="abc",
            stored_path=abc_path,
            midi_path=midi_path,
        )

    if query.input_kind == "custom":
        custom_path = input_dir / "input.custom"
        custom_path.write_text(query.value, encoding="utf-8")
        midi_path = input_dir / "input.mid"
        score = _custom_to_music21(query.value)
        _write_music21_midi(score, midi_path)
        return InputInfo(
            input_kind=query.input_kind,
            source="custom",
            stored_path=custom_path,
            midi_path=midi_path,
        )

    source_path = _resolve_path(base_dir, query.value)
    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {source_path}")

    if query.input_kind == "midi_path":
        stored_path = input_dir / "input.mid"
        shutil.copy2(source_path, stored_path)
        return InputInfo(
            input_kind=query.input_kind,
            source=str(source_path),
            stored_path=stored_path,
            midi_path=stored_path,
        )

    if query.input_kind == "musicxml_path":
        stored_path = input_dir / source_path.name
        shutil.copy2(source_path, stored_path)
        midi_path = input_dir / "input.mid"
        score = _parse_music21_from_file(stored_path)
        _write_music21_midi(score, midi_path)
        return InputInfo(
            input_kind=query.input_kind,
            source=str(source_path),
            stored_path=stored_path,
            midi_path=midi_path,
        )

    if query.input_kind == "mei_path":
        stored_path = input_dir / source_path.name
        shutil.copy2(source_path, stored_path)
        midi_path = input_dir / "input.mid"
        score = _parse_music21_from_file(stored_path)
        _write_music21_midi(score, midi_path)
        return InputInfo(
            input_kind=query.input_kind,
            source=str(source_path),
            stored_path=stored_path,
            midi_path=midi_path,
        )

    raise ValueError(f"Unsupported input kind: {query.input_kind}")


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _parse_music21_from_string(value: str, fmt: str):
    try:
        from music21 import converter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "music21 is required to parse ABC or MusicXML/MEI inputs."
        ) from exc
    return converter.parse(value, format=fmt)


def _parse_music21_from_file(path: Path):
    try:
        from music21 import converter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "music21 is required to parse ABC or MusicXML/MEI inputs."
        ) from exc
    return converter.parse(str(path))


def _write_music21_midi(score, midi_path: Path) -> None:
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    score.write("midi", fp=str(midi_path))


def _custom_to_music21(value: str):
    try:
        from music21 import chord, note, stream
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "music21 is required to parse custom chord inputs."
        ) from exc

    part = stream.Part()
    tokens = [tok for tok in value.replace("|", " ").split() if tok]
    if not tokens:
        raise ValueError("Custom input is empty.")

    for token in tokens:
        if "-" in token:
            pitches = [_normalize_pitch_name(name) for name in token.split("-") if name]
            part.append(chord.Chord(pitches, quarterLength=1.0))
        else:
            part.append(note.Note(_normalize_pitch_name(token), quarterLength=1.0))

    score = stream.Score()
    score.insert(0, part)
    return score


def _normalize_pitch_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("Empty pitch token in custom input.")
    if name[-1].isdigit():
        return name
    return f"{name}4"
