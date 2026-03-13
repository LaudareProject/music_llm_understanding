from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import tempfile
from typing import Iterable
from urllib.request import urlopen
from xml.etree import ElementTree as ET

from music_latents.humdrum_io import as_uncompressed_musicxml, run_command
from music_latents.humdrum_keycor import ensemble_vote, parse_keycor_scores
from music_latents.humdrum_types import HUMDRUM_LINUX_BIN_BASE_URL, HumdrumError

_as_uncompressed_musicxml = as_uncompressed_musicxml
_run_command = run_command


PROFILE_FLAGS: dict[str, list[str]] = {
    "kk": [],
    "simple": ["-s"],
    "aarden": ["--aarden"],
    "bellman": ["--bellman"],
    "temperley": ["--temperley"],
}


@dataclass(frozen=True)
class KeyCorrelationResult:
    profile: str
    tonality: str
    score: float


def compute_semitone_shift(
    average_pitch: float | None,
    target_pitch: float = 60.0,
) -> int:
    if average_pitch is None:
        return 0
    return int(round(target_pitch - average_pitch))


def convert_musicxml_to_midi(
    musicxml_path: Path, midi_path: Path, bin_dir: Path
) -> None:
    xml2hum = _resolve_binary(bin_dir, "xml2hum")
    hum2mid = _resolve_binary(bin_dir, "hum2mid")

    with _as_uncompressed_musicxml(musicxml_path) as input_musicxml:
        hum_text = _run_command(
            [str(xml2hum), str(input_musicxml)], cwd=input_musicxml.parent
        )

    midi_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".krn", delete=False) as handle:
        temp_hum_path = Path(handle.name)
        handle.write(hum_text)

    try:
        _run_command(
            [str(hum2mid), str(temp_hum_path), "-o", str(midi_path)],
            cwd=midi_path.parent,
        )
    finally:
        temp_hum_path.unlink(missing_ok=True)


def convert_musicxml_to_centered_midi(
    musicxml_path: Path,
    midi_path: Path,
    bin_dir: Path,
    target_pitch: float = 60.0,
) -> int:
    xml2hum = _resolve_binary(bin_dir, "xml2hum")
    transpose = _resolve_binary(bin_dir, "transpose")
    hum2mid = _resolve_binary(bin_dir, "hum2mid")

    avg_pitch = estimate_average_midi_pitch_from_musicxml(musicxml_path)
    requested_shift = compute_semitone_shift(avg_pitch, target_pitch)
    applied_shift = requested_shift

    with _as_uncompressed_musicxml(musicxml_path) as input_musicxml:
        hum_text = _run_command(
            [str(xml2hum), str(input_musicxml)], cwd=input_musicxml.parent
        )

    if requested_shift != 0:
        try:
            hum_text = _run_command(
                [str(transpose), "-q", "-d", "0", "-c", str(requested_shift)],
                stdin_text=hum_text,
            )
        except HumdrumError:
            # Some scores contain pathological pitch encodings that make the
            # Humdrum transposer fail (e.g., unreasonable octave values).
            # Fall back to untransposed Humdrum for robustness.
            applied_shift = 0

    midi_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".krn", delete=False) as handle:
        temp_hum_path = Path(handle.name)
        handle.write(hum_text)

    try:
        _run_command(
            [str(hum2mid), str(temp_hum_path), "-o", str(midi_path)],
            cwd=midi_path.parent,
        )
    finally:
        temp_hum_path.unlink(missing_ok=True)

    return applied_shift


def estimate_average_midi_pitch_from_musicxml(musicxml_path: Path) -> float | None:
    with _as_uncompressed_musicxml(musicxml_path) as input_musicxml:
        root = ET.parse(input_musicxml).getroot()

    pitches: list[int] = []
    for note_el in root.iter():
        if not _is_tag(note_el.tag, "note"):
            continue
        if _find_child(note_el, "rest") is not None:
            continue

        pitch_el = _find_child(note_el, "pitch")
        if pitch_el is None:
            continue
        step = _find_child_text(pitch_el, "step")
        octave = _find_child_text(pitch_el, "octave")
        alter = _find_child_text(pitch_el, "alter")
        midi = _musicxml_pitch_to_midi(step, octave, alter)
        if midi is not None:
            pitches.append(midi)

    if not pitches:
        return None
    return float(sum(pitches) / len(pitches))


def _musicxml_pitch_to_midi(
    step_text: str | None,
    octave_text: str | None,
    alter_text: str | None,
) -> int | None:
    if step_text is None or octave_text is None:
        return None
    step_to_pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    step = step_text.strip().upper()
    if step not in step_to_pc:
        return None
    try:
        octave = int(octave_text.strip())
    except ValueError:
        return None

    alter = 0
    if alter_text is not None:
        try:
            alter = int(round(float(alter_text.strip())))
        except ValueError:
            alter = 0

    return int((octave + 1) * 12 + step_to_pc[step] + alter)


def _is_tag(tag: str, local: str) -> bool:
    return tag == local or tag.endswith(f"}}{local}")


def _find_child(element: ET.Element, local_name: str) -> ET.Element | None:
    for child in list(element):
        if _is_tag(child.tag, local_name):
            return child
    return None


def _find_child_text(element: ET.Element, local_name: str) -> str | None:
    child = _find_child(element, local_name)
    if child is None or child.text is None:
        return None
    return child.text


def estimate_tonality_by_voting(
    musicxml_path: Path,
    bin_dir: Path,
    profiles: Iterable[str] = ("kk", "simple", "aarden", "bellman", "temperley"),
) -> tuple[str, list[KeyCorrelationResult]]:
    xml2hum = _resolve_binary(bin_dir, "xml2hum")
    keycor = _resolve_binary(bin_dir, "keycor")

    with _as_uncompressed_musicxml(musicxml_path) as input_musicxml:
        hum_text = _run_command(
            [str(xml2hum), str(input_musicxml)], cwd=input_musicxml.parent
        )
    results: list[KeyCorrelationResult] = []

    for profile in profiles:
        flags = PROFILE_FLAGS.get(profile)
        if flags is None:
            raise ValueError(f"Unsupported keycor profile: {profile}")
        output = _run_command([str(keycor), "-a", *flags], stdin_text=hum_text)
        rows = parse_keycor_scores(output)
        if not rows:
            raise HumdrumError(
                f"No key correlation scores parsed for profile '{profile}' on {musicxml_path}."
            )
        tonality, score = max(rows, key=lambda entry: entry[1])
        results.append(
            KeyCorrelationResult(profile=profile, tonality=tonality, score=score)
        )

    return ensemble_vote(results), results


def _resolve_binary(bin_dir: Path, name: str) -> Path:
    candidate = (bin_dir / name).resolve()
    if not candidate.exists():
        _download_humdrum_binary(name=name, destination=candidate)
    if not candidate.is_file():
        raise FileNotFoundError(f"Humdrum binary path is not a file: {candidate}")
    if not candidate.stat().st_mode & 0o111:
        candidate.chmod(candidate.stat().st_mode | 0o755)
    return candidate


def _download_humdrum_binary(name: str, destination: Path) -> None:
    url = f"{HUMDRUM_LINUX_BIN_BASE_URL}/{name}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=30) as response:
        payload = response.read()
    if not payload:
        raise FileNotFoundError(
            f"Downloaded empty binary payload for '{name}' from {url}."
        )
    destination.write_bytes(payload)
    destination.chmod(destination.stat().st_mode | 0o755)
