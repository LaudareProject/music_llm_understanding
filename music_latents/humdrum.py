from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlopen
from xml.etree import ElementTree as ET
import zipfile


PROFILE_FLAGS: dict[str, list[str]] = {
    "kk": [],
    "simple": ["-s"],
    "aarden": ["--aarden"],
    "bellman": ["--bellman"],
    "temperley": ["--temperley"],
}

HUMDRUM_LINUX_BIN_BASE_URL = "https://extras.humdrum.org/bin/linux"

NOTE_TO_PITCH_CLASS: dict[str, int] = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}

PITCH_CLASS_TO_NAME = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

KEYCOR_ROW_RE = re.compile(
    r"^\s*([A-G](?:-flat|-sharp|#|b)?)\s+(Major|Minor)\s+(-?\d+(?:\.\d+)?)\s*$"
)
KEYCOR_INDEX_RE = re.compile(r"\b(Major|Minor)\[(\d{1,2})\]\s*=\s*(-?\d+(?:\.\d+)?)")


class HumdrumError(RuntimeError):
    pass


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


def parse_keycor_scores(output: str) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for line in output.splitlines():
        match = KEYCOR_ROW_RE.match(line)
        if match is not None:
            note_name = normalize_note_name(match.group(1))
            mode = match.group(2).lower()
            score = float(match.group(3))
            pitch_class = NOTE_TO_PITCH_CLASS.get(note_name)
            if pitch_class is not None:
                rows.append((f"{PITCH_CLASS_TO_NAME[pitch_class]}_{mode}", score))
            continue

        for indexed_match in KEYCOR_INDEX_RE.finditer(line):
            mode = indexed_match.group(1).lower()
            pitch_class = int(indexed_match.group(2)) % 12
            score = float(indexed_match.group(3))
            rows.append((f"{PITCH_CLASS_TO_NAME[pitch_class]}_{mode}", score))
    return rows


def ensemble_vote(results: list[KeyCorrelationResult]) -> str:
    if not results:
        raise ValueError("Cannot vote with empty key-correlation results.")

    votes: dict[str, int] = {}
    scores: dict[str, float] = {}
    for result in results:
        votes[result.tonality] = votes.get(result.tonality, 0) + 1
        scores[result.tonality] = scores.get(result.tonality, 0.0) + result.score

    best_vote = max(votes.values())
    candidates = [tonality for tonality, count in votes.items() if count == best_vote]
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda tonality: scores.get(tonality, float("-inf")))


def normalize_note_name(name: str) -> str:
    value = name.strip().replace("-flat", "b").replace("-sharp", "#")
    if not value:
        return value
    return value[0].upper() + value[1:]


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
    try:
        with urlopen(url, timeout=30) as response:
            payload = response.read()
    except URLError as exc:
        raise FileNotFoundError(
            f"Missing Humdrum binary '{name}' and auto-download failed from {url}."
        ) from exc

    if not payload:
        raise FileNotFoundError(
            f"Downloaded empty binary payload for '{name}' from {url}."
        )

    destination.write_bytes(payload)
    destination.chmod(destination.stat().st_mode | 0o755)


def _run_command(
    cmd: list[str],
    cwd: Path | None = None,
    stdin_text: str | None = None,
) -> str:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        input=stdin_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise HumdrumError(
            f"Command failed ({completed.returncode}): {' '.join(cmd)} :: {stderr}"
        )
    return completed.stdout


@contextmanager
def _as_uncompressed_musicxml(musicxml_path: Path):
    suffix = musicxml_path.suffix.lower()
    if suffix != ".mxl":
        yield musicxml_path
        return

    try:
        with zipfile.ZipFile(musicxml_path, "r") as archive:
            xml_members = [
                name
                for name in archive.namelist()
                if not name.endswith("/")
                and Path(name).suffix.lower() in {".xml", ".musicxml"}
                and not name.lower().startswith("meta-inf/")
            ]
            if not xml_members:
                raise HumdrumError(f"No MusicXML content found inside {musicxml_path}.")

            selected = sorted(
                xml_members,
                key=lambda item: (
                    0 if item.lower().endswith(".musicxml") else 1,
                    len(item),
                    item.lower(),
                ),
            )[0]
            payload = archive.read(selected)
    except zipfile.BadZipFile as exc:
        raise HumdrumError(f"Invalid MXL archive: {musicxml_path}") from exc

    suffix = Path(selected).suffix or ".xml"
    with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(payload)

    try:
        yield temp_path
    finally:
        temp_path.unlink(missing_ok=True)
