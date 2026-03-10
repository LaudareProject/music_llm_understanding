from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List
from xml.etree import ElementTree as ET
import zipfile

from music_latents.humdrum import HumdrumError, estimate_tonality_by_voting


PITCH_CLASS_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MODES = ["major", "minor"]
TARGET_TONALITIES = [f"{pitch}_{mode}" for mode in MODES for pitch in PITCH_CLASS_NAMES]


@dataclass(frozen=True)
class TonalityAnalysis:
    file_path: Path
    file_size: int
    tonality: str | None
    accepted: bool
    reason: str


def find_musicxml_files(dataset_root: Path) -> List[Path]:
    patterns = ("**/*.mxl", "**/*.musicxml", "**/*.xml")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(path for path in dataset_root.glob(pattern) if path.is_file())
    return sorted({path.resolve() for path in files})


def canonical_tonality(tonic_pitch_class: int, mode: str) -> str:
    normalized_mode = mode.lower().strip()
    if normalized_mode not in {"major", "minor"}:
        raise ValueError(f"Unsupported mode: {mode}")
    tonic = PITCH_CLASS_NAMES[int(tonic_pitch_class) % 12]
    return f"{tonic}_{normalized_mode}"


def analyze_musicxml_tonality(
    file_path: Path,
    bin_dir: Path | None = None,
) -> TonalityAnalysis:
    file_size = file_path.stat().st_size

    try:
        resolved_bin_dir = (bin_dir or (Path.cwd() / "bin")).resolve()
        estimated_tonality, _ = estimate_tonality_by_voting(
            musicxml_path=file_path,
            bin_dir=resolved_bin_dir,
        )
    except (FileNotFoundError, PermissionError) as exc:
        return TonalityAnalysis(
            file_path=file_path,
            file_size=file_size,
            tonality=None,
            accepted=False,
            reason=f"humdrum_setup_error:{type(exc).__name__}",
        )
    except HumdrumError as exc:
        return TonalityAnalysis(
            file_path=file_path,
            file_size=file_size,
            tonality=None,
            accepted=False,
            reason=f"humdrum_error:{type(exc).__name__}",
        )

    declared_tonality = _extract_xml_declared_tonality(file_path)
    if declared_tonality is None:
        return TonalityAnalysis(
            file_path=file_path,
            file_size=file_size,
            tonality=estimated_tonality,
            accepted=True,
            reason="ok_estimated_missing_key_signature",
        )

    if declared_tonality != estimated_tonality:
        return TonalityAnalysis(
            file_path=file_path,
            file_size=file_size,
            tonality=estimated_tonality,
            accepted=True,
            reason="ok_estimated_key_signature_mismatch",
        )

    return TonalityAnalysis(
        file_path=file_path,
        file_size=file_size,
        tonality=estimated_tonality,
        accepted=True,
        reason="ok",
    )


def _extract_xml_declared_tonality(file_path: Path) -> str | None:
    try:
        root = _load_musicxml_root(file_path)
    except (ET.ParseError, zipfile.BadZipFile, KeyError):
        return None

    fifths_text = _find_first_tag_text(root, "fifths")
    if fifths_text is None:
        return None

    mode_text = _find_first_tag_text(root, "mode") or "major"
    mode = mode_text.strip().lower()
    if mode not in {"major", "minor"}:
        return None

    try:
        fifths = int(fifths_text.strip())
    except ValueError:
        return None

    major_map = {
        -7: "Cb",
        -6: "Gb",
        -5: "Db",
        -4: "Ab",
        -3: "Eb",
        -2: "Bb",
        -1: "F",
        0: "C",
        1: "G",
        2: "D",
        3: "A",
        4: "E",
        5: "B",
        6: "F#",
        7: "C#",
    }
    minor_map = {
        -7: "Ab",
        -6: "Eb",
        -5: "Bb",
        -4: "F",
        -3: "C",
        -2: "G",
        -1: "D",
        0: "A",
        1: "E",
        2: "B",
        3: "F#",
        4: "C#",
        5: "G#",
        6: "D#",
        7: "A#",
    }

    tonic = major_map.get(fifths) if mode == "major" else minor_map.get(fifths)
    if tonic is None:
        return None
    pitch_map = {"Cb": 11, "Db": 1, "Gb": 6, "E#": 5, "Fb": 4}
    pitch_names = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    explicit = {
        "C": 0,
        "C#": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
    }
    pc = explicit.get(tonic)
    if pc is None:
        pc = pitch_map.get(tonic)
    if pc is None:
        return None
    return f"{pitch_names[pc]}_{mode}"


def _find_first_tag_text(root: ET.Element, local_name: str) -> str | None:
    for element in root.iter():
        if element.tag.endswith(local_name) and element.text:
            return element.text
    return None


def _load_musicxml_root(file_path: Path) -> ET.Element:
    if file_path.suffix.lower() != ".mxl":
        return ET.parse(file_path).getroot()

    with zipfile.ZipFile(file_path, "r") as archive:
        xml_members = [
            name
            for name in archive.namelist()
            if not name.endswith("/")
            and Path(name).suffix.lower() in {".xml", ".musicxml"}
            and not name.lower().startswith("meta-inf/")
        ]
        if not xml_members:
            raise KeyError("No xml member in mxl")
        selected = sorted(
            xml_members,
            key=lambda item: (
                0 if item.lower().endswith(".musicxml") else 1,
                len(item),
                item.lower(),
            ),
        )[0]
        payload = archive.read(selected)
    return ET.fromstring(payload)


def select_balanced_tonality_subset(
    files: Iterable[Path],
    target_per_tonality: int,
    analyzer: Callable[[Path], TonalityAnalysis] = analyze_musicxml_tonality,
    tonalities: List[str] | None = None,
    workers: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[Dict[str, List[Path]], List[TonalityAnalysis]]:
    if target_per_tonality <= 0:
        raise ValueError("target_per_tonality must be > 0")
    if workers <= 0:
        raise ValueError("workers must be > 0")

    sorted_files = sorted(files, key=lambda path: path.stat().st_size)
    analyses = _analyze_files(
        sorted_files,
        analyzer=analyzer,
        workers=workers,
        progress_callback=progress_callback,
    )

    if tonalities is None:
        available_counts: Dict[str, int] = {}
        for analysis in analyses:
            if not analysis.accepted or analysis.tonality is None:
                continue
            available_counts[analysis.tonality] = (
                available_counts.get(analysis.tonality, 0) + 1
            )
        target_tonalities = sorted(
            [
                tonality
                for tonality, count in available_counts.items()
                if count >= target_per_tonality
            ]
        )
        if not target_tonalities:
            raise ValueError(
                "No tonalities have enough validated songs for requested balancing."
            )
    else:
        target_tonalities = tonalities

    grouped: Dict[str, List[Path]] = {tonality: [] for tonality in target_tonalities}

    for analysis in analyses:
        if not analysis.accepted or analysis.tonality not in grouped:
            continue
        if len(grouped[analysis.tonality]) >= target_per_tonality:
            continue
        grouped[analysis.tonality].append(analysis.file_path)

        if all(len(entries) >= target_per_tonality for entries in grouped.values()):
            break

    shortages = {
        tonality: target_per_tonality - len(paths)
        for tonality, paths in grouped.items()
        if len(paths) < target_per_tonality
    }
    if shortages:
        details = ", ".join(
            f"{tonality}:{missing}" for tonality, missing in shortages.items()
        )
        raise ValueError(
            f"Insufficient validated songs for requested balancing: {details}"
        )

    return grouped, analyses


def _analyze_files(
    files: List[Path],
    *,
    analyzer: Callable[[Path], TonalityAnalysis],
    workers: int,
    progress_callback: Callable[[int, int], None] | None,
) -> List[TonalityAnalysis]:
    total = len(files)
    if total == 0:
        return []

    analyses_buffer: List[TonalityAnalysis | None] = [None for _ in range(total)]

    completed = 0

    def _tick() -> None:
        nonlocal completed
        completed += 1
        if progress_callback is not None:
            progress_callback(completed, total)

    def _safe_analyze(path: Path) -> TonalityAnalysis:
        try:
            return analyzer(path)
        except Exception as exc:  # pragma: no cover - defensive hardening
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            return TonalityAnalysis(
                file_path=path,
                file_size=size,
                tonality=None,
                accepted=False,
                reason=f"analyzer_exception:{type(exc).__name__}",
            )

    if workers == 1:
        for idx, file_path in enumerate(files):
            analyses_buffer[idx] = _safe_analyze(file_path)
            _tick()
        return [item for item in analyses_buffer if item is not None]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(_safe_analyze, file_path): idx
            for idx, file_path in enumerate(files)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            analyses_buffer[idx] = future.result()
            _tick()

    return [item for item in analyses_buffer if item is not None]
