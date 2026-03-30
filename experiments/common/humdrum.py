from contextlib import contextmanager
from pathlib import Path
import subprocess
import tempfile
from urllib.request import urlopen
import zipfile


HUMDRUM_LINUX_BIN_BASE_URL = "https://extras.humdrum.org/bin/linux"


def resolve_binary(bin_dir, name):
    candidate = (bin_dir / name).resolve()
    if not candidate.exists():
        candidate.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(f"{HUMDRUM_LINUX_BIN_BASE_URL}/{name}", timeout=30) as response:
            candidate.write_bytes(response.read())
        candidate.chmod(candidate.stat().st_mode | 0o755)
    return candidate


def run_command(cmd, cwd=None, stdin_text=None):
    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        input=stdin_text,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout


@contextmanager
def as_uncompressed_musicxml(path):
    if path.suffix.lower() != ".mxl":
        yield path
        return

    with zipfile.ZipFile(path, "r") as archive:
        members = [
            name
            for name in archive.namelist()
            if not name.endswith("/")
            and Path(name).suffix.lower() in {".xml", ".musicxml"}
            and not name.lower().startswith("meta-inf/")
        ]
        selected = sorted(
            members,
            key=lambda item: (
                0 if item.lower().endswith(".musicxml") else 1,
                len(item),
                item.lower(),
            ),
        )[0]
        payload = archive.read(selected)

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=Path(selected).suffix or ".xml", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(payload)
    try:
        yield temp_path
    finally:
        temp_path.unlink(missing_ok=True)


def _musicxml_to_midi_partitura(musicxml_path, midi_path):
    import partitura

    score = partitura.load_score(str(musicxml_path))
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    partitura.save_score_midi(score, str(midi_path))


def _musicxml_to_midi_music21(musicxml_path, midi_path):
    from music21 import converter

    score = converter.parse(str(musicxml_path))
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    score.write("midi", fp=str(midi_path))


def musicxml_to_midi(musicxml_path, midi_path, bin_dir):
    xml2hum = resolve_binary(bin_dir, "xml2hum")
    hum2mid = resolve_binary(bin_dir, "hum2mid")
    try:
        with as_uncompressed_musicxml(musicxml_path) as input_musicxml:
            hum_text = run_command(
                [str(xml2hum), str(input_musicxml)], cwd=input_musicxml.parent
            )
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".krn", delete=False
        ) as handle:
            temp_hum = Path(handle.name)
            handle.write(hum_text)
        try:
            run_command(
                [str(hum2mid), str(temp_hum), "-o", str(midi_path)],
                cwd=midi_path.parent,
            )
        finally:
            temp_hum.unlink(missing_ok=True)
        return
    except Exception:
        pass
    try:
        print("Falling back to partitura")
        _musicxml_to_midi_partitura(musicxml_path, midi_path)
        return
    except Exception:
        pass
    print("Falling back to musc21")
    _musicxml_to_midi_music21(musicxml_path, midi_path)


def _kern_to_midi_partitura(kern_path, midi_path):
    import partitura

    score = partitura.load_kern(str(kern_path))
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    partitura.save_score_midi(score, str(midi_path))


def _kern_to_midi_music21(kern_path, midi_path):
    from music21 import converter

    score = converter.parse(str(kern_path))
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    score.write("midi", fp=str(midi_path))


def kern_to_midi(kern_path, midi_path, bin_dir):
    hum2mid = resolve_binary(bin_dir, "hum2mid")
    try:
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        run_command(
            [str(hum2mid), str(kern_path), "-o", str(midi_path)],
            cwd=midi_path.parent,
        )
        return
    except Exception:
        pass
    try:
        print("Falling back to partitura")
        _kern_to_midi_partitura(kern_path, midi_path)
        return
    except Exception:
        pass
    print("Falling back to musc21")
    _kern_to_midi_music21(kern_path, midi_path)
