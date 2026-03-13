from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import subprocess
import tempfile
from urllib.error import URLError
from urllib.request import urlopen
import zipfile

from music_latents.humdrum_types import HUMDRUM_LINUX_BIN_BASE_URL, HumdrumError


def resolve_binary(bin_dir: Path, name: str) -> Path:
    candidate = (bin_dir / name).resolve()
    if not candidate.exists():
        download_humdrum_binary(name=name, destination=candidate)
    if not candidate.is_file():
        raise FileNotFoundError(f"Humdrum binary path is not a file: {candidate}")
    if not candidate.stat().st_mode & 0o111:
        candidate.chmod(candidate.stat().st_mode | 0o755)
    return candidate


def download_humdrum_binary(name: str, destination: Path) -> None:
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


def run_command(
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
def as_uncompressed_musicxml(musicxml_path: Path):
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

    temp_suffix = Path(selected).suffix or ".xml"
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=temp_suffix, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(payload)

    try:
        yield temp_path
    finally:
        temp_path.unlink(missing_ok=True)
