from __future__ import annotations

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from music_latents.humdrum import (
    HUMDRUM_LINUX_BIN_BASE_URL,
    HumdrumError,
    KeyCorrelationResult,
    _as_uncompressed_musicxml,
    _resolve_binary,
    compute_semitone_shift,
    convert_musicxml_to_centered_midi,
    estimate_average_midi_pitch_from_musicxml,
    ensemble_vote,
    parse_keycor_scores,
)


def test_parse_keycor_scores_extracts_major_minor_rows() -> None:
    output = """
C       Major 0.35
D-flat  Major 0.25
F       Minor 0.79
G-sharp Minor -0.22
"""
    rows = parse_keycor_scores(output)
    assert ("C_major", 0.35) in rows
    assert ("C#_major", 0.25) in rows
    assert ("F_minor", 0.79) in rows
    assert ("Ab_minor", -0.22) in rows


def test_parse_keycor_scores_extracts_indexed_format_rows() -> None:
    output = """
The best key is: F Major
Major[0] = 0.657087\t\t\tMinor[0] = 0.46381
Major[5] = 0.834356\t\t\tMinor[10] = 0.348975
"""
    rows = parse_keycor_scores(output)
    assert ("C_major", 0.657087) in rows
    assert ("C_minor", 0.46381) in rows
    assert ("F_major", 0.834356) in rows
    assert ("Bb_minor", 0.348975) in rows


def test_ensemble_vote_uses_score_tiebreak() -> None:
    results = [
        KeyCorrelationResult(profile="kk", tonality="C_major", score=0.60),
        KeyCorrelationResult(profile="simple", tonality="G_major", score=0.70),
        KeyCorrelationResult(profile="aarden", tonality="C_major", score=0.45),
        KeyCorrelationResult(profile="bellman", tonality="G_major", score=0.90),
    ]
    assert ensemble_vote(results) == "G_major"


def test_resolve_binary_downloads_when_missing(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def read(self) -> bytes:
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def _fake_urlopen(url: str, timeout: int):
        assert url == f"{HUMDRUM_LINUX_BIN_BASE_URL}/xml2hum"
        assert timeout == 30
        return _FakeResponse(b"#!/bin/sh\n")

    monkeypatch.setattr("music_latents.humdrum.urlopen", _fake_urlopen)

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_dir = Path(tmpdir)
        binary = _resolve_binary(bin_dir, "xml2hum")
        assert binary.exists()
        assert binary.is_file()
        assert binary.stat().st_mode & 0o111


def test_compute_semitone_shift_rounds_to_nearest_int() -> None:
    assert compute_semitone_shift(average_pitch=58.6, target_pitch=60.0) == 1
    assert compute_semitone_shift(average_pitch=58.4, target_pitch=60.0) == 2
    assert compute_semitone_shift(average_pitch=61.6, target_pitch=60.0) == -2


def test_estimate_average_midi_pitch_from_musicxml_without_music21(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "score.musicxml"
    xml_path.write_text(
        """
<score-partwise>
  <part>
    <measure>
      <note><pitch><step>C</step><octave>4</octave></pitch></note>
      <note><pitch><step>D</step><octave>4</octave></pitch></note>
      <note><rest/></note>
      <note><pitch><step>E</step><alter>1</alter><octave>4</octave></pitch></note>
    </measure>
  </part>
</score-partwise>
""".strip(),
        encoding="utf-8",
    )
    avg = estimate_average_midi_pitch_from_musicxml(xml_path)
    assert avg == (60 + 62 + 65) / 3


def test_as_uncompressed_musicxml_accepts_xml(tmp_path: Path) -> None:
    xml_path = tmp_path / "score.musicxml"
    xml_path.write_text("<score-partwise/>", encoding="utf-8")

    with _as_uncompressed_musicxml(xml_path) as converted:
        assert converted == xml_path


def test_as_uncompressed_musicxml_extracts_mxl(tmp_path: Path) -> None:
    import zipfile

    mxl_path = tmp_path / "score.mxl"
    with zipfile.ZipFile(mxl_path, "w") as archive:
        archive.writestr("META-INF/container.xml", "<container/>")
        archive.writestr(
            "score.musicxml", "<score-partwise><part-list/></score-partwise>"
        )

    with _as_uncompressed_musicxml(mxl_path) as converted:
        assert converted.suffix in {".xml", ".musicxml"}
        assert converted.exists()
        assert converted.read_text(encoding="utf-8").startswith("<score-partwise")


def test_convert_musicxml_to_centered_midi_fallback_when_transpose_fails(
    monkeypatch, tmp_path: Path
) -> None:
    xml_path = tmp_path / "score.musicxml"
    xml_path.write_text("<score-partwise><part/></score-partwise>", encoding="utf-8")
    midi_path = tmp_path / "out.mid"

    monkeypatch.setattr(
        "music_latents.humdrum.estimate_average_midi_pitch_from_musicxml",
        lambda *_args, **_kwargs: 69.0,
    )
    monkeypatch.setattr(
        "music_latents.humdrum._resolve_binary",
        lambda _bin_dir, name: Path(f"/tmp/{name}"),
    )

    def _fake_run(cmd, cwd=None, stdin_text=None):
        exe = Path(cmd[0]).name
        if exe == "xml2hum":
            return "**kern\n4c\n*-\n"
        if exe == "transpose":
            raise HumdrumError("transpose failed")
        if exe == "hum2mid":
            out_path = Path(cmd[cmd.index("-o") + 1])
            out_path.write_bytes(b"MThd")
            return ""
        return ""

    monkeypatch.setattr("music_latents.humdrum._run_command", _fake_run)

    applied_shift = convert_musicxml_to_centered_midi(xml_path, midi_path, tmp_path)
    assert applied_shift == 0
    assert midi_path.exists()
