from __future__ import annotations

from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from music_latents.ewld import TonalityAnalysis, select_balanced_tonality_subset
from scripts.prepare_ewld_tonality_experiment import (
    build_config_payload,
    resolve_dataset_path,
)
from music_latents.ewld import _extract_xml_declared_tonality


def test_select_balanced_tonality_subset_prefers_smallest(tmp_path: Path) -> None:
    small = tmp_path / "small.mxl"
    medium = tmp_path / "medium.mxl"
    large = tmp_path / "large.mxl"
    small.write_bytes(b"1")
    medium.write_bytes(b"22")
    large.write_bytes(b"333")

    mapping = {
        small: TonalityAnalysis(small, 1, "C_major", True, "ok"),
        medium: TonalityAnalysis(medium, 2, "C_major", True, "ok"),
        large: TonalityAnalysis(large, 3, "D_major", True, "ok"),
    }

    grouped, _ = select_balanced_tonality_subset(
        [large, medium, small],
        target_per_tonality=1,
        analyzer=lambda path: mapping[path],
        tonalities=["C_major", "D_major"],
    )

    assert grouped["C_major"] == [small]
    assert grouped["D_major"] == [large]


def test_select_balanced_tonality_subset_raises_when_insufficient(
    tmp_path: Path,
) -> None:
    only = tmp_path / "only.mxl"
    only.write_bytes(b"x")

    grouped_analysis = TonalityAnalysis(only, 1, "C_major", True, "ok")
    try:
        select_balanced_tonality_subset(
            [only],
            target_per_tonality=1,
            analyzer=lambda _: grouped_analysis,
            tonalities=["C_major", "D_major"],
        )
    except ValueError as exc:
        assert "D_major" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing tonality")


def test_find_musicxml_files_ignores_directories(tmp_path: Path) -> None:
    root = tmp_path / "EWLD"
    root.mkdir()
    bogus_dir = root / "a.mxl"
    bogus_dir.mkdir()
    valid_file = root / "b.mxl"
    valid_file.write_bytes(b"dummy")

    from music_latents.ewld import find_musicxml_files

    files = find_musicxml_files(root)
    assert valid_file.resolve() in files
    assert bogus_dir.resolve() not in files


def test_build_config_payload_includes_average_and_median_groups(
    tmp_path: Path,
) -> None:
    c_song = tmp_path / "c_song.mxl"
    d_song = tmp_path / "d_song.mxl"
    c_song.write_bytes(b"c")
    d_song.write_bytes(b"d")

    grouped = {
        "C_major": [c_song],
        "C_minor": [c_song],
        "C#_major": [c_song],
        "C#_minor": [c_song],
        "D_major": [d_song],
        "D_minor": [d_song],
        "Eb_major": [d_song],
        "Eb_minor": [d_song],
        "E_major": [d_song],
        "E_minor": [d_song],
        "F_major": [d_song],
        "F_minor": [d_song],
        "F#_major": [d_song],
        "F#_minor": [d_song],
        "G_major": [d_song],
        "G_minor": [d_song],
        "Ab_major": [d_song],
        "Ab_minor": [d_song],
        "A_major": [d_song],
        "A_minor": [d_song],
        "Bb_major": [d_song],
        "Bb_minor": [d_song],
        "B_major": [d_song],
        "B_minor": [d_song],
    }

    payload = build_config_payload(
        grouped,
        config_parent=tmp_path,
        experiment_name="EWLDTest",
        output_root="../outputs",
    )

    assert "average" in payload
    assert "median" in payload
    assert "avg_c_major" in payload["average"]
    assert "med_c_major" in payload["median"]
    assert payload["plots"]["tonality_averages"]["options"]["group_filter"] == "average"
    assert payload["plots"]["tonality_medians"]["options"]["group_filter"] == "median"
    assert payload["plots"]["tonality_averages"]["options"]["lda_groups"] is True
    assert payload["plots"]["tonality_medians"]["options"]["lda_groups"] is True
    for plot in payload["plots"].values():
        assert plot["options"]["cosine_references"] == ["avg_c_major", "avg_a_major"]
    assert payload["plots"]["tonality_pca_components"]["axes"] == ["pca01", "pca02"]
    axis_names = [axis["name"] for axis in payload["axes"]]
    assert "pca01" in axis_names
    assert "pca02" in axis_names
    assert payload["plots"]["tonality_averages_pca_only"]["options"]["pca"] is True
    assert (
        payload["plots"]["tonality_averages_orthonormalize_only"]["options"][
            "orthonormalize"
        ]
        is True
    )
    assert (
        payload["plots"]["tonality_averages_pca_and_orthonormalize"]["options"]["pca"]
        is True
    )
    assert (
        payload["plots"]["tonality_averages_pca_and_orthonormalize"]["options"][
            "orthonormalize"
        ]
        is True
    )
    c_query = next(item for item in payload["queries"] if item["id"] == "c_major_001")
    d_minor_query = next(
        item for item in payload["queries"] if item["id"] == "d_minor_001"
    )
    assert c_query["color"] == "#1f77b4"
    assert d_minor_query["color"] == "#d62728"
    assert payload["average"]["avg_c_major"]["color"] == "#1f77b4"
    assert payload["median"]["med_c_minor"]["color"] == "#d62728"


def test_resolve_dataset_path_uses_dataset_subdir(tmp_path: Path) -> None:
    dataset_root = tmp_path / "EWLD"
    dataset_dir = dataset_root / "dataset"
    dataset_dir.mkdir(parents=True)

    resolved = resolve_dataset_path(dataset_root)
    assert resolved == dataset_dir.resolve()


def test_extract_declared_tonality_from_mxl(tmp_path: Path) -> None:
    mxl = tmp_path / "x.mxl"
    xml = """<score-partwise><part><measure><attributes><key><fifths>0</fifths><mode>major</mode></key></attributes></measure></part></score-partwise>"""
    with zipfile.ZipFile(mxl, "w") as archive:
        archive.writestr("META-INF/container.xml", "<container/>")
        archive.writestr("score.musicxml", xml)

    assert _extract_xml_declared_tonality(mxl) == "C_major"


def test_analyze_musicxml_accepts_estimate_with_signature_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    from music_latents import ewld as ewld_module
    from music_latents.humdrum import KeyCorrelationResult

    xml_path = tmp_path / "x.musicxml"
    xml_path.write_text(
        "<score-partwise><part><measure><attributes><key><fifths>0</fifths><mode>major</mode></key></attributes></measure></part></score-partwise>",
        encoding="utf-8",
    )

    def _fake_estimate(*args, **kwargs):
        return "G_major", [KeyCorrelationResult("kk", "G_major", 0.9)]

    monkeypatch.setattr(ewld_module, "estimate_tonality_by_voting", _fake_estimate)

    analysis = ewld_module.analyze_musicxml_tonality(xml_path, bin_dir=tmp_path)
    assert analysis.accepted is True
    assert analysis.tonality == "G_major"
    assert analysis.reason == "ok_estimated_key_signature_mismatch"


def test_analyze_musicxml_accepts_estimate_with_missing_signature(
    monkeypatch, tmp_path: Path
) -> None:
    from music_latents import ewld as ewld_module
    from music_latents.humdrum import KeyCorrelationResult

    xml_path = tmp_path / "y.musicxml"
    xml_path.write_text(
        "<score-partwise><part><measure><attributes><divisions>1</divisions></attributes></measure></part></score-partwise>",
        encoding="utf-8",
    )

    def _fake_estimate(*args, **kwargs):
        return "F_major", [KeyCorrelationResult("kk", "F_major", 0.9)]

    monkeypatch.setattr(ewld_module, "estimate_tonality_by_voting", _fake_estimate)

    analysis = ewld_module.analyze_musicxml_tonality(xml_path, bin_dir=tmp_path)
    assert analysis.accepted is True
    assert analysis.tonality == "F_major"
    assert analysis.reason == "ok_estimated_missing_key_signature"
