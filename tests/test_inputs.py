from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from music_latents.config import QuerySpec
from music_latents.humdrum import HumdrumError
from music_latents.inputs import prepare_midi_input


def test_prepare_midi_input_custom_uses_music21(tmp_path: Path) -> None:
    query = QuerySpec(
        query_id="q_custom",
        name="Custom",
        color=None,
        input_kind="custom",
        value="C4-E4-G4",
    )

    query_dir = tmp_path / "q"
    info = prepare_midi_input(query=query, base_dir=tmp_path, query_dir=query_dir)
    assert info.input_kind == "custom"
    assert info.midi_path.exists()


def test_prepare_midi_input_musicxml_falls_back_to_music21_when_humdrum_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    xml_path = tmp_path / "x.musicxml"
    xml_path.write_text(
        """
<score-partwise>
  <part-list>
    <score-part id="P1"><part-name>Music</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>1</duration>
        <type>quarter</type>
      </note>
    </measure>
  </part>
</score-partwise>
""".strip(),
        encoding="utf-8",
    )

    query = QuerySpec(
        query_id="q_xml",
        name="XML",
        color=None,
        input_kind="musicxml_path",
        value="x.musicxml",
    )

    def _fail(*_args, **_kwargs):
        raise HumdrumError("humdrum failure")

    monkeypatch.setattr("music_latents.inputs.convert_musicxml_to_centered_midi", _fail)

    query_dir = tmp_path / "qx"
    info = prepare_midi_input(query=query, base_dir=tmp_path, query_dir=query_dir)
    assert info.input_kind == "musicxml_path"
    assert info.midi_path.exists()
