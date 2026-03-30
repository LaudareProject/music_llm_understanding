from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common.tsv import write_tsv


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_TSV = REPO_ROOT / "experiments" / "pitch_axis" / "notes.tsv"


PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(midi_note):
    pitch_class = PITCH_CLASS_NAMES[midi_note % 12]
    octave = midi_note // 12 - 1
    return f"{pitch_class}{octave}"


def main():
    rows = []
    for midi_note in range(24, 96):
        name = note_name(midi_note)
        axis_side = "."
        if midi_note == 24:
            axis_side = "a"
        if midi_note == 95:
            axis_side = "b"
        rows.append(
            {
                "query_id": f"note_{name.lower().replace('#', 's')}",
                "midi_note": midi_note,
                "name": name,
                "axis_side": axis_side,
            }
        )
    write_tsv(OUTPUT_TSV, rows, ["query_id", "midi_note", "name", "axis_side"])
    print(f"wrote {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
