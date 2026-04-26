from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common.tsv import write_tsv


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_TSV = REPO_ROOT / "experiments" / "pitch_axis" / "notes.tsv"


PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
LENGTHS = [
    (60, "eighth"),
    (30, "sixteenth"),
    (960, "two_quarters"),
    (1440, "three_quarters"),
    (1920, "whole"),
]
VELOCITIES = [25, 50, 75, 100, 127]


def note_name(midi_note):
    pitch_class = PITCH_CLASS_NAMES[midi_note % 12]
    octave = midi_note // 12 - 1
    return f"{pitch_class}{octave}"


def main():
    rows = []
    for midi_note in range(21, 108):
        name = note_name(midi_note)
        for duration_ticks, duration_name in LENGTHS:
            for velocity in VELOCITIES:
                rows.append(
                    {
                        "query_id": (
                            f"note_{name.lower().replace('#', 's')}_{duration_name}_v{velocity}"
                        ),
                        "dataset": "pitch",
                        "midi_note": midi_note,
                        "duration_ticks": duration_ticks,
                        "velocity": velocity,
                        "name": name,
                    }
                )
    write_tsv(
        OUTPUT_TSV,
        rows,
        ["query_id", "dataset", "midi_note", "duration_ticks", "velocity", "name"],
    )
    print(f"wrote {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
