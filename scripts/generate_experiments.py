from __future__ import annotations

import argparse
import random
from pathlib import Path


NOTE_SPECS = [
    ("c", "C"),
    ("cs", "C#"),
    ("d", "D"),
    ("ds", "D#"),
    ("e", "E"),
    ("f", "F"),
    ("fs", "F#"),
    ("g", "G"),
    ("gs", "G#"),
    ("a", "A"),
    ("as", "A#"),
    ("b", "B"),
]

OCTAVES = [1, 2, 3, 4, 5, 6]


def build_query_block(
    rng: random.Random,
    note_name: str,
    length: int,
    octave_min: int,
    octave_max: int,
) -> str:
    tokens = [
        f"{note_name}{rng.randint(octave_min, octave_max)}" for _ in range(length)
    ]
    return " ".join(tokens)


def render_yaml(
    *,
    experiment_name: str,
    output_root: str,
    melody_length: int,
    octave_min: int,
    octave_max: int,
    seed: int,
) -> str:
    rng = random.Random(seed)

    lines: list[str] = [
        "experiment:",
        f'  name: "{experiment_name}"',
        f'  output_root: "{output_root}"',
        "  models:",
        '    - "aria_embedding"',
        '    - "phrase_vae"',
        "",
        "models:",
        "  aria_embedding:",
        '    type: "aria_embedding"',
        '    repo_id: "loubb/aria-medium-embedding"',
        '    checkpoint_filename: "model.safetensors"',
        '    model_config_name: "medium-emb"',
        '    device: "cpu"',
        "  phrase_vae:",
        '    type: "phrase_vae"',
        '    repo_id: "LongshenOu/phrase-vae"',
        '    device: "cpu"',
        "    trim: 16",
        "",
        "queries:",
    ]

    for note_id, note_name in NOTE_SPECS:
        melody = build_query_block(
            rng=rng,
            note_name=note_name,
            length=melody_length,
            octave_min=octave_min,
            octave_max=octave_max,
        )
        lines.extend(
            [
                f'  - id: "{note_id}"',
                f'    name: "{note_name}"',
                f'    custom: "{melody}"',
            ]
        )

    lines.extend(
        [
            "",
            "axes:",
            '  - name: "C_vs_Gb"',
            '    query_a: "c"',
            '    query_b: "fs"',
            '    method: "explicit"',
            '  - name: "A_vs_Eb"',
            '    query_a: "a"',
            '    query_b: "ds"',
            '    method: "explicit"',
            "",
            "plots:",
            "  note_circle:",
            "    axes:",
            '      - "C_vs_Gb"',
            '      - "A_vs_Eb"',
            "    options:",
            "      orthonormalize: false",
            "      pca: false",
            "  note_circle_orthonormal:",
            "    axes:",
            '      - "C_vs_Gb"',
            '      - "A_vs_Eb"',
            "    options:",
            "      orthonormalize: true",
            '      origin_strategy: "mean"',
            "      pca: false",
            "  note_circle_pca:",
            "    axes:",
            '      - "C_vs_Gb"',
            '      - "A_vs_Eb"',
            "    options:",
            "      orthonormalize: false",
            "      pca: true",
            "",
        ]
    )

    return "\n".join(lines)


def render_note_grid_yaml(*, experiment_name: str, output_root: str) -> str:
    lines: list[str] = [
        "experiment:",
        f'  name: "{experiment_name}"',
        f'  output_root: "{output_root}"',
        "  models:",
        '    - "aria_embedding"',
        '    - "phrase_vae"',
        "",
        "models:",
        "  aria_embedding:",
        '    type: "aria_embedding"',
        '    repo_id: "loubb/aria-medium-embedding"',
        '    checkpoint_filename: "model.safetensors"',
        '    model_config_name: "medium-emb"',
        '    device: "cpu"',
        "  phrase_vae:",
        '    type: "phrase_vae"',
        '    repo_id: "LongshenOu/phrase-vae"',
        '    device: "cpu"',
        "    trim: 16",
        "",
        "queries:",
    ]

    for octave in OCTAVES:
        for note_id, note_name in NOTE_SPECS:
            query_id = f"{note_id}{octave}"
            note_label = f"{note_name}{octave}"
            lines.extend(
                [
                    f'  - id: "{query_id}"',
                    f'    name: "{note_label}"',
                    f'    custom: "{note_label}"',
                ]
            )

    lines.extend(
        [
            "",
            "axes:",
            '  - name: "C1_vs_B6"',
            '    query_a: "c1"',
            '    query_b: "b6"',
            '    method: "explicit"',
            "",
            "plots:",
            "  c1_b6:",
            "    axes:",
            '      - "C1_vs_B6"',
            "    options:",
            "      orthonormalize: false",
            "      pca: false",
            "  c1_b6_pca:",
            "    axes:",
            '      - "C1_vs_B6"',
            "    options:",
            "      orthonormalize: false",
            "      pca: true",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate note-melody experiment config with random octaves."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/melodies.yaml"),
        help="Output YAML path.",
    )
    parser.add_argument(
        "--output-note-grid",
        type=Path,
        default=Path("configs/notes_c1_b6.yaml"),
        help="Output YAML path for C1..B6 single-note experiment.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--melody-length",
        type=int,
        default=20,
        help="Number of notes per melody.",
    )
    parser.add_argument(
        "--octave-min",
        type=int,
        default=1,
        help="Minimum octave (inclusive).",
    )
    parser.add_argument(
        "--octave-max",
        type=int,
        default=6,
        help="Maximum octave (inclusive).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="Test",
        help="experiment.name value.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="../outputs",
        help="experiment.output_root value.",
    )
    args = parser.parse_args()

    if args.melody_length <= 0:
        raise ValueError("--melody-length must be > 0")
    if args.octave_min > args.octave_max:
        raise ValueError("--octave-min must be <= --octave-max")

    payload = render_yaml(
        experiment_name=args.experiment_name,
        output_root=args.output_root,
        melody_length=args.melody_length,
        octave_min=args.octave_min,
        octave_max=args.octave_max,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(f"Wrote experiment config: {args.output}")

    note_grid_payload = render_note_grid_yaml(
        experiment_name=args.experiment_name,
        output_root=args.output_root,
    )
    args.output_note_grid.parent.mkdir(parents=True, exist_ok=True)
    args.output_note_grid.write_text(note_grid_payload, encoding="utf-8")
    print(f"Wrote note-grid config: {args.output_note_grid}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
