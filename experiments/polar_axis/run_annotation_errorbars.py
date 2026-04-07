from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common.folder_handling import create_folder
from experiments.common.tsv import read_tsv


REPO_ROOT = Path(__file__).resolve().parents[2]
SONGS_TSV = REPO_ROOT / "experiments" / "polar_axis" / "songs.tsv"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments" / "polar_axis"


def as_float(values):
    return np.asarray([float(v) for v in values], dtype=np.float64)


def save_errorbar(rows, mean_key, std_key, title, out_path):
    ordered = sorted(rows, key=lambda row: float(row[mean_key]))
    x = np.arange(len(ordered), dtype=np.float64)
    y = as_float([row[mean_key] for row in ordered])
    yerr = as_float([row[std_key] for row in ordered])
    labels = [row["title"] for row in ordered]

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=2, linewidth=1, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Piece index (sorted)")
    ax.set_ylabel("Mean across annotators")
    ax.grid(True, alpha=0.25)

    if len(labels) <= 80:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
    else:
        ax.set_xticks(np.linspace(0, len(labels) - 1, 20, dtype=int))
        ax.set_xticklabels([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    rows = read_tsv(SONGS_TSV)
    output_dir = create_folder(OUTPUT_ROOT, {"plot": "annotation_errorbars"})

    save_errorbar(
        rows,
        "cosine",
        "cosine_std",
        "VGMIDI piece cosine distance with annotator spread",
        output_dir / "cosine_errorbars.png",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
