from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from music_latents import ExperimentRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a symbolic music latent experiment."
    )
    parser.add_argument("config", type=Path, help="Path to experiment YAML config.")
    parser.add_argument(
        "--reuse-results-dir",
        type=Path,
        default=None,
        help=(
            "Reuse existing latent embeddings from this experiment directory and "
            "write new axis outputs there."
        ),
    )
    args = parser.parse_args()

    runner = ExperimentRunner(
        args.config,
        reuse_results_dir=args.reuse_results_dir,
    )
    output_dir = runner.run()
    print(f"Experiment completed: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
