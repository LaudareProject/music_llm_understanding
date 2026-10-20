from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments import settings
from experiments.common.comparison_report import generate_comparison_reports
from experiments.common.feature_backend import EMBEDDING_BACKENDS


def main():
    parser = argparse.ArgumentParser(
        description="Generate complete per-experiment embedding comparisons."
    )
    parser.add_argument(
        "--backend",
        choices=EMBEDDING_BACKENDS,
        nargs="+",
        default=list(EMBEDDING_BACKENDS),
        help="Backends that must have completed runs (default: all embedding backends).",
    )
    parser.add_argument(
        "--experiment",
        choices=tuple(settings.EXPERIMENTS),
        nargs="*",
        default=list(settings.EXPERIMENTS),
        help="Experiments to report (default: all configured experiments).",
    )
    args = parser.parse_args()
    output_dir = generate_comparison_reports(
        REPO_ROOT, experiments=args.experiment, backends=args.backend
    )
    print(output_dir)


if __name__ == "__main__":
    main()
