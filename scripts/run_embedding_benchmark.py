from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments import settings
from experiments.common.feature_backend import EMBEDDING_BACKENDS, FEATURE_BACKENDS
from experiments.common.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Run the configured experiment matrix for feature backends."
    )
    parser.add_argument(
        "--backend",
        choices=FEATURE_BACKENDS,
        nargs="+",
        default=list(EMBEDDING_BACKENDS),
        help="Backends to run (default: all embedding backends; musif can be selected alone).",
    )
    parser.add_argument(
        "--experiment",
        choices=tuple(settings.EXPERIMENTS),
        nargs="*",
        default=list(settings.EXPERIMENTS),
        help="Experiments to run (default: every settings.EXPERIMENTS entry).",
    )
    args = parser.parse_args()

    failures = []
    for backend in args.backend:
        for experiment_name in args.experiment:
            print(f"\n=== {experiment_name} | backend={backend} ===", flush=True)
            try:
                output_dir = run_experiment(
                    settings, experiment_name, feature_backend=backend
                )
                print(f"Completed: {output_dir}", flush=True)
            except Exception as exc:
                failures.append((experiment_name, backend, exc))
                print(f"Failed: {experiment_name} | {backend}: {exc}", flush=True)

    if failures:
        details = "\n".join(
            f"  {name} / {backend}: {error}" for name, backend, error in failures
        )
        raise RuntimeError(
            f"{len(failures)} experiment/backend pair(s) failed; no complete matrix was produced:\n{details}"
        )


if __name__ == "__main__":
    main()
