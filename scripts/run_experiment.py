from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from music_latents import ExperimentRunner


def _collect_default_configs() -> list[Path]:
    configs_dir = Path(__file__).resolve().parents[1] / "configs"
    candidates = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))
    filtered = [
        path for path in candidates if path.name not in {"example.yaml", "example.yml"}
    ]
    return sorted(filtered)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a symbolic music latent experiment."
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        help="Path to experiment YAML config. If omitted, runs all configs/*.ya?ml except example.",
    )
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

    if args.config is not None:
        print(f"Running config: {args.config}")
        runner = ExperimentRunner(
            args.config,
            reuse_results_dir=args.reuse_results_dir,
        )
        output_dir = runner.run()
        print(f"Experiment completed: {output_dir}")
        return 0

    if args.reuse_results_dir is not None:
        raise ValueError("--reuse-results-dir requires a specific config path.")

    config_paths = _collect_default_configs()
    if not config_paths:
        raise FileNotFoundError("No config files found in configs/.")

    print(f"Running {len(config_paths)} configs from configs/")

    for config_path in config_paths:
        print(f"Running config: {config_path}")
        runner = ExperimentRunner(config_path)
        output_dir = runner.run()
        print(f"Experiment completed ({config_path.name}): {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
