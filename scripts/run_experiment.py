from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
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


def _collect_batch_outputs(
    config_path: Path, experiment_dir: Path, batch_dir: Path
) -> None:
    config_name = config_path.stem

    queries_root = experiment_dir / "queries"
    if queries_root.exists():
        for query_dir in queries_root.iterdir():
            if not query_dir.is_dir():
                continue
            query_id = query_dir.name
            for model_dir in query_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model_name = model_dir.name
                latents_dir = batch_dir / model_name / "latents"
                latents_dir.mkdir(parents=True, exist_ok=True)

                for filename in ("latent.npy", "metadata.json", "latent.png"):
                    source = model_dir / filename
                    if not source.exists():
                        continue
                    target = latents_dir / f"{config_name}__{query_id}__{filename}"
                    shutil.copy2(source, target)

    axes_root = experiment_dir / "axes"
    if axes_root.exists():
        for model_dir in axes_root.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            scores_dir = batch_dir / model_name / "scores"
            plots_dir = batch_dir / model_name / "plots"
            scores_dir.mkdir(parents=True, exist_ok=True)
            plots_dir.mkdir(parents=True, exist_ok=True)

            for source in model_dir.iterdir():
                if not source.is_file():
                    continue
                if source.suffix.lower() == ".png":
                    target = plots_dir / f"{config_name}__{source.name}"
                else:
                    target = scores_dir / f"{config_name}__{source.name}"
                shutil.copy2(source, target)


def _cleanup_empty_parents(path: Path, stop_at: Path) -> None:
    current = path.resolve()
    stop = stop_at.resolve()
    while current != stop and current.is_dir():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


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

    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = (Path("outputs") / batch_timestamp).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_work_dir = batch_dir / ".work"
    batch_work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(config_paths)} configs from configs/")
    print(f"Aggregated output directory: {batch_dir}")

    for config_path in config_paths:
        print(f"Running config: {config_path}")
        runner = ExperimentRunner(
            config_path,
            output_root_override=batch_work_dir,
            experiment_name_override=config_path.stem,
        )
        output_dir = runner.run()
        _collect_batch_outputs(config_path, output_dir, batch_dir)
        shutil.rmtree(output_dir)
        _cleanup_empty_parents(output_dir.parent, stop_at=batch_work_dir)
        print(f"Experiment completed ({config_path.name}): {output_dir}")

    _cleanup_empty_parents(batch_work_dir, stop_at=batch_dir)

    print(f"Batch completed: {batch_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
