from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from music_latents import ExperimentRunner
from music_latents.config_models import QuerySpec
from music_latents.inputs import prepare_midi_input
from music_latents.models import create_model_client


def _build_leace_note_class_samples() -> list[tuple[str, str, int, str]]:
    pitch_class_names = [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    ]
    samples: list[tuple[str, str, int, str]] = []
    for midi_value in range(24, 96):
        pitch_class = pitch_class_names[midi_value % 12]
        octave = midi_value // 12 - 1
        name = f"{pitch_class}{octave}"
        query_id = f"leace_{pitch_class.lower().replace('#', 's')}{octave}"
        samples.append((query_id, name, midi_value, str(octave)))
    return samples


def _fit_leace_pitch_class_erasers(
    run_config,
    config_tag: str,
) -> dict[str, Path]:

    try:
        import torch
        from concept_erasure import LeaceEraser
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LEACE support requires concept-erasure and torch. Install with: uv add concept-erasure"
        ) from exc

    model_names = run_config.settings.models or list(run_config.models.keys())
    samples = _build_leace_note_class_samples()
    labels = np.asarray([sample[3] for sample in samples], dtype=object)

    classes, inverse = np.unique(labels, return_inverse=True)
    if classes.shape[0] < 2:
        raise ValueError("LEACE fitting requires at least two octave classes.")

    model_clients = {}
    for model_name in model_names:
        model_cfg = run_config.models[model_name]
        model_type = model_cfg.get("type")
        if not model_type:
            raise ValueError(f"Model '{model_name}' is missing required field 'type'.")
        model_clients[model_name] = create_model_client(
            model_type=model_type,
            name=model_name,
            config={k: v for k, v in model_cfg.items() if k not in {"type", "trim"}},
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eraser_root = (Path("outputs") / ".leace" / timestamp / config_tag).resolve()
    eraser_root.mkdir(parents=True, exist_ok=True)

    erasers_by_model: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="leace_note_class_") as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        midi_paths: dict[str, Path] = {}
        for query_id, name, midi_value, _ in samples:
            query = QuerySpec(
                query_id=query_id,
                name=name,
                color=None,
                target=None,
                input_kind="value",
                value=midi_value,
            )
            query_dir = tmp_dir / query_id
            query_dir.mkdir(parents=True, exist_ok=True)
            input_info = prepare_midi_input(
                query=query,
                base_dir=tmp_dir,
                query_dir=query_dir,
            )
            midi_paths[query_id] = input_info.midi_path

        for model_name in model_names:
            rows: list[np.ndarray] = []
            model_client = model_clients[model_name]
            for query_id, _, _, _ in samples:
                result = model_client.encode_midi(midi_paths[query_id])
                rows.append(np.asarray(result.latent, dtype=np.float64).reshape(-1))

            X = np.stack(rows, axis=0)
            y = np.asarray(inverse, dtype=np.int64)
            z = np.eye(classes.shape[0], dtype=np.float64)[y]

            x_tensor = torch.tensor(X, dtype=torch.float64)
            z_tensor = torch.tensor(z, dtype=torch.float64)
            eraser = LeaceEraser.fit(x_tensor, z_tensor)

            proj_left = np.asarray(
                eraser.proj_left.detach().cpu().numpy(), dtype=np.float64
            )
            proj_right = np.asarray(
                eraser.proj_right.detach().cpu().numpy(), dtype=np.float64
            )
            bias = (
                None
                if eraser.bias is None
                else np.asarray(eraser.bias.detach().cpu().numpy(), dtype=np.float64)
            )

            eraser_dir = eraser_root / model_name
            eraser_dir.mkdir(parents=True, exist_ok=True)
            eraser_path = eraser_dir / "leace_eraser.npz"
            np.savez(
                eraser_path,
                proj_left=proj_left,
                proj_right=proj_right,
                bias=bias,
                classes=classes,
            )
            erasers_by_model[model_name] = eraser_path

    print(
        "LEACE erasers saved: "
        + ", ".join([f"{name}={path}" for name, path in erasers_by_model.items()])
    )
    return erasers_by_model


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

    for model_dir in experiment_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name in {
            "queries",
            "axes",
            "supervised",
        }:
            continue

        plots_root = model_dir / "plots"
        json_root = model_dir / "json"
        if not plots_root.exists() and not json_root.exists():
            continue

        model_name = model_dir.name
        batch_plots_dir = batch_dir / model_name / "plots"
        batch_json_dir = batch_dir / model_name / "json"
        batch_plots_dir.mkdir(parents=True, exist_ok=True)
        batch_json_dir.mkdir(parents=True, exist_ok=True)

        if plots_root.exists():
            for source in plots_root.rglob("*"):
                if not source.is_file():
                    continue
                rel = source.relative_to(plots_root).as_posix().replace("/", "__")
                target = batch_plots_dir / f"{config_name}__{rel}"
                shutil.copy2(source, target)

        if json_root.exists():
            for source in json_root.rglob("*"):
                if not source.is_file():
                    continue
                rel = source.relative_to(json_root).as_posix().replace("/", "__")
                target = batch_json_dir / f"{config_name}__{rel}"
                shutil.copy2(source, target)

    axes_root = experiment_dir / "axes"
    if axes_root.exists():
        for model_dir in axes_root.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            batch_json_dir = batch_dir / model_name / "json"
            batch_plots_dir = batch_dir / model_name / "plots"
            batch_json_dir.mkdir(parents=True, exist_ok=True)
            batch_plots_dir.mkdir(parents=True, exist_ok=True)

            for source in model_dir.iterdir():
                if not source.is_file():
                    continue
                if source.suffix.lower() == ".png":
                    target = batch_plots_dir / f"{config_name}__{source.name}"
                else:
                    target = batch_json_dir / f"{config_name}__{source.name}"
                shutil.copy2(source, target)

    supervised_root = experiment_dir / "supervised"
    if supervised_root.exists():
        for model_dir in supervised_root.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            batch_supervised_dir = batch_dir / model_name / "supervised"
            batch_supervised_dir.mkdir(parents=True, exist_ok=True)

            for source in model_dir.rglob("*"):
                if not source.is_file():
                    continue
                rel = source.relative_to(model_dir).as_posix().replace("/", "__")
                target = batch_supervised_dir / f"{config_name}__{rel}"
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
    parser.add_argument(
        "--center-and-whiten",
        action="store_true",
        help="Enable global centering+whitening for all plots in this run.",
    )
    parser.add_argument(
        "--orthogonal-projection-to",
        nargs="+",
        default=None,
        help=(
            "Model-specific projection source as MODEL=PATH (repeatable). "
            "You may also pass a single PATH to apply to all models."
        ),
    )
    parser.add_argument(
        "--fit-leace-octave",
        "--fit-leace-note-class",
        dest="fit_leace_octave",
        action="store_true",
        help=(
            "Fit LEACE erasers using fixed C1..B6 note values with octave targets, "
            "then apply erasure to run(s)."
        ),
    )
    args = parser.parse_args()

    projection_spec: Path | dict[str, Path] | None = None
    projection_args: list[str] = list(args.orthogonal_projection_to or [])

    if projection_args:
        parsed: dict[str, Path] = {}
        for raw in projection_args:
            if "=" in raw:
                model_name, path_raw = raw.split("=", 1)
                model_name = model_name.strip()
                path_raw = path_raw.strip()
                if not model_name:
                    raise ValueError(
                        "--orthogonal-projection-to MODEL=PATH requires a non-empty model name."
                    )
                if not path_raw:
                    raise ValueError(
                        "--orthogonal-projection-to MODEL=PATH requires a non-empty path."
                    )
                parsed[model_name] = Path(path_raw)
            else:
                if len(projection_args) > 1:
                    raise ValueError(
                        "When passing multiple --orthogonal-projection-to values, use MODEL=PATH format for each."
                    )
                projection_spec = Path(raw)
        if projection_spec is None:
            projection_spec = parsed

    if args.config is not None:
        print(f"Running config: {args.config}")
        runner = ExperimentRunner(
            args.config,
            reuse_results_dir=args.reuse_results_dir,
            center_and_whiten_override=args.center_and_whiten,
            orthogonal_projection_to=projection_spec,
            leace_erase_with=None,
        )
        if args.fit_leace_octave:
            leace_erasers = _fit_leace_pitch_class_erasers(
                run_config=runner.config,
                config_tag=Path(args.config).stem,
            )
            (
                runner.leace_erasers,
                runner.leace_eraser_sources,
            ) = runner._load_leace_erasers(leace_erasers)
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
            center_and_whiten_override=args.center_and_whiten,
            orthogonal_projection_to=projection_spec,
            leace_erase_with=None,
        )
        if args.fit_leace_octave:
            leace_erasers = _fit_leace_pitch_class_erasers(
                run_config=runner.config,
                config_tag=config_path.stem,
            )
            (
                runner.leace_erasers,
                runner.leace_eraser_sources,
            ) = runner._load_leace_erasers(leace_erasers)
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
