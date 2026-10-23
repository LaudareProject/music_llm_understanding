from __future__ import annotations

import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap

import numpy as np

from experiments.common.feature_backend import (
    BACKEND_OUTPUT_DIRS,
    EMBEDDING_BACKENDS,
    FEATURE_BACKENDS,
)
from experiments.common.runner_search import anchor_method_label
from experiments.common.tsv import read_tsv, write_tsv
from experiments.settings import EXPERIMENTS


REPORT_COLUMNS = (
    "backend",
    "metric",
    "value",
    "ci95",
    "folds",
    "support",
    "output_dir",
)
COVERAGE_COLUMNS = (
    "experiment",
    "backend",
    "artifact",
    "status",
    "output_dir",
)
PLOT_ARTIFACTS = (
    "projections.tsv",
    "n_vs_spearman.tsv",
    "gridsearch_results.csv",
)
PROJECTION_MODE_LABELS = {
    "0": "direction-based",
    "1": "regression hyper-plane",
}
COMPACT_ANCHOR_LABELS = {
    "first_vs_last": "Det. extremes",
    "random_sample_first_last": "Random sample",
    "random_median": "Median rand.",
    "percentile_medians": "Percentile med.",
}
GRID_SCORE_TOLERANCE = 1e-6


def _bar_annotation(params) -> str | None:
    if not params:
        return None
    method = params.get("anchor__method")
    if not method:
        return None
    label = COMPACT_ANCHOR_LABELS.get(str(method), anchor_method_label(method))
    regression = str(params.get("projection__regression", "False")).lower() in {
        "true",
        "1",
    }
    return f"{label}\n{'w/ reg' if regression else 'no reg'}"


def _latest_run(repo_root: Path, experiment: str, backend: str) -> Path | None:
    output_root = repo_root / "outputs" / BACKEND_OUTPUT_DIRS[backend] / experiment
    candidates = []
    if not output_root.is_dir():
        return None
    for config_path in output_root.glob("*/config.json"):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if config.get("experiment") != experiment or config.get("feature_backend") != backend:
            continue
        run_dir = config_path.parent
        if (run_dir / "metrics.tsv").is_file():
            candidates.append((config_path.stat().st_mtime_ns, run_dir))
    return max(candidates, default=(None, None))[1]


def _missing_plot_artifacts(run_dir: Path):
    return [name for name in PLOT_ARTIFACTS if not (run_dir / name).is_file()]


def _metrics_for_run(backend: str, run_dir: Path):
    metrics = read_tsv(run_dir / "metrics.tsv")
    by_name = {row.get("metric"): row for row in metrics}
    row = by_name.get("spearman")
    if row is None:
        raise ValueError(f"{run_dir} has no Spearman metric")
    return [
        {
            "backend": backend,
            "metric": "spearman",
            "value": row["mean"],
            "ci95": row.get("ci95", ""),
            "folds": row.get("folds", ""),
            "support": row.get("rows_kept", ""),
            "output_dir": str(run_dir),
        }
    ]


def _write_plot(path: Path, experiment: str, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    backends = list(dict.fromkeys(row["backend"] for row in rows))
    fig, ax = plt.subplots(figsize=(max(8, len(backends) * 1.5), 5.5))
    values = [
        float(next(row["value"] for row in rows if row["backend"] == backend))
        for backend in backends
    ]
    errors = [
        float(next(row["ci95"] for row in rows if row["backend"] == backend) or 0.0)
        for backend in backends
    ]
    ax.bar(np.arange(len(backends)), values, color="#4C78A8")
    ax.errorbar(
        np.arange(len(backends)),
        values,
        yerr=errors,
        fmt="none",
        color="#222222",
        capsize=4,
    )
    ax.set_ylabel("Spearman correlation (mean ± 95% CI)")
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.set_xticks(np.arange(len(backends)))
    ax.set_xticklabels(backends, rotation=25, ha="right")
    ax.set_title(f"{experiment}: backend comparison")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _params_from_text(text: str):
    params = {}
    for chunk in text.split("; "):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        params[key.strip()] = value.strip()
    return params


def _selected_grid_params(run_dir: Path, score: float):
    grid_path = run_dir / "gridsearch_results.csv"
    if not np.isfinite(score):
        return None, 0
    matches = []
    candidate_count = 0
    with grid_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            candidate_count += 1
            try:
                row_score = float(row["mean_test_score"])
            except (KeyError, ValueError):
                continue
            if np.isfinite(row_score) and abs(row_score - score) <= GRID_SCORE_TOLERANCE:
                matches.append(row)
    if not matches:
        return None, candidate_count

    def _match_key(row):
        params = _params_from_text(row["params"])
        try:
            anchor_n = int(params.get("anchor__n", "inf"))
        except ValueError:
            anchor_n = float("inf")
        try:
            rank = int(row["rank_test_score"])
            row_score = float(row["mean_test_score"])
        except (KeyError, ValueError):
            rank = 0
            row_score = float("-inf")
        return (
            anchor_n,
            -row_score,
            rank,
            params.get("anchor__method", ""),
        )

    return _params_from_text(min(matches, key=_match_key)["params"]), candidate_count


def _config_summary(experiment: str, params, grid_size: int) -> str:
    if params is None:
        return f"grid rows={grid_size} · selected configuration unavailable"

    pca = params.get("pca", "?")
    if pca == "PCA":
        components = params.get("pca__n_components", "?")
        if components == "None":
            components = "all"
        pca_label = f"PCA({components}, whiten={params.get('pca__whiten', '?')})"
    elif pca == "VarianceThreshold":
        pca_label = "variance-threshold"
    elif pca == "FunctionTransformer":
        pca_label = "identity"
    else:
        pca_label = pca

    pls = params.get("pls", "?")
    if pls == "AnchorPLS":
        pls_label = f"PLS(scale={params.get('pls__scale', '?')})"
    elif pls == "LinearDiscriminantAnalysis":
        pls_label = "LDA"
    elif pls == "FunctionTransformer":
        pls_label = "identity"
    else:
        pls_label = pls

    std = params.get("std", "?")
    std_label = {
        "StandardScaler": "scale",
        "FunctionTransformer": "raw",
    }.get(std, std)

    pipeline = EXPERIMENTS[experiment]["pipeline"].pipeline
    projection = dict(pipeline.steps).get("projection")
    projection_label = type(projection).__name__ if projection is not None else "?"
    if "projection" in params:
        projection_label = params["projection"]
    regression = params.get("projection__regression")
    if regression is not None:
        projection_label = f"{projection_label}(regression={regression})"

    method = params.get("anchor__method", "?")
    return " · ".join(
        [
            f"grid rows={grid_size}",
            f"n={params.get('anchor__n', '?')}",
            anchor_method_label(method),
            f"pca={pca_label}",
            f"pls={pls_label}",
            f"std={std_label}",
            f"projection={projection_label}",
        ]
    )




def _read_projections(run_dir: Path):
    projections = {}
    for row in read_tsv(run_dir / "projections.tsv"):
        try:
            target = float(row["target"])
            predicted = float(row["predicted"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite(target) and np.isfinite(predicted)):
            continue
        projections[row["query_id"]] = (target, predicted)
    return projections


def _load_sweep(run_dir: Path):
    sweep = {}
    for row in read_tsv(run_dir / "n_vs_spearman.tsv"):
        try:
            n = int(row["n"])
            score = float(row["spearman"])
            ci95 = float(row["ci95"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (
            str(row.get("projection_regression", "0")),
            str(row.get("anchor_method", "?")),
        )
        sweep.setdefault(key, []).append((n, score, ci95))
    for points in sweep.values():
        points.sort(key=lambda point: point[0])
    return sweep


def _write_predicted_comparison(
    path: Path, experiment: str, runs, configs, metrics
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    projections = {backend: _read_projections(run_dir) for backend, run_dir in runs.items()}
    if not projections:
        raise ValueError(f"{experiment} has no finite prediction rows for comparison")

    first_backend = next(iter(projections))
    common_ids = set(projections[first_backend])
    for rows in projections.values():
        common_ids &= set(rows)
    if not common_ids:
        raise ValueError(f"{experiment} backends have no common query_ids to compare")

    real = np.asarray([projections[first_backend][qid][0] for qid in common_ids])
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    colors = plt.get_cmap("tab10").colors
    for index, (backend, rows) in enumerate(projections.items()):
        targets = np.asarray([rows[qid][0] for qid in common_ids])
        if not np.allclose(targets, real, rtol=0.0, atol=1e-8):
            raise ValueError(
                f"{experiment}/{backend} has targets that do not align with {first_backend}"
            )
        predicted = np.asarray([rows[qid][1] for qid in common_ids])
        mean, ci95 = metrics.get(backend, (float("nan"), float("nan")))
        ax.scatter(
            real,
            predicted,
            s=28,
            alpha=0.62,
            color=colors[index % len(colors)],
            label=f"{backend} (ρ={mean:.3f} ± {ci95:.3f})",
            edgecolors="none",
        )

    all_predicted = np.concatenate(
        [
            np.asarray([rows[qid][1] for qid in common_ids])
            for rows in projections.values()
        ]
    )
    low = float(min(np.min(real), np.min(all_predicted)))
    high = float(max(np.max(real), np.max(all_predicted)))
    ax.plot(
        [low, high], [low, high], linestyle="--", linewidth=1.6, color="#222222", alpha=0.8
    )
    ax.set_xlabel("Real")
    ax.set_ylabel("Predicted")
    ax.set_title(
        f"{experiment} · out-of-fold predictions (selected grid configuration per backend)"
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=9, loc="best")

    notes = []
    note_line_count = 0
    for backend, run_dir in runs.items():
        if backend not in projections:
            continue
        note = textwrap.fill(
            f"{backend}: {configs.get(backend, 'unknown')} | run={run_dir.name}",
            width=88,
            subsequent_indent="  ",
        )
        notes.append(note)
        note_line_count += note.count("\n") + 1
    bottom = min(0.58, 0.035 * note_line_count + 0.04)
    fig.text(
        0.01,
        0.012,
        "\n".join(notes),
        fontsize=7.5,
        family="monospace",
        va="bottom",
    )
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _write_selection_sweep(path: Path, experiment: str, sweeps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweeps = {backend: sweep for backend, sweep in sweeps.items() if sweep}
    if not sweeps:
        return False

    modes = sorted({mode for sweep in sweeps.values() for mode, _ in sweep})
    all_n_values = sorted(
        {
            point[0]
            for sweep in sweeps.values()
            for points in sweep.values()
            for point in points
            if np.isfinite(point[1])
        }
    )
    methods = sorted({method for sweep in sweeps.values() for _, method in sweep})
    if not modes or not methods:
        return False

    fig, axes = plt.subplots(
        len(modes),
        len(methods),
        figsize=(4.3 * len(methods), 3.5 * len(modes)),
        sharey=True,
        squeeze=False,
    )
    colors = plt.get_cmap("tab10").colors
    for row_index, mode in enumerate(modes):
        for col_index, method in enumerate(methods):
            ax = axes[row_index][col_index]
            for backend_index, (backend, sweep) in enumerate(sweeps.items()):
                points = sweep.get((mode, method), [])
                points = [
                    point
                    for point in points
                    if np.isfinite(point[1]) and np.isfinite(point[2])
                ]
                if not points:
                    continue
                n_values = np.asarray([point[0] for point in points], dtype=np.float64)
                scores = np.asarray([point[1] for point in points], dtype=np.float64)
                ci95 = np.asarray([point[2] for point in points], dtype=np.float64)
                color = colors[backend_index % len(colors)]
                ax.plot(
                    n_values,
                    scores,
                    marker="o",
                    markersize=3.5,
                    linewidth=1.3,
                    color=color,
                    label=backend,
                )
                ax.fill_between(
                    n_values, scores - ci95, scores + ci95, color=color, alpha=0.08
                )
            ax.set_title(
                f"{PROJECTION_MODE_LABELS.get(mode, mode)} · {method}", fontsize=10
            )
            ax.grid(axis="y", alpha=0.18)
            ax.set_xticks(all_n_values)
            ax.tick_params(axis="x", labelsize=8)
            if row_index == len(modes) - 1:
                ax.set_xlabel("n")
            ax.set_ylim(0.0, 1.0)
            if col_index == 0:
                ax.set_ylabel("Spearman")

    handles = []
    labels = []
    for ax in axes.flat:
        axis_handles, axis_labels = ax.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(6, len(labels)),
            frameon=False,
        )
    fig.suptitle(f"{experiment} · selection algorithm and n sweep per backend")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _write_aggregate_spearman(
    output_dir: Path, report_rows, experiments, backends, selected_params=None
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    experiments = [experiment for experiment in experiments if report_rows.get(experiment)]
    if not experiments:
        return
    fig, ax = plt.subplots(figsize=(max(9, len(experiments) * 1.6), 5.5))
    positions = np.arange(len(experiments), dtype=float)
    present = []
    for backend in backends:
        values = []
        for experiment in experiments:
            row = next(
                (
                    row
                    for row in report_rows[experiment]
                    if row["backend"] == backend and row["metric"] == "spearman"
                ),
                None,
            )
            if row is None:
                break
            values.append(float(row["value"]))
        else:
            present.append((backend, values))
    if not present:
        plt.close(fig)
        return

    width = 0.8 / len(present)
    colors = plt.get_cmap("tab10").colors
    selected_params = selected_params or {}
    for index, (backend, values) in enumerate(present):
        offset = (index - (len(present) - 1) / 2) * width
        bar_container = ax.bar(
            positions + offset,
            values,
            width,
            label=backend,
            color=colors[index % len(colors)],
        )
        for bar, experiment in zip(bar_container, experiments):
            annotation = _bar_annotation(
                selected_params.get(experiment, {}).get(backend)
            )
            height = float(bar.get_height())
            if annotation is None or height <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height / 2.0,
                annotation,
                ha="center",
                va="center",
                rotation=90,
                fontsize=5.5,
                color="white",
                linespacing=0.9,
                clip_on=True,
            )
    ax.set_xticks(positions)
    ax.set_xticklabels(experiments, rotation=25, ha="right")
    ax.set_ylabel("Spearman correlation")
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.set_title("Backend comparison across correlation tasks")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "all_embeddings_spearman.png", dpi=180)
    plt.close(fig)


def generate_comparison_reports(
    repo_root: Path | None = None,
    experiments: tuple[str, ...] | list[str] | None = None,
    backends: tuple[str, ...] | list[str] = EMBEDDING_BACKENDS,
) -> Path:
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    experiments = tuple(experiments or EXPERIMENTS)
    backends = tuple(backends)
    invalid_backends = sorted(set(backends) - set(FEATURE_BACKENDS))
    if invalid_backends:
        raise ValueError(f"Comparison reports require known backends: {invalid_backends}")
    invalid_experiments = sorted(set(experiments) - set(EXPERIMENTS))
    if invalid_experiments:
        raise ValueError(f"Unknown experiment(s): {invalid_experiments}")

    report_rows = {}
    plot_run_dirs = {}
    selected_params = {}
    coverage_rows = []
    missing = []
    for experiment in experiments:
        rows = []
        experiment_plot_runs = {}
        for backend in backends:
            run_dir = _latest_run(repo_root, experiment, backend)
            if run_dir is None:
                missing.append(f"{experiment}/{backend}")
                continue
            try:
                rows.extend(_metrics_for_run(backend, run_dir))
            except (OSError, ValueError, KeyError) as exc:
                missing.append(f"{experiment}/{backend}: {exc}")
                continue
            missing_artifacts = _missing_plot_artifacts(run_dir)
            if missing_artifacts:
                print(
                    f"warning: {experiment}/{backend}: comparison plots omit this backend; "
                    f"{run_dir.name} lacks {', '.join(missing_artifacts)}",
                    file=sys.stderr,
                )
                coverage_rows.append(
                    {
                        "experiment": experiment,
                        "backend": backend,
                        "artifact": ",".join(missing_artifacts),
                        "status": "missing",
                        "output_dir": str(run_dir),
                    }
                )
            else:
                experiment_plot_runs[backend] = run_dir
                coverage_rows.append(
                    {
                        "experiment": experiment,
                        "backend": backend,
                        "artifact": "all",
                        "status": "available",
                        "output_dir": str(run_dir),
                    }
                )
        report_rows[experiment] = rows
        plot_run_dirs[experiment] = experiment_plot_runs
    if missing:
        raise FileNotFoundError(
            "Cannot create a complete comparison report; missing or invalid run outputs:\n"
            + "\n".join(f"  {item}" for item in missing)
        )

    parent = repo_root / "outputs" / "model_comparison"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    staged_dir = Path(tempfile.mkdtemp(prefix=".staging_", dir=parent))
    final_dir = parent / stamp
    try:
        for experiment, rows in report_rows.items():
            write_tsv(staged_dir / f"{experiment}.tsv", rows, REPORT_COLUMNS)
            _write_plot(staged_dir / f"{experiment}.png", experiment, rows)

            plot_runs = plot_run_dirs[experiment]
            if plot_runs:
                metrics = {
                    row["backend"]: (
                        float(row["value"]),
                        float(row["ci95"] or 0.0),
                    )
                    for row in rows
                    if row["metric"] == "spearman"
                }
                configs = {}
                selected_params[experiment] = {}
                for backend, run_dir in plot_runs.items():
                    params, grid_size = _selected_grid_params(
                        run_dir, metrics[backend][0]
                    )
                    configs[backend] = _config_summary(experiment, params, grid_size)
                    selected_params[experiment][backend] = params
                _write_predicted_comparison(
                    staged_dir / f"{experiment}_predicted_vs_real.png",
                    experiment,
                    plot_runs,
                    configs,
                    metrics,
                )
                _write_selection_sweep(
                    staged_dir / f"{experiment}_selection_sweep.png",
                    experiment,
                    {
                        backend: _load_sweep(run_dir)
                        for backend, run_dir in plot_runs.items()
                    },
                )

        _write_aggregate_spearman(
            staged_dir, report_rows, experiments, backends, selected_params
        )
        write_tsv(
            staged_dir / "plot_coverage.tsv",
            coverage_rows,
            COVERAGE_COLUMNS,
        )
        os.replace(staged_dir, final_dir)
    except BaseException:
        shutil.rmtree(staged_dir, ignore_errors=True)
        raise
    return final_dir
