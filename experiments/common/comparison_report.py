from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from experiments.common.feature_backend import (
    BACKEND_OUTPUT_DIRS,
    EMBEDDING_BACKENDS,
)
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
SENTIMENT_METRICS = ("class_accuracy", "balanced_class_accuracy")


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


def _metrics_for_run(experiment: str, backend: str, run_dir: Path):
    metrics = read_tsv(run_dir / "metrics.tsv")
    by_name = {row.get("metric"): row for row in metrics}
    if experiment == "sentiment_prediction":
        missing = [metric for metric in SENTIMENT_METRICS if metric not in by_name]
        if missing:
            raise ValueError(
                f"{run_dir} lacks required sentiment metrics: {', '.join(missing)}"
            )
        return [
            {
                "backend": backend,
                "metric": name,
                "value": by_name[name]["value"],
                "ci95": "",
                "folds": "",
                "support": by_name[name].get("support", ""),
                "output_dir": str(run_dir),
            }
            for name in SENTIMENT_METRICS
        ]

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
    if experiment == "sentiment_prediction":
        positions = np.arange(len(backends), dtype=float)
        width = 0.36
        for offset, metric in enumerate(SENTIMENT_METRICS):
            values = [
                float(next(row["value"] for row in rows if row["backend"] == backend and row["metric"] == metric))
                for backend in backends
            ]
            ax.bar(positions + (offset - 0.5) * width, values, width, label=metric)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.legend(frameon=False)
    else:
        values = [
            float(next(row["value"] for row in rows if row["backend"] == backend))
            for backend in backends
        ]
        errors = [
            float(next(row["ci95"] for row in rows if row["backend"] == backend) or 0.0)
            for backend in backends
        ]
        ax.bar(np.arange(len(backends)), values, color="#4C78A8")
        ax.errorbar(np.arange(len(backends)), values, yerr=errors, fmt="none", color="#222222", capsize=4)
        ax.set_ylabel("Spearman correlation (mean ± 95% CI)")
        ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.set_xticks(np.arange(len(backends)))
    ax.set_xticklabels(backends, rotation=25, ha="right")
    ax.set_title(f"{experiment}: embedding backend comparison")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_comparison_reports(
    repo_root: Path | None = None,
    experiments: tuple[str, ...] | list[str] | None = None,
    backends: tuple[str, ...] | list[str] = EMBEDDING_BACKENDS,
) -> Path:
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    experiments = tuple(experiments or EXPERIMENTS)
    backends = tuple(backends)
    invalid_backends = sorted(set(backends) - set(EMBEDDING_BACKENDS))
    if invalid_backends:
        raise ValueError(f"Comparison reports require embedding backends: {invalid_backends}")
    invalid_experiments = sorted(set(experiments) - set(EXPERIMENTS))
    if invalid_experiments:
        raise ValueError(f"Unknown experiment(s): {invalid_experiments}")

    report_rows = {}
    missing = []
    for experiment in experiments:
        rows = []
        for backend in backends:
            run_dir = _latest_run(repo_root, experiment, backend)
            if run_dir is None:
                missing.append(f"{experiment}/{backend}")
                continue
            try:
                rows.extend(_metrics_for_run(experiment, backend, run_dir))
            except (OSError, ValueError, KeyError) as exc:
                missing.append(f"{experiment}/{backend}: {exc}")
        report_rows[experiment] = rows
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
        os.replace(staged_dir, final_dir)
    except BaseException:
        shutil.rmtree(staged_dir, ignore_errors=True)
        raise
    return final_dir
