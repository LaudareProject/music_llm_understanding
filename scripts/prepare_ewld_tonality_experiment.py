from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from music_latents.ewld import (  # noqa: E402
    TARGET_TONALITIES,
    analyze_musicxml_tonality,
    find_musicxml_files,
    select_balanced_tonality_subset,
)


def _song_id(tonality: str, index: int) -> str:
    return f"{tonality.lower()}_{index:03d}".replace("#", "s")


def _group_id(prefix: str, tonality: str) -> str:
    return f"{prefix}_{tonality.lower()}".replace("#", "s")


def _tonality_label(tonality: str) -> str:
    tonic, mode = tonality.split("_", maxsplit=1)
    return f"{tonic} {mode}"


def _modality_color(tonality: str) -> str:
    return "#1f77b4" if tonality.endswith("major") else "#d62728"


def render_filter_log(
    *,
    dataset_root: Path,
    target_per_tonality: int,
    grouped_paths: dict[str, list[Path]],
    analyses,
) -> str:
    lines: list[str] = []
    lines.append(f"dataset_root={dataset_root}")
    lines.append(f"target_per_tonality={target_per_tonality}")
    lines.append(f"selected_tonalities={len(grouped_paths)}")
    lines.append(
        "selection_policy=trust_estimated_tonality_even_when_key_signature_missing_or_mismatched"
    )
    lines.append("")
    lines.append("[selected_counts]")
    for tonality in sorted(grouped_paths.keys()):
        lines.append(f"{tonality}: {len(grouped_paths[tonality])}")

    lines.append("")
    lines.append("[selected_files]")
    for tonality in sorted(grouped_paths.keys()):
        lines.append(f"{tonality}:")
        for path in grouped_paths[tonality]:
            lines.append(f"  - {path}")

    reason_counts: dict[str, int] = {}
    for item in analyses:
        reason_counts[item.reason] = reason_counts.get(item.reason, 0) + 1
    lines.append("")
    lines.append("[analysis_reason_counts]")
    for reason, count in sorted(reason_counts.items()):
        lines.append(f"{reason}: {count}")

    lines.append("")
    lines.append("[analysis_entries]")
    for item in analyses:
        lines.append(
            " | ".join(
                [
                    f"accepted={item.accepted}",
                    f"tonality={item.tonality}",
                    f"reason={item.reason}",
                    f"size={item.file_size}",
                    f"path={item.file_path}",
                ]
            )
        )
    return "\n".join(lines) + "\n"


def resolve_dataset_path(dataset_root: Path) -> Path:
    dataset_path = (dataset_root / "dataset").resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_path} "
            f"(expected under --dataset-root {dataset_root.resolve()})"
        )
    return dataset_path


def default_worker_count() -> int:
    return max(1, (os.cpu_count() or 1) // 2)


def print_progress_bar(completed: int, total: int, width: int = 32) -> None:
    if total <= 0:
        return
    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\rEstimating tonalities [{bar}] {completed}/{total}",
        end="",
        flush=True,
    )
    if completed >= total:
        print()


def build_config_payload(
    grouped_paths: dict[str, list[Path]],
    *,
    config_parent: Path,
    experiment_name: str,
    output_root: str,
) -> dict:
    queries = []
    average = {}
    median = {}

    tonalities = sorted(grouped_paths.keys())
    for tonality in tonalities:
        song_ids = []
        for idx, path in enumerate(grouped_paths[tonality], start=1):
            query_id = _song_id(tonality, idx)
            song_ids.append(query_id)
            rel_path = os.path.relpath(Path(path).resolve(), config_parent)
            queries.append(
                {
                    "id": query_id,
                    "name": f"{_tonality_label(tonality)} #{idx}",
                    "color": _modality_color(tonality),
                    "musicxml_path": rel_path,
                }
            )

        avg_id = _group_id("avg", tonality)
        med_id = _group_id("med", tonality)
        average[avg_id] = {
            "name": f"{_tonality_label(tonality)} avg",
            "color": _modality_color(tonality),
            "queries": song_ids,
        }
        median[med_id] = {
            "name": f"{_tonality_label(tonality)} median",
            "color": _modality_color(tonality),
            "queries": song_ids,
        }

    if len(tonalities) < 2:
        raise ValueError("Need at least two tonalities to build tonality axes.")

    axis_pairs: list[tuple[str, str]] = []
    if len(tonalities) >= 4:
        axis_pairs.append((tonalities[0], tonalities[len(tonalities) // 2]))
        axis_pairs.append((tonalities[1], tonalities[(len(tonalities) // 2) + 1]))
    elif len(tonalities) == 3:
        axis_pairs.append((tonalities[0], tonalities[1]))
        axis_pairs.append((tonalities[0], tonalities[2]))
    else:
        axis_pairs.append((tonalities[0], tonalities[1]))

    avg_axis_names = [
        f"avg_{left}_vs_{right}".lower().replace("#", "s") for left, right in axis_pairs
    ]
    med_axis_names = [
        f"med_{left}_vs_{right}".lower().replace("#", "s") for left, right in axis_pairs
    ]

    avg_axes = [
        {
            "name": avg_axis_names[idx],
            "query_a": _group_id("avg", left),
            "query_b": _group_id("avg", right),
            "method": "explicit",
        }
        for idx, (left, right) in enumerate(axis_pairs)
    ]
    med_axes = [
        {
            "name": med_axis_names[idx],
            "query_a": _group_id("med", left),
            "query_b": _group_id("med", right),
            "method": "explicit",
        }
        for idx, (left, right) in enumerate(axis_pairs)
    ]

    reference_id = _group_id("avg", tonalities[0])
    pca_axes = [
        {
            "name": "pca01",
            "query_a": reference_id,
            "query_b": reference_id,
            "method": "pca_component",
            "params": {"component_index": 0},
        },
        {
            "name": "pca02",
            "query_a": reference_id,
            "query_b": reference_id,
            "method": "pca_component",
            "params": {"component_index": 1},
        },
    ]

    cosine_references = [_group_id("avg", "C_major"), _group_id("avg", "A_major")]
    missing_reference_groups = [
        reference for reference in cosine_references if reference not in average
    ]
    if missing_reference_groups:
        raise ValueError(
            "Missing required EWLD cosine reference groups: "
            + ", ".join(missing_reference_groups)
            + ". Ensure C_major and A_major are present in the selected tonalities."
        )

    payload = {
        "experiment": {
            "name": experiment_name,
            "output_root": output_root,
            "models": ["aria_embedding", "phrase_vae"],
        },
        "models": {
            "aria_embedding": {
                "type": "aria_embedding",
                "repo_id": "loubb/aria-medium-embedding",
                "checkpoint_filename": "model.safetensors",
                "model_config_name": "medium-emb",
                "device": "cpu",
            },
            "phrase_vae": {
                "type": "phrase_vae",
                "repo_id": "LongshenOu/phrase-vae",
                "device": "cpu",
                "trim": 16,
            },
        },
        "queries": queries,
        "average": average,
        "median": median,
        "axes": [*avg_axes, *med_axes, *pca_axes],
        "plots": {
            "tonality_pca_components": {
                "cosine": cosine_references,
                "options": {
                    "origin_strategy": "mean",
                    "lda_groups": True,
                    "averaged_only": True,
                    "group_filter": "all",
                    "pca": True,
                },
            },
            "tonality_averages": {
                "cosine": cosine_references,
                "options": {
                    "origin_strategy": "mean",
                    "lda_groups": True,
                    "averaged_only": True,
                    "group_filter": "average",
                },
            },
            "tonality_averages_mds": {
                "mds": True,
                "options": {
                    "averaged_only": True,
                    "group_filter": "average",
                },
            },
            "tonality_averages_pca_only": {
                "cosine": cosine_references,
                "options": {
                    "origin_strategy": "mean",
                    "lda_groups": True,
                    "pca": True,
                    "averaged_only": True,
                    "group_filter": "average",
                },
            },
            "tonality_averages_orthonormalize_only": {
                "cosine": cosine_references,
                "options": {
                    "origin_strategy": "mean",
                    "lda_groups": True,
                    "orthonormalize": True,
                    "averaged_only": True,
                    "group_filter": "average",
                },
            },
            "tonality_averages_pca_and_orthonormalize": {
                "cosine": cosine_references,
                "options": {
                    "origin_strategy": "mean",
                    "lda_groups": True,
                    "pca": True,
                    "orthonormalize": True,
                    "averaged_only": True,
                    "group_filter": "average",
                },
            },
            "tonality_medians": {
                "cosine": cosine_references,
                "options": {
                    "origin_strategy": "mean",
                    "lda_groups": True,
                    "averaged_only": True,
                    "group_filter": "median",
                },
            },
            "tonality_medians_mds": {
                "mds": True,
                "options": {
                    "averaged_only": True,
                    "group_filter": "median",
                },
            },
            "tonality_medians_pca_only": {
                "cosine": cosine_references,
                "options": {
                    "origin_strategy": "mean",
                    "lda_groups": True,
                    "pca": True,
                    "averaged_only": True,
                    "group_filter": "median",
                },
            },
            "tonality_medians_orthonormalize_only": {
                "cosine": cosine_references,
                "options": {
                    "origin_strategy": "mean",
                    "lda_groups": True,
                    "orthonormalize": True,
                    "averaged_only": True,
                    "group_filter": "median",
                },
            },
            "tonality_medians_pca_and_orthonormalize": {
                "cosine": cosine_references,
                "options": {
                    "origin_strategy": "mean",
                    "lda_groups": True,
                    "pca": True,
                    "orthonormalize": True,
                    "averaged_only": True,
                    "group_filter": "median",
                },
            },
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare EWLD balanced tonality config (average + median groups)."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("EWLD"))
    parser.add_argument("--bin-dir", type=Path, default=Path("bin"))
    parser.add_argument(
        "--output-config", type=Path, default=Path("configs/ewld_tonality.yaml")
    )
    parser.add_argument("--target-per-tonality", type=int, default=50)
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Parallel workers for tonality estimation (default: half CPUs).",
    )
    parser.add_argument(
        "--tonality-set",
        choices=["available", "all24"],
        default="available",
        help="Use only tonalities with enough validated songs, or require all 24 tonalities.",
    )
    parser.add_argument("--experiment-name", type=str, default="EWLDTonality")
    parser.add_argument("--output-root", type=str, default="../outputs")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    dataset_path = resolve_dataset_path(dataset_root)

    files = find_musicxml_files(dataset_path)
    if not files:
        raise FileNotFoundError(f"No MusicXML/MXL files found under {dataset_path}")

    grouped_paths, analyses = select_balanced_tonality_subset(
        files,
        target_per_tonality=args.target_per_tonality,
        analyzer=lambda path: analyze_musicxml_tonality(
            path,
            bin_dir=args.bin_dir.resolve(),
        ),
        tonalities=TARGET_TONALITIES if args.tonality_set == "all24" else None,
        workers=args.workers,
        progress_callback=print_progress_bar,
    )

    output_config = args.output_config.resolve()
    output_config.parent.mkdir(parents=True, exist_ok=True)

    payload = build_config_payload(
        grouped_paths,
        config_parent=output_config.parent,
        experiment_name=args.experiment_name,
        output_root=args.output_root,
    )
    output_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    output_filter_log = (Path.cwd() / "ewld_tonality_filter.log").resolve()
    output_filter_log.write_text(
        render_filter_log(
            dataset_root=dataset_path,
            target_per_tonality=args.target_per_tonality,
            grouped_paths=grouped_paths,
            analyses=analyses,
        ),
        encoding="utf-8",
    )

    print(f"Wrote config: {output_config}")
    print(f"Wrote filter log: {output_filter_log}")
    accepted_counts = Counter(
        item.tonality for item in analyses if item.accepted and item.tonality
    )
    print("Validated songs per tonality:")
    for tonality in sorted(accepted_counts.keys()):
        print(f"- {tonality}: {accepted_counts[tonality]}")

    print("Selected songs per tonality:")
    for tonality in sorted(grouped_paths.keys()):
        print(f"- {tonality}: {len(grouped_paths[tonality])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
