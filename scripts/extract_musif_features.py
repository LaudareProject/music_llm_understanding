from argparse import ArgumentParser
from pathlib import Path
import shutil
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common.tsv import read_tsv
from experiments.common.feature_backend import (
    musif_features_csv_path,
    musif_cache_dir,
    musif_sources_dir,
    run_musif_extraction,
)
from experiments.common.humdrum import kern_to_midi, musicxml_to_midi
from experiments.pitch_axis.pipeline import write_note_midi


def parse_args():
    parser = ArgumentParser(description="Extract musif features for one dataset")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset identifier already present in the TSV (used to name the CSV)",
    )
    parser.add_argument(
        "--tsv",
        required=True,
        type=Path,
        help="TSV file that contains all rows for the dataset",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=2,
        help="Parallel jobs for musif extraction (default: 2)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-running extraction even if the CSV already exists",
    )
    return parser.parse_args()


def infer_input_kind(row, src):
    input_kind = str(row.get("input_kind", "")).strip().lower()
    if input_kind:
        return input_kind
    suffix = src.suffix.lower()
    if suffix in {".mid", ".midi"}:
        return "midi"
    if suffix in {".xml", ".musicxml", ".mxl"}:
        return "musicxml"
    if suffix in {".krn", ".kern"}:
        return "kern"
    return "unknown"


def prepare_midi(row, source_dir, repo_root, bin_dir):
    query_id = row["query_id"]
    dest = source_dir / f"{query_id}.mid"
    if "path" not in row or not str(row.get("path", "")).strip():
        if {"midi_note", "duration_ticks", "velocity"}.issubset(set(row.keys())):
            write_note_midi(
                dest,
                int(row["midi_note"]),
                int(row["duration_ticks"]),
                int(row["velocity"]),
            )
            return
        raise ValueError(f"Missing path for query {query_id}")

    src = (repo_root / row["path"]).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source file {src} not found for {query_id}")

    input_kind = infer_input_kind(row, src)
    if input_kind == "midi":
        if not dest.exists() or src.stat().st_mtime > dest.stat().st_mtime:
            shutil.copy2(src, dest)
        return
    if input_kind == "musicxml":
        if not dest.exists() or src.stat().st_mtime > dest.stat().st_mtime:
            musicxml_to_midi(src, dest, bin_dir)
        return
    if input_kind == "kern":
        if not dest.exists() or src.stat().st_mtime > dest.stat().st_mtime:
            kern_to_midi(src, dest, bin_dir)
        return
    raise ValueError(
        f"Unsupported input_kind='{input_kind}' for query {query_id} ({src})"
    )


def copy_midi_files(rows, source_dir, repo_root):
    source_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = repo_root / "bin"
    kept = []
    skipped = []
    for row in rows:
        query_id = row["query_id"]
        try:
            prepare_midi(row, source_dir, repo_root, bin_dir)
            kept.append(row)
        except Exception as exc:
            print(f"skipping {query_id}: {exc}")
            skipped.append(query_id)
    return kept, skipped


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    rows = read_tsv(args.tsv)
    if rows and "dataset" not in rows[0]:
        raise ValueError(
            f"TSV {args.tsv} has no 'dataset' column. Ensure prepare_tsv writes a dataset field."
        )
    dataset_rows = [
        row for row in rows if str(row.get("dataset", "")).strip() == args.dataset
    ]
    if not dataset_rows:
        raise ValueError(f"No rows found for dataset '{args.dataset}' in {args.tsv}")

    output_csv = musif_features_csv_path(repo_root, args.dataset)
    if output_csv.exists() and not args.force:
        print(f"Musif CSV already exists at {output_csv}, use --force to regenerate")
        return

    source_dir = musif_sources_dir(repo_root, args.dataset)
    kept_rows, skipped = copy_midi_files(dataset_rows, source_dir, repo_root)
    if skipped:
        print(f"Skipped {len(skipped)} rows during midi preparation")
    if not kept_rows:
        raise ValueError("No rows left after midi preparation")

    cache_dir = musif_cache_dir(repo_root, args.dataset)
    run_musif_extraction(source_dir, output_csv, cache_dir, repo_root, njobs=args.njobs)
    print(f"Musif features written to {output_csv}")


if __name__ == "__main__":
    main()
