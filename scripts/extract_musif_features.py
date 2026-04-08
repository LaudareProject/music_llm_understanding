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
        "--dataset-column",
        default="dataset",
        help="Column used to filter rows. Defaults to 'dataset'. When missing, all rows are kept.",
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


def copy_midi_files(rows, source_dir, repo_root):
    source_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        query_id = row["query_id"]
        path = row.get("path")
        if not path:
            raise ValueError(f"Missing path for query {query_id}")
        src = (repo_root / path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"MIDI file {src} not found for {query_id}")
        dest = source_dir / f"{query_id}.mid"
        if not dest.exists() or src.stat().st_mtime > dest.stat().st_mtime:
            shutil.copy2(src, dest)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    rows = read_tsv(args.tsv)
    dataset_rows = []
    for row in rows:
        if args.dataset_column not in row:
            dataset_rows = rows
            break
        if str(row.get(args.dataset_column, "")).strip() == args.dataset:
            dataset_rows.append(row)
    if not dataset_rows:
        raise ValueError(f"No rows found for dataset '{args.dataset}' in {args.tsv}")

    output_csv = musif_features_csv_path(repo_root, args.dataset)
    if output_csv.exists() and not args.force:
        print(f"Musif CSV already exists at {output_csv}, use --force to regenerate")
        return

    source_dir = musif_sources_dir(repo_root, args.dataset)
    copy_midi_files(dataset_rows, source_dir, repo_root)

    cache_dir = musif_cache_dir(repo_root, args.dataset)
    run_musif_extraction(source_dir, output_csv, cache_dir, repo_root, njobs=args.njobs)
    print(f"Musif features written to {output_csv}")


if __name__ == "__main__":
    main()
