from argparse import ArgumentParser
from pathlib import Path
import shutil
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common.tsv import read_tsv
from experiments.common.feature_backend import (
    musif_features_csv_path,
    musif_cache_dir,
    musif_sources_dir,
    run_musif_extraction,
)
from experiments.common.musif_filter import MUSIF_IGNORE_COLUMNS
from experiments.common.humdrum import kern_to_midi, musicxml_to_midi


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


def query_id_from_filename(value):
    return Path(str(value or "")).stem


def valid_query_ids_from_existing_df(df):
    if df.empty or "FileName" not in df.columns:
        return set()
    numeric_cols = [col for col in df.columns if col not in MUSIF_IGNORE_COLUMNS]
    if not numeric_cols:
        return set()
    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    has_finite = numeric.notna().any(axis=1)
    has_non_zero = numeric.fillna(0.0).ne(0.0).any(axis=1)
    valid = has_finite & has_non_zero
    return {
        query_id_from_filename(name)
        for name in df.loc[valid, "FileName"].tolist()
        if str(name or "").strip()
    }


def merge_existing_and_new_features(existing_df, new_df):
    if existing_df.empty:
        return new_df
    if "FileName" not in existing_df.columns or "FileName" not in new_df.columns:
        return pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    new_ids = {
        query_id_from_filename(name)
        for name in new_df["FileName"].tolist()
        if str(name or "").strip()
    }
    existing_ids = existing_df["FileName"].map(query_id_from_filename)
    filtered_existing = existing_df.loc[~existing_ids.isin(new_ids)]
    return pd.concat([filtered_existing, new_df], ignore_index=True, sort=False)


def build_run_source_dir(source_dir, cache_dir, rows):
    run_source_dir = cache_dir / "run_midi"
    if run_source_dir.exists():
        shutil.rmtree(run_source_dir)
    run_source_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        query_id = row["query_id"]
        src = source_dir / f"{query_id}.mid"
        if src.exists():
            shutil.copy2(src, run_source_dir / src.name)
    return run_source_dir


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
    existing_df = pd.DataFrame()
    if output_csv.exists() and not args.force:
        existing_df = pd.read_csv(output_csv)

    valid_existing = (
        set() if args.force else valid_query_ids_from_existing_df(existing_df)
    )
    rows_to_extract = (
        dataset_rows
        if args.force
        else [row for row in dataset_rows if row["query_id"] not in valid_existing]
    )
    if not rows_to_extract:
        print(
            f"All {len(dataset_rows)} rows already have valid musif features in {output_csv}"
        )
        return
    if not args.force and valid_existing:
        print(
            f"Skipping {len(valid_existing)} rows with existing valid features; extracting {len(rows_to_extract)}"
        )

    source_dir = musif_sources_dir(repo_root, args.dataset)
    kept_rows, skipped = copy_midi_files(rows_to_extract, source_dir, repo_root)
    if skipped:
        print(f"Skipped {len(skipped)} rows during midi preparation")
    if not kept_rows:
        raise ValueError("No rows left after midi preparation")

    cache_dir = musif_cache_dir(repo_root, args.dataset)
    run_source_dir = build_run_source_dir(source_dir, cache_dir, kept_rows)
    fresh_csv = cache_dir / "fresh_features.csv"
    run_musif_extraction(
        run_source_dir, fresh_csv, cache_dir, repo_root, njobs=args.njobs
    )

    fresh_df = pd.read_csv(fresh_csv)
    final_df = (
        fresh_df
        if args.force
        else merge_existing_and_new_features(existing_df, fresh_df)
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    print(f"Musif features written to {output_csv}")


if __name__ == "__main__":
    main()
