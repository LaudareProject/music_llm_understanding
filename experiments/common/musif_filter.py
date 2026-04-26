import csv
import math
from pathlib import Path


MUSIF_IGNORE_COLUMNS = {"FileName", "Id", "WindowId", "WindowRange"}


def _to_float(value):
    text = str(value or "").strip()
    if not text or text in {"-", "nan", "NaN", "None"}:
        return float("nan")
    lowered = text.lower()
    if lowered == "true":
        return 1.0
    if lowered == "false":
        return 0.0
    try:
        return float(text)
    except Exception:
        return float("nan")


def _is_nan_token(value):
    text = str(value or "").strip().lower()
    return text in {"nan", "none"}


def query_ids_with_nan_in_musif_csv(csv_path):
    _, out = _scan_musif_csv(csv_path)
    return out


def read_musif_features_by_query(csv_path):
    out, _ = _scan_musif_csv(csv_path)
    return out


def _scan_musif_csv(csv_path):
    out = {}
    nan_ids = set()
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_name = str(row.get("FileName", "") or "")
            if not file_name:
                continue
            query_id = Path(file_name).stem
            numeric = {}
            has_nan = False
            for key, value in row.items():
                if key in MUSIF_IGNORE_COLUMNS:
                    continue
                number = _to_float(value)
                if math.isnan(number) and _is_nan_token(value):
                    has_nan = True
                    break
                if math.isnan(number):
                    continue
                numeric[key] = number
            if has_nan:
                nan_ids.add(query_id)
                continue
            out[query_id] = numeric
    return out, nan_ids


def filter_rows_with_musif_nan(rows, repo_root):
    datasets = sorted({str(row.get("dataset", "default") or "default") for row in rows})
    nan_ids_by_dataset = {}
    for dataset_name in datasets:
        csv_path = repo_root / "data" / "musif" / dataset_name / "features.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing musif features for dataset '{dataset_name}': {csv_path}"
            )
        nan_ids_by_dataset[dataset_name] = query_ids_with_nan_in_musif_csv(csv_path)

    kept = []
    dropped = []
    for row in rows:
        dataset_name = str(row.get("dataset", "default") or "default")
        query_id = row["query_id"]
        if query_id in nan_ids_by_dataset.get(dataset_name, set()):
            dropped.append(query_id)
            continue
        kept.append(row)

    return kept, dropped
