import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from experiments.common.tsv import read_tsv


def run_script_and_capture_output_dir(script_path, output_root, extra_env=None):
    output_root = Path(output_root)
    before = {path.name for path in output_root.glob(f"{script_path.stem}_*")}
    env = None
    if extra_env:
        env = {**os.environ, **{str(k): str(v) for k, v in extra_env.items()}}
    subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
        cwd=script_path.parents[2],
        env=env,
    )
    after = list(output_root.glob(f"{script_path.stem}_*"))
    new_dirs = [path for path in after if path.name not in before and path.is_dir()]
    if new_dirs:
        return max(new_dirs, key=lambda path: path.stat().st_mtime)
    if after:
        return max(after, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError(f"No output directory found for {script_path}")


def to_float(value):
    return float(value) if value != "nan" else float("nan")


def best_row(rows, predicate, score_key, maximize):
    candidates = [row for row in rows if predicate(row)]
    if not candidates:
        return None
    return (
        max(candidates, key=lambda row: to_float(row[score_key]))
        if maximize
        else min(candidates, key=lambda row: to_float(row[score_key]))
    )


def metric_value(row, metric_name):
    value = to_float(row[metric_name])
    return value
