import json
import os
from datetime import datetime
from pathlib import Path
import sys


def create_folder(output_root, config=None):
    run_origin = os.environ.get("RUN_ORIGIN", "cli").strip().lower()
    script_stem = Path(sys.argv[0]).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_origin == "feature_selection":
        folder_name = f"feature_selection_{script_stem}_{timestamp}"
    else:
        folder_name = f"{script_stem}_{timestamp}"
    out_dir = output_root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if config is not None:
        payload = dict(config)
        payload["script"] = script_stem
        payload["run_origin"] = run_origin
        (out_dir / "config.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    return out_dir
