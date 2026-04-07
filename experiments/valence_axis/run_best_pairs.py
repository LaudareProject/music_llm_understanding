from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.valence_axis.run_base import main


if __name__ == "__main__":
    main(anchor_selection_method="best_pairs")
