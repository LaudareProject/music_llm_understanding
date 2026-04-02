from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.composer_axis.run import main


if __name__ == "__main__":
    main(use_hann_selection=False, anchor_selection_method="random_median")
