from pathlib import Path
import os
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pitch_axis.run_all import main


if __name__ == "__main__":
    os.environ["FEATURE_BACKEND"] = "musif"
    main()
