import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pitch_axis.run import main


def run():
    os.environ["PROJECTION_MODE"] = "regression"
    main()


if __name__ == "__main__":
    run()
