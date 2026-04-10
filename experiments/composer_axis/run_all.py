from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common.runner import run_experiment
from experiments import settings


if __name__ == "__main__":
    run_experiment(settings, "composer_axis")
