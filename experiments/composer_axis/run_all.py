import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import settings
from experiments.common.runner import run_experiment

if __name__ == "__main__":
    run_experiment(settings, "composer_axis_asap")
    run_experiment(settings, "composer_axis_ewld")
    run_experiment(settings, "composer_axis_music21")
