import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pitch_axis.transformed_feature_selection import run_feature_selection


def main():
    run_feature_selection("raw")


if __name__ == "__main__":
    main()
