from __future__ import annotations

from music_latents.runner_axes import RunnerAxesMixin
from music_latents.runner_core import RunnerCoreMixin
from music_latents.runner_supervised import RunnerSupervisedMixin


class ExperimentRunner(
    RunnerSupervisedMixin,
    RunnerAxesMixin,
    RunnerCoreMixin,
):
    _bootstrap_samples: int = 1000
    _plot_dpi: int = 300

    _pitch_class_palette = {
        0: "#e41a1c",
        1: "#377eb8",
        2: "#4daf4a",
        3: "#984ea3",
        4: "#ff7f00",
        5: "#a65628",
        6: "#f781bf",
        7: "#999999",
        8: "#66c2a5",
        9: "#fc8d62",
        10: "#8da0cb",
        11: "#e78ac3",
    }

    _pitch_class_aliases = {
        "c": 0,
        "b#": 0,
        "cs": 1,
        "c#": 1,
        "db": 1,
        "d": 2,
        "ds": 3,
        "d#": 3,
        "eb": 3,
        "e": 4,
        "fb": 4,
        "f": 5,
        "e#": 5,
        "fs": 6,
        "f#": 6,
        "gb": 6,
        "g": 7,
        "gs": 8,
        "g#": 8,
        "ab": 8,
        "a": 9,
        "as": 10,
        "a#": 10,
        "bb": 10,
        "b": 11,
        "cb": 11,
    }
