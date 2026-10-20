# Symbolic Music Latent Experiments

This repo provides a lightweight experiment runner to extract latent representations from symbolic music inputs using pretrained models.

## Quickstart

1. Create a YAML config (see `configs/example.yaml`).
2. Install deps with uv:

```bash
uv sync
```

3. Run the experiments.

## Musif feature extraction

Musif features are stored under `data/musif/<dataset>/features.csv` so that every experiment can reuse the same CSV per dataset. To regenerate them:

```bash
uv run python scripts/extract_musif_features.py --dataset vgmidi --tsv ./experiments/valence_axis/songs.tsv
```

Use `--force` to regenerate an existing CSV.

## Experiments

### Composer chronology:

```bash
uv run python experiments/composer_axis/prepare_tsv.py
uv run python experiments/composer_axis/run_all.py
```

### Pitch axis:

```bash
uv run python experiments/pitch_axis/prepare_tsv.py
uv run python experiments/pitch_axis/run_all.py
```

`prepare_tsv.py` builds `experiments/pitch_axis/notes.tsv` with note-level covariates (`duration_ticks`, `velocity`) used for control analyses.

### Arousal and valence:

```bash
uv run python experiments/arousal_axis/prepare_tsv.py
uv run python experiments/arousal_axis/run_all.py

uv run python experiments/valence_axis/prepare_tsv.py
uv run python experiments/valence_axis/run_all.py
```

### Settings

All experiment pipelines and grid-search parameters are defined in `experiments/settings.py`.

Each `run_all.py` executes a GridSearchCV over the configured pipelines with a single predefined split, writes `gridsearch_results.csv`, and then re-runs the best pipeline while varying n, projection method, and anchor selection algorithm.

Outputs are isolated under `outputs/<backend>_embeddings/<experiment>/<timestamp>/` (and under `outputs/baseline_features/` for Musif); each embedding backend also has its own cache under `cache/embeddings/<backend>/`. ARIA keeps its existing cache under `data/<dataset>/aria_embeddings/`.

Select a backend with the `FEATURE_BACKEND` environment variable or the explicit `feature_backend` argument on the shared runner. Musif remains independently runnable with `FEATURE_BACKEND=musif`.

## Models and benchmark matrix

- **Aria**: `aria` global MIDI embeddings.
- **CLaMP 3 C2**: official symbolic checkpoint and MTF preprocessing; `clamp3_c2` backend.
- **Adversarial-MidiBERT**: official Octuple tokenizer and pretrained weights; `adversarial_midibert` backend. Inputs whose time signature changes inside a bar (the official converter rejects them, e.g. scores with an anacrusis) close the short bar at the change and start the new signature at position 0; the official 256-bar vocabulary limit still applies.
- **MIDI-GPT yellow_medium**: official MIDI-GPT tokenizer and completed checkpoint; install the optional `models` extra and select `midigpt`. Bars whose time signature is missing from the checkpoint vocabulary (the official encoder rejects them) are reduced by gcd, else mapped to the nearest supported bar length, with `beat_length` following the mapped signature.
Install MIDI-GPT support with:

```bash
uv sync --extra models
```


The default command attempts the full matrix in order. It returns a failure summary if any backend/experiment pair fails.

```bash
uv run python scripts/run_embedding_benchmark.py
```

Because `beat` is not yet identifiable, use the four available backends explicitly:

```bash
uv run python scripts/run_embedding_benchmark.py --backend aria clamp3_c2 adversarial_midibert midigpt
```

Use the same backend selection when generating reports:

```bash
uv run python scripts/report_embedding_benchmark.py --backend aria clamp3_c2 adversarial_midibert midigpt
```


Run Musif independently with `uv run python scripts/run_embedding_benchmark.py --backend musif`. The optional `--experiment` and `--backend` arguments restrict a run.

After a complete matrix has finished, produce per-experiment TSV tables and PNG plots (composer datasets remain in separate experiment reports):

```bash
uv run python scripts/report_embedding_benchmark.py
```

Reports include exact source-run paths, Spearman means and 95% confidence intervals for axis experiments, and class plus balanced-class accuracy for sentiment prediction. Report generation refuses to write unless every requested experiment/backend result is present.

The `beat` backend remains unavailable: the task brief identifies “BEAT piano weights” but neither the project nor published model repositories provide an identifiable model/checkpoint/API. No substitute model is used.

## Troubleshooting

### Linux i386 runtime compatibility for Humdrum binaries

If your Humdrum binaries are 32-bit (`i386`) and you are on a 64-bit Linux
system, install 32-bit runtime compatibility.

Debian/Ubuntu:

```bash
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install libc6:i386 libstdc++6:i386 zlib1g:i386
```

Fedora:

```bash
sudo dnf install glibc.i686 libstdc++.i686 zlib.i686
```

Arch Linux:

```bash
sudo pacman -Syu
sudo pacman -S lib32-glibc lib32-gcc-libs lib32-zlib
```
