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

Musif is a separate submodule and its dependencies are not installed by `uv sync`. After initializing the submodule, install its declared dependencies into the project environment:

```bash
uv pip install -e ./musif
```

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

- **Aria**: global MIDI embeddings from the `aria` submodule.
- **BEAT piano**: `beat` backend using the official BEAT-code submodule and piano checkpoint.
- **CLaMP3 C2**: `clamp3_c2` backend using official M3 symbolic code from the `aria` submodule.
- **Adversarial-MidiBERT**: `adversarial_midibert` backend; loads the official model from its submodule and uses a small adapter for MIDI-to-Octuple conversion.
- **MIDI-GPT yellow_medium**: `midigpt` backend; install the optional `models` extra.

Initialize the source submodules. MIDI-GPT users also install its optional dependencies:

```bash
git submodule update --init --recursive
uv sync --extra models
```

### BEAT setup

BEAT's Python dependencies are not part of the default project environment. Install the submodule's official requirements, then extract the supplied checkpoint archive:

```bash
git submodule update --init --recursive
uv sync
uv pip install -r BEAT-code/requirements.txt
tar -xzf BEAT-checkpoints.tar.gz checkpoints/piano/backbone.pt
uv run python scripts/run_embedding_benchmark.py --backend beat
```

Set `BEAT_CODE_PATH` or `BEAT_CHECKPOINT_PATH` only if the checkout or checkpoint is elsewhere. `BEAT_DEVICE` optionally selects inference hardware; the default is CUDA when available, otherwise CPU.

The compatibility adjustments affect a small part of the current benchmark inputs: of 643 successfully converted source MIDIs, the Adversarial-MidiBERT short-bar rule handles 23 files (32 signature changes), and MIDI-GPT maps unsupported meters in 4 files (147 track/bar records). Three composer files fail during MIDI conversion before either encoder runs. The 2,200 generated pitch inputs use 4/4 and trigger neither adjustment. MIDI-GPT's mapping updates `beat_length` to the mapped meter; BEAT's tick rescaling is only a fallback for PrettyMIDI's allocation guard and can shift events by at most half a new tick on a temporary copy.


The default command attempts the full matrix in order. It returns a failure summary if any backend/experiment pair fails.

```bash
uv run python scripts/run_embedding_benchmark.py
```

Run the benchmark's full default matrix, or choose any available embedding backends explicitly:

```bash
uv run python scripts/run_embedding_benchmark.py --backend aria beat clamp3_c2 adversarial_midibert midigpt
```

Use the same backend selection when generating reports:

```bash
uv run python scripts/report_embedding_benchmark.py --backend aria beat clamp3_c2 adversarial_midibert midigpt
```

Run Musif independently with `uv run python scripts/run_embedding_benchmark.py --backend musif`. The optional `--experiment` and `--backend` arguments restrict a run.

After a complete matrix has finished, generate per-experiment metric reports and aggregate multi-backend plots:

```bash
uv run python scripts/report_embedding_benchmark.py
```

Use `--backend aria beat clamp3_c2 adversarial_midibert midigpt musif` to include Musif when its runs cover the requested experiments. Each task report includes an overlay of out-of-fold predictions versus real values, with the selected anchor count and grid configuration in the figure notes, plus an n-sweep comparison split by selection algorithm and projection mode; sweep plots use a fixed 0-1 Spearman axis. `plot_coverage.tsv` records missing per-run prediction/grid/sweep artifacts; only runs with those artifacts appear in the corresponding plots.

For a Musif comparison on song-level tasks:

```bash
uv run python scripts/report_embedding_benchmark.py --backend aria beat clamp3_c2 adversarial_midibert midigpt musif --experiment arousal_axis valence_axis composer_axis_asap composer_axis_ewld composer_axis_music21
```

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
