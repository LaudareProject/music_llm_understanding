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

Outputs are written under `outputs/aria_embeddings/<experiment>/<timestamp>/` or `outputs/baseline_features/<experiment>/<timestamp>/` based on `FEATURE_BACKEND`.

To switch to musif features, use `settings.FEATURE_BACKEND='musif'`

## Models

- **Aria embedding**: uses `aria` to generate global MIDI embeddings.

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
