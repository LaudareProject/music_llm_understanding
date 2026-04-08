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

Musif features are stored under `data/musif/<dataset>/features.csv` so that every experiment can reuse the same CSV per dataset. Run the helper script before invoking any of the `run_all_musif.py` scripts:

```bash
uv run python scripts/extract_musif_features.py --dataset vgmidi
```

Use `--force` to regenerate an existing CSV.

## Experiments

### Composer chronology:

```bash
uv run python experiments/composer_axis/prepare_tsv.py
uv run python experiments/composer_axis/run_all.py
```

### Pitch axis C1-B6:

```bash
uv run python experiments/pitch_axis/prepare_tsv.py
uv run python experiments/pitch_axis/run_all.py
```

`prepare_tsv.py` builds `experiments/pitch_axis/notes.tsv` with note-level covariates (`duration_ticks`, `velocity`) used for control analyses.

### Best-model selection metric

All composer/pitch experiment runners accept `BEST_METRIC` to define what "best" means during combo/mapping/anchor selection.

- Supported values: `spearman` (default), `l1`, `pearson`, `r2`
- Example:

```bash
BEST_METRIC=l1 uv run python experiments/composer_axis/run_all.py
BEST_METRIC=r2 uv run python experiments/pitch_axis/run_all.py
```

The selected strategy is also used for covariate-control artifacts (`best_strategy_covariate_pearson_heatmap.png`, `best_strategy_covariate_spearman_heatmap.png`, `best_strategy_covariate_correlations.tsv`).

Outputs are written under `outputs/experiments/<experiment>/<timestamp>/`.

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
