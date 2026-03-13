# Symbolic Music Latent Experiments

This repo provides a lightweight experiment runner to extract latent representations from symbolic music inputs using pretrained models.

## Quickstart

1. Create a YAML config (see `configs/example.yaml`).
2. Install deps with uv:

```bash
uv sync
```

3. Run the experiment:

```bash
uv run python scripts/run_experiment.py configs/example.yaml
```

If no config is provided, the runner executes all `configs/*.ya?ml` files except `example.yaml`.

Outputs are stored under `outputs/<experiment_name>/<timestamp>/` with per-query/model latent files and metadata.

## Models

- **Aria embedding**: uses `aria` to generate global MIDI embeddings.
- **PhraseVAE**: uses `PhraseLDM_code` to encode MIDI into REMI-z latents.

## Inputs

- `abc`: ABC string content
- `midi_path`: path to a MIDI file
- `musicxml_path`: path to a MusicXML file
- `mei_path`: path to a MEI file
- `custom`: space-separated tokens; use `-` inside a token to form chords (e.g. `C-E-G`)

`abc`, `custom`, and `mei_path` inputs are rendered to MIDI through `music21`.

`musicxml_path` is converted to MIDI through Humdrum tools (`xml2hum`, `transpose`, `hum2mid`) with pitch-centering (target average MIDI pitch 60). If Humdrum conversion is unavailable/fails, the pipeline falls back to `music21` conversion.

## Tests

```bash
uv run pytest
```

## EWLD Tonality Experiment

To build an EWLD-based experiment config that balances songs per estimated tonality
(Krumhansl-Schmuckler via Humdrum `keycor` profile voting), validates
key-signature consistency, and creates both average and median tonality groups:

Place Humdrum binaries in `bin/` and make them executable:

```bash
chmod +x bin/xml2hum bin/hum2mid bin/keycor
```

EWLD preprocessing also normalizes pitch register before encoding:
for each song, it computes the average MIDI pitch, then applies a chromatic
transposition so the average is centered at MIDI pitch 60
(rounded semitone shift).

Then run:

```bash
uv run python scripts/prepare_ewld_tonality_experiment.py \
  --dataset-root EWLD \
  --target-per-tonality 50 \
  --workers 8 \
  --output-config configs/ewld_tonality.yaml
```

Then run it with:

```bash
uv run python scripts/run_experiment.py configs/ewld_tonality.yaml
```

## Supervised Vector-Feature Experiment

You can also run a supervised experiment directly from latent vectors.
Each query can optionally include a `target` (or `label`) value, and the
runner can fit:

- regression: elastic-net linear regression
- classification: elastic-net logistic regression

Pipeline:

1. build vector features (`embedding`, `norm`, `polar`)
2. standardize features
3. PCA retain 95% variance
4. train elastic-net model
5. use 5-fold CV if samples > 50, else leave-one-out CV
6. project weights back to original features and plot sorted importance

Example config snippet:

```yaml
queries:
  - id: "q1"
    midi_path: "a.mid"
    target: 0.2
  - id: "q2"
    midi_path: "b.mid"
    target: 0.8

supervised:
  task: "regression" # or "classification"
  features: ["embedding", "descriptive", "polar"]
  l1_ratio: 0.5
  alpha: 1.0
  max_iter: 5000
  top_k: 50
```

`descriptive` includes location/scale, shape, norm/energy, sparsity/concentration,
and sign-structure descriptors (mean/median/std/variance/IQR/MAD/min/max/range,
skewness/kurtosis/trimmed mean/quantiles, L1/L2/Linf/RMS/energy,
near-zero fraction/Gini-like concentration/binned entropy, positive/negative fractions).

Outputs are written to:

- `supervised/<model>/supervised_report.json`
- `supervised/<model>/feature_importance.png`

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
