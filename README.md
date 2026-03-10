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
