# Symbolic Music Latent Experiments

This repo provides a lightweight experiment runner to extract latent representations from symbolic music inputs using pretrained models.

## Quickstart

1. Create a YAML config (see `configs/experiments/example.yaml`).
2. Install deps with uv:

```bash
uv sync
```

3. Run the experiment:

```bash
uv run python scripts/run_experiment.py configs/experiments/example.yaml
```

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

ABC/MusicXML/MEI inputs are converted to MIDI via `music21` before encoding.
Custom inputs are rendered to MIDI via `music21` with each token set to one quarter note.

## Tests

```bash
uv run pytest
```
