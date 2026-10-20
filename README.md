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
- **BEAT piano**: official `MIDIToPianoNPZ` preprocessing, `PianoTokenizer`, and inference loader; `beat` backend. Uses the piano checkpoint, pools final-layer content-token states into a 768-dimensional vector, and windows at 2048 tokens. For inputs beyond PrettyMIDI's tick-allocation guard, it scales ticks on a temporary copy while preserving event times to the new PPQ resolution; the source MIDI is unchanged.
- **CLaMP 3 C2**: project-owned symbolic inference source, separate from the `aria` submodule; it uses the official C2 checkpoint and MTF preprocessing through `clamp3_c2`.
- **Adversarial-MidiBERT**: official Octuple tokenizer and pretrained weights; `adversarial_midibert` backend. Inputs whose time signature changes inside a bar (the official converter rejects them, e.g. scores with an anacrusis) close the short bar at the change and start the new signature at position 0; the official 256-bar vocabulary limit still applies.
- **MIDI-GPT yellow_medium**: official MIDI-GPT tokenizer and completed checkpoint; install the optional `models` extra and select `midigpt`. Bars whose time signature is missing from the checkpoint vocabulary (the official encoder rejects them) are reduced by gcd, else mapped to the nearest supported bar length, with `beat_length` following the mapped signature.

Install MIDI-GPT support with:

```bash
uv sync --extra models
```

### BEAT setup

Install the adapter's Python prerequisite and both optional model extras:

```bash
uv sync --extra beat --extra models
git clone https://github.com/Lekai-Qian/BEAT-code.git
```

Install the official BEAT-code runtime requirements in the same environment. Set `BEAT_CODE_PATH` if the checkout is not at `./BEAT-code`. Extract the supplied checkpoint archive at the project root so that `checkpoints/piano/backbone.pt` exists, or set `BEAT_CHECKPOINT_PATH` to that file. The loader uses the official piano inference loader to support the released older vocabulary checkpoint; it does not download or copy weights. `BEAT_DEVICE` optionally selects the inference device (defaults to CUDA when available, otherwise CPU).

```bash
tar -xzf BEAT-checkpoints.tar.gz checkpoints/piano/backbone.pt
export BEAT_CODE_PATH="$PWD/BEAT-code"
export BEAT_CHECKPOINT_PATH="$PWD/checkpoints/piano/backbone.pt"
uv run python scripts/run_embedding_benchmark.py --backend beat
```


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

After a complete matrix has finished, produce per-experiment TSV tables and PNG plots (composer datasets remain in separate experiment reports):

```bash
uv run python scripts/report_embedding_benchmark.py
```

Reports include exact source-run paths, Spearman means and 95% confidence intervals for axis experiments, and class plus balanced-class accuracy for sentiment prediction. Report generation refuses to write unless every requested experiment/backend result is present.

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
