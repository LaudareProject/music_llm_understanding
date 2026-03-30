# Symbolic Music Latent Experiments

This repo provides a lightweight experiment runner to extract latent representations from symbolic music inputs using pretrained models.

## Quickstart

1. Create a YAML config (see `configs/example.yaml`).
2. Install deps with uv:

```bash
uv sync
```

3. Run the experiments.

## Experiments

### Composer chronology:

```bash
uv run python experiments/composer_axis/prepare_tsv.py
uv run python experiments/composer_axis/run.py
```

### Pitch axis C1-B6:

```bash
uv run python experiments/pitch_axis/prepare_tsv.py
uv run python experiments/pitch_axis/run.py
```

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
