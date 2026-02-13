Project focus: large models for symbolic music generation and understanding, centered on using pretrained models to extract latent representations from symbolic music inputs.

Primary models: Aria (embedding model) and PhraseLDM/PhraseVAE (phrase-level VAE) provided via git submodules.

Inputs to support: ABC strings, MIDI files, MusicXML files, MEI files.

Core workflow: convert inputs to model-compatible representations, run models to obtain latents, and save outputs with metadata in timestamped experiment folders.

Implementation stance: keep code simple and modular; avoid compatibility fallbacks for alternate module paths when the environment is fixed.
