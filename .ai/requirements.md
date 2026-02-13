Provide a YAML-driven experiment interface with multiple queries per experiment.

Each experiment must:
- Download required model checkpoints on demand.
- Convert inputs to model-compatible formats.
- Extract latent representations per model.
- Save latents and metadata in time-stamped output directories to keep experiments isolated.
