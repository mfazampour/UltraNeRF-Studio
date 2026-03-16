# Tests

This folder contains the PyTorch test suite for the repository.

The tests focus on:

- mock-data validation for dataset loading
- core model behavior
- pose and Lie utilities
- image-processing helpers
- CPU-safe import checks for rendering utilities
- PyTorch-only regression coverage for the legacy port

Run the suite with:

```bash
python -m pytest -q
```

Within this repository, the preferred environment is the Conda environment
created under `miniconda3/envs/ultranerf`.
