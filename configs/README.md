# Configs

This folder contains experiment presets for the main PyTorch training scripts.

Typical usage:

```bash
python run_ultranerf.py --config configs/config_base_nerf.txt --expname my_run
python run_barf.py --config configs/config_barf.txt
```

Common groups:

- `config_base_nerf.txt`
  Baseline ultrasound rendering.

- `config_barf.txt`, `config_base_barf.txt`
  Pose-refinement experiments.

- `config_confmap*.txt`
  Confidence-map variants.

- `config_liver*.txt`, `config_patient*.txt`, `config_spine_phantom.txt`
  Dataset-specific presets.

These files are flat `configargparse` config files consumed by the `run_*.py`
entry points.
