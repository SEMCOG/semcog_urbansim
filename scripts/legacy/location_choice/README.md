# Estimation workspace

Operational estimation archives live outside the repository under
`/home/da/RDF2055/d_drive/estimation/`, organized by model family and
Eastern-time dated run folder.

Only `variables_2050.py` remains here; the estimation code has moved:

- **REPM** — now `repm/` at the repository root. Run training from the root
  with `python -m repm.xgb_training`. Its output default remains
  `/home/da/RDF2055/d_drive/estimation/REPM`; it never writes a production
  model into this repository unless that path is explicitly configured.
- **Location choice** — now `scripts/legacy/location_choice/`. The MNL/ARD
  estimation stack is superseded by the PyTorch LCMs loaded from
  `input_paths.HLCM_MODEL_DIR` / `ELCM_MODEL_DIR`.

Simulation configuration and deployed model artifacts remain under `configs/`.
