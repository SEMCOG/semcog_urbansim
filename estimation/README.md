# Estimation workspace

This directory contains code, configuration, and versioned model artifacts used
to estimate SEMCOG forecast models. Operational estimation archives are stored
outside the repository under `/home/da/RDF2055/d_drive/estimation/`, organized
by model family and Eastern-time dated run folder.

- `repm/` — real-estate price model training, diagnostics, specifications, and
  XGBoost model artifacts.
- `location_choice/` — household and employment location-choice estimation
  programs and their configuration files.

Run REPM training from the repository root with
`python -m estimation.repm.xgb_training`. Its output default remains
`/home/da/RDF2055/d_drive/estimation/REPM`; it never writes a new production
model into this repository unless that path is explicitly configured.
