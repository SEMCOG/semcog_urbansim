# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the SEMCOG (Southeast Michigan Council of Governments) UrbanSim implementation — an agent-based urban land use simulation that forecasts household, employment, and building changes from base year 2020 to 2050 (or 2055 on the `forecast_2055` branch). The simulation runs year-by-year using the `orca` pipeline orchestrator.

## Environment Setup

```bash
micromamba activate forecast
```

All Python commands should be run in this environment from `/mnt/semcog_urbansim`.

## Running the Simulation

Primary simulation entry point:
```bash
# Run with full logging in background
nohup python test_forecast_2050.py >> runs/run_stdout/simulation_log.txt 2>&1 &
```

Run configuration is set at the top of `test_forecast_2050.py`:
- `base_year`, `final_year`: simulation year range
- `hlcm_model_path`, `elcm_model_path`: paths to trained PyTorch model files (on mounted network drive `/mnt/hgfs/RDF2050/`)
- `ENABLE_SCENARIO`: toggle for alternative scenario controls
- `use_checkpoint` + `runnum_to_resume`: resume from a specific run's last completed year

Output HDF5 files are written to `runs/runNNN.h5`. Run number increments automatically via `urbansim.utils.misc.get_run_number()`.

## Architecture

### Core Pipeline Files

- **`test_forecast_2050.py`** — Main simulation script. Configures injectables, then calls `orca.run([...], iter_vars=range(run_start+1, final_year+1), ...)`. Each element in the list is a named orca step defined in `models.py`.
- **`models.py`** — Defines all orca simulation steps (`@orca.step`). On import, also loads HLCM/ELCM PyTorch models and registers them as injectables. The file is large (~3000+ lines) and contains every model in the pipeline.
- **`dataset.py`** — Loads all HDF5 tables from `data/` into orca at startup (via `orca.get_injectable("store")`). Also defines derived orca tables and columns used throughout the simulation.
- **`assumptions.py`** — Registers core orca injectables: `year`, `building_type_map`, `transcad_available`, and other constants used across models.
- **`variables/`** — orca column definitions organized by entity type: `variables_parcel.py`, `variables_building.py`, `variables_zone.py`, `variables_demographic.py`, `variables_employment.py`, `variables_access.py`, `variables_tract.py`. All are imported via `variables/__init__.py`.
- **`lcm_utils.py`** — Utilities for loading and running PyTorch-based Location Choice Models (HLCM and ELCM), including `load_torch_lcm()`, `load_hlcm_model_configs_from_path()`, and `load_elcm_model_configs_from_path()`.
- **`utils.py`** — General utilities: `get_run_filename()`, `hedonic_simulate()`, `lcm_simulate()`, `SimulationChoiceModel`, `run_log()`, transition/relocation helpers.
- **`output_indicators.py`** — Post-simulation aggregation. Called at end of run to compute indicators and (optionally) upload to Carto via `output_indicators.main(data_out, ...)`.

### Annual Simulation Step Order

Each simulated year runs these steps in sequence (see `test_forecast_2050.py`):
1. `build_networks_2050`, `neighborhood_vars`, `update_taz_hlcm_trend`
2. `cache_hh_seeds` (first year only), demolition/development events, `refiner`
3. `households_transition`, `fix_lpr`, `households_relocation_2050`
4. `jobs_transition`, `drop_pseudo_buildings`
5. `feasibility`, `residential_developer`, `non_residential_developer`, `update_sp_filter`
6. REPM steps (`repm_step_names` injectable — XGBoost-based price models)
7. `refine_housing_units`, `mcd_hu_sampling`
8. HLCM steps (`hlcm_step_names` injectable — PyTorch NN household location choice)
9. ELCM steps (`elcm_step_names` injectable — PyTorch NN employment location choice)
10. `jobs_scaling_model`, `gq_pop_scaling_model`, `update_bg_hh_increase`

### Model Types

**HLCM (Household Location Choice Model)**: PyTorch neural networks segmented by `large_area × {has_children, no_children} × {with_seniors, without_seniors} × {lowinc, midinc, highinc}`. Model `.pt` files loaded from the configured `hlcm_model_path`.

**ELCM (Employment Location Choice Model)**: PyTorch neural networks segmented by `large_area × employment_sector × {homebased, nonhomebased}`. ELCM sectors run in a specific order defined by `elcm_sector_order` in `models.py`.

**REPM (Real Estate Price Model)**: XGBoost-based hedonic price models for residential (`rsh`) and non-residential (`nrh`) building types. Config files in `configs/repm/` and `configs/repm_2050/`.

**Developer Models**: Pro-forma-based feasibility analysis (`feasibility` step) followed by `residential_developer` and `non_residential_developer`. Config in `configs/proforma.yaml`, `configs/res_developer.yaml`, `configs/nonres_developer.yaml`.

**Refiner**: Applies policy-based additions/deletions from `refiner_events` table in the HDF5 data store.

### Configuration Files (`configs/`)

- `yaml_configs_elcm_hlcm.yaml` — Lists all HLCM and ELCM `.pt` model filenames; drives which models are loaded
- `available_networks_2050.yaml` — Pandana network configuration for accessibility calculations
- `hlcm_constraints.yaml`, `elcm_constraints.yaml` — LCM simulation constraints
- `res_repm_constraints.yaml`, `nonres_repm_constraints.yaml` — REPM constraints
- `mcd_hu_sampling.yaml` — MCD-level housing unit sampling config
- `configs/repm_2050/`, `configs/hlcm_2050/`, `configs/elcm_2050/` — Per-segment YAML model configs

### Data

- **HDF5 store**: primary data in `data/` (loaded as orca injectable `store`). Tables include `households`, `buildings`, `parcels`, `jobs`, `persons`, `zones`, `semmcds`, etc.
- **Run outputs**: `runs/runNNN.h5` — HDF5 files with year-by-year snapshots of `out_run_tables`
- **Indicator outputs**: per-run subdirectories adjacent to the HDF5 run file
- **External model files**: mounted at `/mnt/hgfs/RDF2050/estimation/models/` (PyTorch `.pt` files and associated metadata)

### Checkpoint / Resume

Set in `test_forecast_2050.py`:
```python
orca.add_injectable('use_checkpoint', True)
orca.add_injectable('runnum_to_resume', 'run1360.h5')
```
When enabled, the simulation starts from the last completed year of the specified run. After completion, results from the old run are merged into the new output HDF5.

### Estimation Scripts

- `HLCM_estimation.py`, `ELCM_estimation.py` — Model estimation scripts
- `REPM_feature_selection.py` — REPM variable selection
- `estimation_variables_2050.py` — Variable definitions used during estimation
- `notebooks/` — Jupyter-style analysis notebooks (stored as `.py` files)
