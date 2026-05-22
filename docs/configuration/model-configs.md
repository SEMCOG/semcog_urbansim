# Model Config Files

All YAML configuration files live in the `configs/` directory.

---

## `yaml_configs_elcm_hlcm.yaml`

The master list of LCM model files. Controls which HLCM and ELCM `.pt` models are loaded.

```yaml
hlcm:
  model_type: location_choice
  vacant_variable: vacant_residential_units
  config_filenames:
    - la161_has_children_lowinc_hlcm.pt
    - la161_has_children_midinc_hlcm.pt
    - ...
elcm:
  model_type: location_choice
  vacant_variable: job_spaces
  calibration: ...
  config_filenames:
    - la161_sector3_nonhomebased_elcm.pt
    - ...
```

This file is **auto-updated** at startup by `lcm_utils.load_hlcm_model_configs_from_path()` and `lcm_utils.load_elcm_model_configs_from_path()`, which scan the model directories and write the current list of `.pt` files. You should not need to edit this manually.

---

## `available_networks_2050.yaml`

Defines Pandana network configurations for the `build_networks_2050` step.

Specifies:
- Node and edge table paths for walk and drive networks
- Network variables to aggregate (e.g., jobs within 10 min walk)
- Aggregation radii (in meters or minutes)

---

## `hlcm_constraints.yaml`

Defines variable groups and expected coefficient signs for **HLCM estimation**. Used during model training, not simulation.

Structure:
```yaml
variable_groups:
  price:
    variables: [sqft_price_res, ...]
    expected_sign: negative
  accessibility:
    variables: [jobs_walk_cumulative_10min, ...]
    expected_sign: positive
```

---

## `elcm_constraints.yaml`

Same structure as `hlcm_constraints.yaml` but for employment location choice variable groups.

---

## `mcd_hu_sampling.yaml`

Configuration for the `mcd_hu_sampling` step:

```yaml
vacant_variable: vacant_residential_units
# Additional parameters controlling MCD quota computation
```

---

## `proforma.yaml`

Pro-forma cost assumptions for the developer feasibility model. See [Developer & Feasibility](../models/developer.md) for full details.

Key sections:
- `fars` — list of FAR values to test per parcel
- `uses` — building use types
- `costs` — construction cost per sq ft by use and height tier
- `parking_rates`, `parking_cost_d` — parking cost assumptions
- `profit_factor`, `cap_rate`, `interest_rate` — financial parameters

---

## `res_developer.yaml`

Controls the `residential_developer` step:
- Target vacancy rates by geography
- Maximum units per year per geography
- Which building forms to prefer

---

## `nonres_developer.yaml`

Controls the `non_residential_developer` step:
- Target vacancy rates for non-residential space
- Maximum sq ft by form per year

---

## `configs/repm_xgb/`

Contains one subdirectory per trained REPM model segment:

```
configs/repm_xgb/
├── res_hedonic_3_la3/        ← Residential, type 3, large area 3
│   ├── model.pkl             ← Trained XGBoost or Ridge model
│   └── metadata.pkl          ← Feature names, scaler, segment metadata
├── nonres_hedonic_23_la115/  ← Non-residential, type 23, large area 115
│   ├── model.pkl
│   └── metadata.pkl
└── grid_search/              ← Hyperparameter search artifacts (read-only)
```

Subdirectory names follow the pattern `{res|nonres}_{hedonic_id}_la{large_area_id}`.

---

## `configs/hlcm_2050/`

Per-segment YAML configuration files for HLCM model estimation. One file per model segment. Not used during simulation — used only during `HLCM_estimation.py`.

---

## `configs/elcm_2050/`

Per-segment YAML configuration files for ELCM estimation. Not used during simulation.

---

## `configs/data_structure.yaml`

Auto-generated at each run startup by `verify_data_structure.yaml_from_store()`. Documents the schema (table names, column names, dtypes) of the current HDF5 input. Useful for debugging data issues.

Do not edit manually — it is overwritten on each run.
