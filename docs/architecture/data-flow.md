# Data Flow

This page traces how data moves through the model — from raw inputs, through each simulation year, to final outputs.

---

## Overview

```
External Sources
      │
      ▼
Input HDF5 Store ──────────────────────────────────────────┐
      │                                                     │
      ▼                                                     │
[dataset.py] loads tables into orca                        │
[assumptions.py] registers configuration                   │
      │                                                     │
      ▼                                                     │
Annual Loop (2021 → 2050)                                  │
  │                                                         │
  ├── Control totals (REMI) ──► households_transition       │
  │                              jobs_transition            │
  │                                                         │
  ├── Event tables ─────────► scheduled_demolition_events   │
  │                            scheduled_development_events │
  │                            refiner                      │
  │                                                         │
  ├── Accessibility CSVs ───► build_networks_2050           │
  │                            neighborhood_vars            │
  │                                                         │
  ├── PyTorch .pt files ────► hlcm_step_names               │
  │                            elcm_step_names              │
  │                                                         │
  ├── XGBoost .pkl files ───► repm_step_names               │
  │                                                         │
  ▼                                                         │
Year Snapshot ──────────────────────────────────────────────┘
      │
      ▼
runs/runNNN.h5 (HDF5 with /base/, /2021/, /2022/, ..., /2050/)
      │
      ▼
[output_indicators.py]
      │
      ├──► PostgreSQL (plannerprojection:5432/land)
      └──► CartoDB (maps.semcog.org/forecast/)
```

---

## Input Data Sources

### Primary HDF5 Store

**Path:** `/mnt/hgfs/urbansim/RDF2050/model_inputs/base_hdf/forecast_data_input_031523.h5`

The single most important input. Contains all base-year entity tables and all control/reference tables needed to run the simulation. Loaded once at startup by `dataset.py` via `assumptions.verify()`.

See [Input Data](../data/inputs.md) for the full table list.

### External CSV Tables (loaded at runtime)

| Table | Path | Used By |
|---|---|---|
| `accessibility_walk_indicator_by_parcel` | `.../Accessibility/access_to_core_2024/outputs_model/indicators/walk/walk_indicators_by_parcel_20251111.csv` | `variables_access.py` |
| `accessibility_bike_indicator_by_parcel` | `.../Accessibility/access_to_core_2024/outputs_model/indicators/bike/bike_indicators_by_parcel_20251111.csv` | `variables_access.py` |
| `accessibility_drive_indicator_by_parcel` | `.../Accessibility/access_to_core_2024/outputs_model/indicators/drive/drive_indicators_by_parcel_20251111.csv` | `variables_access.py` |
| `poi` | `.../RDF2055/model_inputs/base_tables/pois.csv` | Pandana network aggregations |
| `bg_hh_increase` | `data/ACS_HH_14_19_BG.csv` | `mcd_hu_sampling` |
| Travel survey | `/mnt/D/RDF2055/input_data/travel_survey/Full_Interim_Dataset_2026-03-04/` | `travel_survey_vars.py` |

### Trained Model Files

| Model Type | Path | Used By |
|---|---|---|
| HLCM PyTorch `.pt` | `/mnt/hgfs/RDF2050/estimation/models/models_survey_finetune/pts/` | `lcm_utils.load_torch_lcm()` |
| ELCM PyTorch `.pt` | `/mnt/hgfs/RDF2050/estimation/models/elcm_models_25May30/pts/` | `lcm_utils.load_torch_lcm()` |
| REPM XGBoost `.pkl` | `configs/repm_xgb/*/` | `models.py` REPM steps |

---

## Data Flow Within a Year

### 1. Control Totals → Entity Tables

```
annual_household_control_totals  ──► households_transition
                                          │
                                          ▼
                                  households table
                                  (rows added/removed,
                                   new rows have building_id = -1)

annual_employment_control_totals ──► jobs_transition
                                          │
                                          ▼
                                  jobs table
                                  (rows added/removed,
                                   new rows have building_id = -1)
```

### 2. Events → Buildings Table

```
events_addition    ──► scheduled_development_events
events_deletion    ──► scheduled_demolition_events
refiner_events     ──► refiner
                              │
                              ▼
                       buildings table
                       (rows added/removed)
```

### 3. Feasibility → New Buildings

```
buildings table (prices via sqft_price_res/nonres)
parcels table   (zoning, pct_undev)
proforma.yaml   (cost assumptions)
        │
        ▼
   feasibility step
        │
        ▼
  proposals table (parcel × form × FAR)
        │
        ▼
  residential_developer
  non_residential_developer
        │
        ▼
  buildings table (new rows appended)
```

### 4. REPM → Building Prices

```
buildings table (hedonic_id, building attributes)
parcels table   (location attributes)
neighborhood variables
accessibility variables
        │
        ▼
  [repm_* steps] (XGBoost models)
        │
        ▼
  buildings.sqft_price_res
  buildings.sqft_price_nonres
  (updated in place)
```

### 5. MCD Quota → HLCM Capacity

```
mcd_total table (target HH counts per MCD per year)
households table (current counts by MCD)
bg_hh_increase  (block group growth trend)
buildings table (vacancy, age, geoid)
        │
        ▼
  mcd_hu_sampling step
        │
        ▼
  buildings.mcd_model_quota
  (used as capacity cap in HLCM)
```

### 6. Unplaced Agents → Location Choice

```
households (building_id == -1) ──► HLCM models
                                        │
                                        ▼
                                  buildings
                                  (mcd_model_quota as capacity)
                                        │
                                        ▼
                                  households.building_id updated

jobs (building_id == -1) ──────► ELCM models
                                        │
                                        ▼
                                  buildings
                                  (jobs_non_home_based as capacity)
                                        │
                                        ▼
                                  jobs.building_id updated
```

---

## Output Data

### Run HDF5 File

**Path:** `runs/runNNN.h5`

Structure:
```
runNNN.h5
├── /base/
│   ├── jobs
│   ├── households
│   ├── buildings
│   ├── parcels
│   ├── persons
│   └── ... (all out_base_tables)
├── /2021/
│   ├── buildings
│   ├── jobs
│   ├── households
│   ├── persons
│   ├── parcels
│   ├── group_quarters
│   ├── dropped_buildings
│   └── bg_hh_increase
├── /2022/
│   └── ... (same structure)
...
└── /2050/
    └── ...
```

The `/base/` group contains a snapshot of all base tables (static reference data + base-year entity tables). Each year group contains the simulation state at the **end** of that year.

### Indicators

After the run, `output_indicators.py` reads the HDF5 and produces aggregated indicators at 5-year intervals (configurable via `indicator_spacing`). Indicators include:

- Household counts by geography and demographic segment
- Employment counts by geography and sector
- Housing unit counts and vacancy rates
- Population totals

These are uploaded to PostgreSQL (`plannerprojection:5432/land`) and CartoDB for the public forecast map.

---

## Table Relationships (Broadcasts)

Orca broadcasts define how columns can be joined across tables. The key relationships are:

```
parcels ──────────► buildings   (parcel_id → parcel_id)
buildings ────────► households  (building_id → building_id)
buildings ────────► jobs        (building_id → building_id)
households ───────► persons     (household_id → household_id)
zones ────────────► parcels     (zone_id → zone_id)
nodes_walk ───────► buildings   (nodeid_walk)
nodes_walk ───────► parcels     (nodeid_walk)
nodes_drv ────────► buildings   (nodeid_drv)
nodes_drv ────────► parcels     (nodeid_drv)
building_types ───► buildings   (building_type_id)
schools ──────────► parcels     (parcel_id)
```

This means, for example, that a building-level variable can reference a parcel column directly (e.g., `buildings.pct_undev`), and a household variable can reference a building column (e.g., `households.sqft_price_res`) without any explicit join code.
