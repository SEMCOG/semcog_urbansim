# Outputs

## Run HDF5 File

**Location:** `runs/runNNN.h5`

Run number increments automatically via `urbansim.utils.misc.get_run_number()`. The number is padded (e.g., `run0001.h5`, `run1365.h5`).

A companion directory `runs/runNNN/` is created alongside the HDF5 file and contains:
- `run_config.yaml` — snapshot of all configuration settings at run start (model paths, scenario toggles, git branch, commit hash)

### HDF5 Structure

```
runNNN.h5
├── /base/                     ← base-year snapshots of all reference tables
│   ├── jobs
│   ├── jobs_2019
│   ├── base_job_space
│   ├── employment_sectors
│   ├── annual_relocation_rates_for_jobs
│   ├── households
│   ├── persons
│   ├── buildings
│   ├── pseudo_building_2020
│   ├── parcels
│   ├── zones
│   ├── semmcds
│   ├── counties
│   ├── large_areas
│   ├── building_types
│   ├── zoning
│   ├── annual_employment_control_totals
│   ├── annual_household_control_totals
│   ├── group_quarters
│   ├── group_quarters_households
│   ├── group_quarters_control_totals
│   ├── events_addition
│   ├── events_deletion
│   ├── refiner_events
│   └── ... (other base tables)
│
├── /2021/                     ← end-of-year 2021 state
│   ├── buildings
│   ├── jobs
│   ├── base_job_space
│   ├── parcels
│   ├── households
│   ├── persons
│   ├── group_quarters
│   ├── dropped_buildings
│   └── bg_hh_increase
│
├── /2022/                     ← same structure
│   └── ...
│
...
└── /2050/
    └── ...
```

The `/base/` group is written once. Year groups are written at the end of each simulated year. Tables are stored compressed (`compress=True`).

---

## Output Indicators

After the simulation loop completes, `output_indicators.main()` reads the HDF5 and computes aggregated indicators at **5-year intervals** (configurable via `indicator_spacing`).

### What Is Computed

Indicators are computed at multiple geographic levels:

| Geography | Description |
|---|---|
| Large area | 8 modeling zones |
| MCD / City | Municipal entities |
| TAZ / Zone | Traffic Analysis Zones |
| Legislative district | State house and senate districts |

For each geography × year combination, indicators include:

**Households & Population:**
- Total households
- Households by income group
- Households by size, workers, children
- Population totals
- Group quarters population

**Employment:**
- Total jobs
- Jobs by sector
- Home-based vs. non-home-based employment

**Housing:**
- Total residential units
- Vacant units and vacancy rate
- New units added
- Units by building type

### Upload Destinations

| Destination | Details |
|---|---|
| **PostgreSQL** | `plannerprojection:5432/land` — `whatnots` table. Enables internal SEMCOG analysis and dashboard queries. |
| **CartoDB** | Public interactive map at `maps.semcog.org/forecast/`. Uploaded via `cartoframes`. |

Upload to CartoDB is controlled by the `upload_to_carto` flag in `test_forecast_2050.py` (default `True`).

---

## Run Log

**Location:** `runs/run_stdout/simulation_log.txt`

When launched with `nohup python test_forecast_2050.py >> runs/run_stdout/simulation_log.txt 2>&1 &`, all stdout and stderr is captured. Key events logged by `utils.run_log()`:
- Run start time
- Input HDF path
- Random seed
- Year-by-year progress
- Total run time at completion

---

## `dropped_buildings`

Tracks buildings demolished during the simulation. Updated each year by `scheduled_demolition_events` and `random_demolition_events`. Written to the output HDF5 each year.

| Column | Description |
|---|---|
| `building_id` | ID of demolished building |
| `year` | Year of demolition |
| `parcel_id` | Parcel location |
| `building_type_id` | Type of demolished building |

---

## `bg_hh_increase`

Tracks block group household growth over a rolling 3-year window. Used internally by `mcd_hu_sampling` and also written to output for diagnostics.

| Column | Description |
|---|---|
| `GEOID` | Census block group GEOID (index) |
| `occupied` | Current year occupied housing units |
| `previous_occupied` | 3 years ago occupied units |
| `occupied_year_minus_1/2/3` | Rolling history |

---

## Reading Output HDF5

To read a specific year's data in Python:

```python
import pandas as pd

store = pd.HDFStore('runs/run1365.h5', 'r')

# Get 2035 households
households_2035 = store['/2035/households']

# Get 2050 buildings
buildings_2050 = store['/2050/buildings']

# Get base year jobs
jobs_base = store['/base/jobs']

store.close()
```

To list all available keys:
```python
print(store.keys())
```

---

## Indicator Spacing

The `indicator_spacing` parameter in `test_forecast_2050.py` (default: 5) controls how frequently indicators are computed and uploaded. With `spacing=5`, indicators are generated for years 2020, 2025, 2030, 2035, 2040, 2045, 2050.

The full year-by-year snapshots are always written to the HDF5 regardless of indicator spacing.
