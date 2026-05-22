# SEMCOG Forecast Input Data Wiki

## Purpose

This wiki is the reference guide for data developers who produce and maintain the input data for the SEMCOG UrbanSim land use simulation model. It documents every table in the **model input HDF5 file**, organized by the team or data domain responsible for it.

The HDF5 is assembled by `forecast_data_input/main.py`, which draws from three sources:
- **PostgreSQL** — parcels, buildings, zoning, events, demolitions, crime rates
- **CSV / Excel files** — control totals, rate tables, synthetic population, jobs, amenities
- **Accessibility CSVs** — pre-computed walk/bike/drive indicators by parcel

File paths for each source are configured in `config/files.yaml` (CSV/Excel) and `config/sql.yaml` (SQL queries). Update these configs when paths change — do not hard-code paths in scripts.

---

## Data Domain Map

Each domain is owned by a different team. Use this table to find who to contact and where to go for each piece of data.

| Domain | Tables | Primary Source |
|---|---|---|
| [Land Use](domains/land-use.md) | `parcels`, `buildings`, `zoning`, `land_use_types`, `building_types`, `multi_parcel_buildings`, `pseudo_building_2020` | PostgreSQL (parcel/building database) |
| [Geography](domains/geography.md) | `zones`, `semmcds`, `counties`, `large_areas` | PostgreSQL (boundary tables) / constant |
| [Demographics](domains/demographics.md) | `households`, `persons`, `annual_household_control_totals`, `remi_pop_total`, relocation rates, `target_vacancies`, `mcd_total`, group quarters tables | Population synthesis, REMI, CSV files |
| [Employment](domains/employment.md) | `jobs`, `annual_employment_control_totals`, `annual_relocation_rates_for_jobs`, `building_sqft_per_job`, `employment_sectors`, `employed_workers_rate` | CSV files, REMI |
| [Development & Context](domains/development-context.md) | `events_addition`, `events_deletion`, `refiner_events`, `demolition_rates`, `landmark_worksites`, `schools`, `crime_rates` | PostgreSQL (event/demo tables), CSV |
| [Street Networks](domains/networks.md) | Walk and drive node/edge tables | `semcog_2050_networks.h5` (separate file) |
| [Accessibility](domains/accessibility.md) | `accessibility_walk/bike/drive_indicator_by_parcel`, `transit_stops`, `poi` | Accessibility analysis CSVs |
| [Travel Data](domains/travel.md) | `travel_data`, `travel_data_2030` | TransCAD OMX → CSV |

---

## How the HDF5 Is Assembled

```
PostgreSQL database
    parcels, buildings, zoning, zones, events, demolitions, crime rates, ...
         │
CSV / Excel files (paths in config/files.yaml)
    control totals, rates, jobs, GQ data, amenities, accessibility
         │
         ▼
    forecast_data_input/main.py
    load  →  transform  →  validate  →  export
         │
         ▼
    output HDF5  →  deployed to model input directory
```

Key config files:
- `config/sql.yaml` — SQL queries for PostgreSQL tables
- `config/files.yaml` — paths to CSV/Excel inputs
- `config/transform.yaml` — column rename mappings
- `config/data_structure.yaml` — dtype casting rules

---

## Table Count

The model input HDF contains **40+ tables**. See the [Table Catalog](reference/table-catalog.md) for a complete alphabetical list.
