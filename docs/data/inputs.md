# Input Data

## Primary HDF5 Store

**Path:** `/mnt/hgfs/urbansim/RDF2050/model_inputs/base_hdf/forecast_data_input_031523.h5`

The primary input. A single HDF5 file containing all base-year entity tables and reference/control tables. Loaded once at startup by `assumptions.verify()` and registered as the `store` orca injectable.

The model auto-discovers the latest file in the base_hdf directory but pins to the specific filename above for reproducibility.

---

## Entity Tables (base-year agents)

These tables represent the state of the region in the base year (2020).

| Table | Description | Key Columns |
|---|---|---|
| `households` | All households in SE Michigan | `building_id`, `large_area_id`, `persons`, `workers`, `income`, `age_of_head`, `children`, `cars`, `race_id` |
| `persons` | All persons (linked to households) | `household_id`, `age`, `sex`, `race_id`, `worker`, `relate` |
| `jobs` | All jobs | `building_id`, `sector_id`, `home_based_status`, `large_area_id` |
| `buildings` | All buildings | `parcel_id`, `building_type_id`, `year_built`, `residential_units`, `non_residential_sqft`, `sqft_per_unit`, `stories`, `market_value` |
| `parcels` | All parcels (~1.9M) | `zone_id`, `city_id`, `county_id`, `census_bg_id`, `x`, `y`, `parcel_acres` |
| `pseudo_building_2020` | Temporary buildings for unplaced agents | Same schema as buildings; IDs > 90,000,000 |

---

## Geographic Reference Tables

| Table | Description |
|---|---|
| `zones` | Traffic Analysis Zones (TAZ) |
| `semmcds` | Municipal entities (cities, townships) |
| `counties` | 7 SE Michigan counties |
| `large_areas` | 8 modeling zones (roughly county-level) |
| `zoning` | Parcel-level zoning: `is_developable`, `percent_undev`, allowed building forms, max FAR |
| `land_use_types` | Land use type definitions |
| `building_types` | Building type definitions |
| `employment_sectors` | Employment sector definitions |
| `travel_data` | TAZ-level travel time/distance matrices (base year) |
| `travel_data_2030` | TAZ-level travel data for 2030 (future scenario) |

---

## Control Total Tables

These define the exogenous forecasts that drive household and employment growth.

| Table | Description | Segmentation |
|---|---|---|
| `annual_household_control_totals` | Target HH counts per year | `large_area_id`, `year`, demographics |
| `annual_employment_control_totals` | Target job counts per year | `large_area_id`, `year`, `sector_id`, `home_based_status` |
| `remi_pop_total` | Total population by large area and year (from REMI) | `large_area_id`, `year` |
| `mcd_total` | Target HH counts per MCD per year | `semmcd`, `year` |
| `group_quarters_control_totals` | Group quarters population targets | `large_area_id`, `gq_type`, `year` |

---

## Rate Tables

| Table | Description |
|---|---|
| `annual_relocation_rates_for_households` | Probability of relocation by age and income |
| `annual_relocation_rates_for_jobs` | Job relocation probability by sector |
| `demolition_rates` | Probabilistic demolition rates by building type and age |
| `target_vacancies` | Target residential vacancy rates by large area |
| `target_vacancies_mcd` | Target residential vacancy rates by MCD |
| `building_sqft_per_job` | Square footage required per job by building type |
| `employed_workers_rate` | Labor participation rates by demographics |
| `income_growth_rates` | Annual income growth rates by large area |

---

## Event & Policy Tables

| Table | Description |
|---|---|
| `events_addition` | Planned building additions by year |
| `events_deletion` | Planned building demolitions by year |
| `refiner_events` | Complex policy event interventions |
| `landmark_worksites` | Major employment centers (protected from simulation churn) |
| `multi_parcel_buildings` | Buildings spanning multiple parcels |

---

## Amenity & Context Tables

| Table | Description | Source |
|---|---|---|
| `transit_stops` | Fixed-route bus and rail stops | Transit agencies |
| `crime_rates` | Crime rates by geography | Police records |
| `schools` | K-12 school locations and attributes | SEMCOG |
| `poi` | Points of interest (grocery, healthcare, parks, etc.) | 2025 Transportation Accessibility Analysis |
| `jobs_2019` | 2019 job counts (pre-COVID baseline reference) | |

---

## Accessibility Tables (External CSVs)

These are loaded as separate orca tables from CSV files, not from the HDF5:

| Table | File | Description |
|---|---|---|
| `accessibility_walk_indicator_by_parcel` | `walk_indicators_by_parcel_20251111.csv` | Walk distances from each parcel to amenities |
| `accessibility_bike_indicator_by_parcel` | `bike_indicators_by_parcel_20251111.csv` | Bike distances from each parcel to amenities |
| `accessibility_drive_indicator_by_parcel` | `drive_indicators_by_parcel_20251111.csv` | Drive distances from each parcel to amenities |

All three are indexed by `parcel_id`. See [Accessibility System](../reference/accessibility.md) for variable details.

---

## Supplementary Input Files

| File | Location | Used By |
|---|---|---|
| `ACS_HH_14_19_BG.csv` | `data/` | `bg_hh_increase` table (block group HH trend) |
| `building_to_zone_baseyear_2020_shrink.csv` | `data/` | TAZ trend variable computation |
| `add_worker_dict.pkl/.csv` | `data/` | Worker assignment utilities |
| `drop_worker_dict.pkl/.csv` | `data/` | Job relocation tracking |
| `parcel_to_legis.csv` | `data/` | Parcel → legislative district mapping (for indicator output) |
| Travel survey files | `/mnt/D/RDF2055/input_data/travel_survey/Full_Interim_Dataset_2026-03-04/` | `travel_survey_vars.py` |

---

## Data Verification

At startup, `dataset.py` calls `verify_data_structure.yaml_from_store()`, which checks the HDF5 store against the expected schema and writes an updated `configs/data_structure.yaml`. If a table or column is missing, it prints a warning but does not stop execution.
