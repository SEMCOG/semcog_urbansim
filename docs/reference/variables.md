# Variable Definitions

Derived variables are defined in the `variables/` package. They are computed on demand by orca when a step requests a column that is not in the base table. This page documents the most important ones.

---

## Building Variables (`variables_building.py`)

| Variable | Formula / Logic | Used By |
|---|---|---|
| `general_type` | Maps `building_type_id` to category string via `building_type_map` | HLCM, ELCM, REPM |
| `is_residential` | `general_type == 'Residential'` | Developer, HLCM |
| `is_office` | `general_type == 'Office'` | ELCM |
| `is_industrial` | `general_type == 'Industrial'` | ELCM |
| `hedonic_id` | `building_type_id × large_area_id` | REPM (segment key) |
| `job_spaces` | `non_residential_sqft / building_sqft_per_job[building_type_id]` | ELCM capacity |
| `jobs_non_home_based` | Count of non-home-based jobs assigned to this building | ELCM capacity |
| `vacant_residential_units` | `residential_units - households_count` | HLCM capacity |
| `building_age` | `year - year_built` | REPM, MCD sampling |
| `large_area_id` | Joined from `parcels.large_area_id` | All models |
| `semmcd` | Joined from `parcels.city_id` → semmcd | MCD sampling |
| `zone_id` | Joined from `parcels.zone_id` | HLCM, ELCM |
| `geoid` | Census block group ID (from parcel) | MCD sampling |
| `nodeid_walk` | Nearest walk network node (from parcel) | Accessibility |
| `nodeid_drv` | Nearest drive network node (from parcel) | Accessibility |

---

## Parcel Variables (`variables_parcel.py`)

| Variable | Logic | Used By |
|---|---|---|
| `acres` | `parcel_acres` (alias) | Feasibility |
| `allowed` | True if any building form is zoning-permitted | Feasibility |
| `parcel_is_allowed_{form}` | Zoning lookup for each form (residential, office, etc.) | Feasibility |

---

## Demographic Variables (`variables_demographic.py`)

These are computed on the `households` table:

| Variable | Formula | Used By |
|---|---|---|
| `has_children` | `children > 0` | HLCM segmentation |
| `with_seniors` | Any person in household aged 65+ (from `persons` table) | HLCM segmentation |
| `income_quartile` | `lowinc` / `midinc` / `highinc` based on regional income distribution | HLCM segmentation |
| `is_young_adult` | `age_of_head` between 18 and 35 | HLCM features |
| `is_family` | Married couple or single parent with children | HLCM features |
| `is_senior_hh` | Household head aged 65+ | HLCM features |

---

## Employment Variables (`variables_employment.py`)

| Variable | Logic | Used By |
|---|---|---|
| `sector_XX_count` | Count of jobs in sector XX in a building | ELCM features |
| `emp_density` | Jobs per acre in zone | ELCM features |

---

## Zone Variables (`variables_zone.py`)

Aggregations at the TAZ (zone) level, joined to buildings via the parcel-zone broadcast:

| Variable | Description |
|---|---|
| `hh_count` | Total households in zone |
| `job_count` | Total jobs in zone |
| `emp_density` | Jobs per acre |
| `pop_density` | Population per acre |
| `residential_units` | Total residential units in zone |
| `vacant_residential_units` | Total vacant units in zone |

---

## Accessibility Variables (`variables_access.py`)

Walk, bike, and drive accessibility metrics joined from the pre-computed indicator tables to buildings via `parcel_id`. See [Accessibility System](accessibility.md) for full variable lists.

**Proximity (near-max) variables** — time/distance to nearest facility (90th percentile of distances from each parcel):
- `hospitals_walk_near_max90`
- `grocery_stores_walk_near_max90`
- `schools_k8_walk_near_max90`
- `fixed_route_bus_walk_near_max90`
- ... (19 walk, 20 bike, 16 drive variables)

**Cumulative variables** — count of destinations reachable within N minutes:
- `jobs_walk_cumulative_5min`, `_10min`, `_15min`, `_30min`
- `jobs_bike_cumulative_5min`, ..., `_30min`
- `jobs_drive_cumulative_10min`, ..., `_45min`
- `jobs_drive_gravity_90min` (gravity-weighted job access)
- `fixed_route_bus_weekday_walk_10min`

---

## TAZ Trend Variables

Generated dynamically by `update_taz_hlcm_trend` each year and registered as building columns:

| Variable | Description |
|---|---|
| `hh_count_taz_10yr_change` | Change in zone household count over last 10 years |
| `hh_pop_taz_10yr_change` | Change in zone population over last 10 years |
| `with_children_taz_10yr_change` | Change in households-with-children count |
| `one_person_hh_taz_10yr_change` | Change in single-person household count |
| `hh_count_taz_5yr_change` | 5-year versions of the above |
| `hh_pop_taz_5yr_change` | |
| `with_children_taz_5yr_change` | |
| `one_person_hh_taz_5yr_change` | |

These give the HLCM a sense of which neighborhoods are gaining or losing population and what types of households are moving in.

---

## Census Tract Variables (`variables_tract.py`)

Aggregations at the census tract level, used for neighborhood context in HLCM/ELCM:

| Variable | Description |
|---|---|
| `tract_hh_count` | Households in census tract |
| `tract_job_count` | Jobs in census tract |
| `tract_median_income` | Median household income |
