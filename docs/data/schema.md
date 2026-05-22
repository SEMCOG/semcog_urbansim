# Data Schema

This page documents the key columns of each primary entity table. These tables are the "agents" that the simulation tracks over time.

---

## `buildings`

The central table. Every household and job must be assigned to a building.

| Column | Type | Description |
|---|---|---|
| `building_id` | int (index) | Unique building identifier |
| `parcel_id` | int | Parent parcel |
| `building_type_id` | int | Building type (see Building Types below) |
| `year_built` | int | Year of construction |
| `residential_units` | int | Number of housing units (0 for non-residential) |
| `non_residential_sqft` | int | Non-residential floor area in sq ft |
| `sqft_per_unit` | int | Square footage per residential unit |
| `stories` | int | Number of stories |
| `market_value` | float | Total assessed market value |
| `sqft_price_res` | float | Residential price per sq ft (updated by REPM) |
| `sqft_price_nonres` | float | Non-residential price per sq ft (updated by REPM) |
| `large_area_id` | int | Large area (joined from parcel) |
| `city_id` | int | City/MCD (joined from parcel) |
| `semmcd` | int | SEMCOG MCD code |
| `zone_id` | int | TAZ zone |
| `geoid` | int | Census block group GEOID |
| `hu_filter` | int | 0 = eligible for HLCM; 1 = filtered (withheld from location choice) |
| `sp_filter` | int | Special filter: 0=normal, -1=event/landmark, -2=pseudo |
| `mcd_model_quota` | int | Max new households assignable by HLCM this year (set by `mcd_hu_sampling`) |
| `event_id` | int | 0 for market buildings; non-zero for event buildings |
| `nodeid_walk` | int | Nearest Pandana walk network node |
| `nodeid_drv` | int | Nearest Pandana drive network node |

**Derived columns** (computed on demand by `variables_building.py`):

| Column | Description |
|---|---|
| `general_type` | Simplified type: Residential, Office, Retail, Industrial, etc. |
| `is_residential` | Boolean |
| `hedonic_id` | `building_type_id × large_area_id` — REPM segment key |
| `job_spaces` | Total job capacity (from `building_sqft_per_job`) |
| `jobs_non_home_based` | Current non-home-based job occupancy |
| `vacant_residential_units` | `residential_units - households_in_building` |
| `building_age` | `current_year - year_built` |

---

## `households`

| Column | Type | Description |
|---|---|---|
| `household_id` | int (index) | Unique household identifier |
| `building_id` | int | Current building assignment (-1 = unplaced) |
| `large_area_id` | int | Large area (joined from building) |
| `persons` | int | Household size |
| `workers` | int | Number of employed workers |
| `income` | int | Household income ($) |
| `age_of_head` | int | Age of household head |
| `children` | int | Number of children under 18 |
| `cars` | int | Number of vehicles |
| `race_id` | int | Race of household head |
| `semmcd` | int | Current MCD (joined from building) |

**Derived columns** (`variables_demographic.py`):

| Column | Description |
|---|---|
| `has_children` | 1 if `children > 0` |
| `with_seniors` | 1 if any member is 65+ |
| `income_quartile` | `lowinc` / `midinc` / `highinc` based on income distribution |
| `is_young_adult` | 1 if head age 18-35 |
| `is_family` | 1 if married couple or single parent with children |

---

## `persons`

| Column | Type | Description |
|---|---|---|
| `person_id` | int (index) | Unique person identifier |
| `household_id` | int | Parent household |
| `age` | int | Person's age |
| `sex` | int | 1=male, 2=female |
| `race_id` | int | Race |
| `worker` | int | 1 if employed |
| `relate` | int | Relationship to household head (0=head, 1=spouse, etc.) |
| `member_id` | int | Position within household |

---

## `jobs`

| Column | Type | Description |
|---|---|---|
| `job_id` | int (index) | Unique job identifier |
| `building_id` | int | Current building assignment (-1 = unplaced) |
| `sector_id` | int | Employment sector (see Employment Sectors below) |
| `home_based_status` | int | 0 = non-home-based; 1+ = home-based |
| `large_area_id` | int | Large area (joined from building) |

---

## `parcels`

| Column | Type | Description |
|---|---|---|
| `parcel_id` | int (index) | Unique parcel identifier |
| `zone_id` | int | TAZ zone |
| `city_id` | int | City/MCD |
| `county_id` | int | County |
| `census_bg_id` | int | Census block group GEOID |
| `x`, `y` | float | Parcel centroid coordinates (state plane) |
| `parcel_acres` | float | Parcel area in acres |
| `nodeid_walk` | int | Nearest Pandana walk node |
| `nodeid_drv` | int | Nearest Pandana drive node |
| `pct_undev` | int | Percent of parcel available for development (0–100) |
| `land_value` | float | Assessed land value |

**Derived columns** (`variables_parcel.py`):

| Column | Description |
|---|---|
| `allowed` | Whether any building form is zoning-allowed on this parcel |
| `parcel_is_allowed_XX` | Per-form zoning allowance flag |
| `acres` | Parcel area (alias for `parcel_acres`) |

---

## Building Types

| ID | General Type | Description |
|---|---|---|
| 11, 13, 14 | Institutional | Government, civic |
| 21 | Retail | Retail commercial |
| 23 | Office | Office |
| 31, 32, 33 | Industrial | Manufacturing, warehouse |
| 41, 42 | TCU | Transportation, communication, utilities |
| 51, 52, 53 | Medical | Healthcare facilities |
| 61, 63, 91 | Entertainment | Recreation, restaurants |
| 65 | Hospitality | Hotels/motels |
| 71 | Others | Miscellaneous |
| 81, 82, 83, 84 | Residential | Single/multi-family |
| 92, 93 | Institutional | Education |
| 94 | Other commercial | |
| 95 | TCU | Utility |

---

## Employment Sectors

| Sector ID | Label |
|---|---|
| 2 | Construction & Mining |
| 3 | Manufacturing |
| 4 | Transportation, Warehousing & Utilities |
| 5 | Wholesale Trade |
| 6 | Retail Trade |
| 8 | Finance, Insurance & Real Estate |
| 9 | Services |
| 10 | Health |
| 11 | Education |
| 14 | Government |
| 16 | Agriculture |
| 17 | Information |

---

## Large Areas

| ID | Geographic Area |
|---|---|
| 3 | Wayne County (excl. Detroit) |
| 5 | Washtenaw County |
| 93 | Monroe County |
| 99 | Macomb County |
| 115 | Oakland County |
| 125 | Livingston County |
| 147 | St. Clair County |
| 161 | Detroit |
