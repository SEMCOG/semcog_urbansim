# Land Use

**Primary source:** PostgreSQL parcel and building database

This domain covers the spatial foundation of the model — every parcel, every building, the zoning overlay, and land use and building type definitions.

---

## Tables in This Domain

| Table | Description | Source |
|---|---|---|
| `parcels` | ~1.9M parcels — the finest spatial unit | SQL |
| `buildings` | ~1.8M buildings | SQL |
| `zoning` | Parcel-level zoning (one row per parcel) | SQL |
| `land_use_types` | Land use type definitions | SQL |
| `building_types` | Building type code definitions | CSV |
| `multi_parcel_buildings` | Buildings spanning multiple parcels | SQL |

---

## `parcels`

The foundational spatial table. Every building must have a `parcel_id` that exists here. Used for all spatial joins in the model.

**Index:** `parcel_id` (unique)

**Columns:**

| Column | Type | Rules | Notes |
|---|---|---|---|
| `parcel_id` | int32 | unique, no null | Primary key |
| `census_bg_id` | int32 | no null | Census block group GEOID |
| `centroid_x` | float64 | no null | State plane X coordinate (Michigan GeoRef / NAD83) |
| `centroid_y` | float64 | no null | State plane Y coordinate |
| `city_id` | int16 | no null | Municipality code (= `semmcd` outside Detroit; neighborhood id inside Detroit — see [Glossary](../index.md#glossary)) |
| `county_id` | int16 | 7 valid county codes | Must be one of the 7 SE Michigan counties |
| `land_use_type_id` | int16 | join → `land_use_types` | Must reference a valid land use type |
| `large_area_id` | int16 | 8 valid codes | One of the 8 modeling zones |
| `parcel_sqft` | int32 | no null | Parcel area in sq ft |
| `pptytax` | float64 | no null | Property tax millage rate |
| `school_id` | int32 | no null | School district reference |
| `semmcd` | int16 | no null | SEMCOG MCD code |
| `sev_value` | int32 | no null | State Equalized Value |
| `landvalue` | int32 | no null | Assessed land value |
| `bldgimprval` | int32 | no null | Total building improvement value for the parcel (all buildings combined); used to compute `buildings.improvement_value` and the `bldg_impr_land_ratio` feature in the developer model |
| `zone_id` | int16 | no null | TAZ zone reference |

**Common issues:**
- Parcels with `zone_id = null` after join — no TAZ match; check that the parcel's zone code exists in the TAZ boundary table
- `large_area_id` outside the 8 allowed values — usually out-of-region parcels; filter them out
- Coordinate outliers — verify projection (Michigan GeoRef state plane) and check for misplaced records

---

## `buildings`

Every residential unit and job must be assigned to a building. This is the most-used table in the simulation. Building count is close to parcel count — most parcels have one building.

**Index:** `building_id` (unique)

**Columns exported to model input:**

| Column | Type | Rules | Notes |
|---|---|---|---|
| `building_id` | int32 | unique, no null | Primary key |
| `parcel_id` | int32 | join → `parcels`, no null | Every building must have a valid parcel |
| `building_type_id` | int8 | join → `building_types`, no null | See building type codes below |
| `year_built` | int16 | range [1760 – base year], no null | Cap at base year |
| `residential_units` | int16 | no null | 0 for non-residential |
| `non_residential_sqft` | int64 | no null | 0 for residential |
| `sqft_per_unit` | int16 | no null | 0 for non-residential |
| `stories` | float64 | no null | |
| `improvement_value` | float64 | no null | Derived during assembly: parcel's `bldgimprval` allocated to each building proportionally by its share of total building sqft on the parcel |
| `land_area` | int32 | no null | Building footprint |
| `market_value` | float64 | — | Total assessed value |

**Building type codes:**

| ID Range | General Type |
|---|---|
| 11, 13, 14 | Institutional |
| 21 | Retail |
| 23 | Office |
| 31, 32, 33 | Industrial |
| 41, 42 | TCU |
| 51, 52, 53 | Medical |
| 61, 63, 91 | Entertainment |
| 65 | Hospitality |
| 71 | Other |
| 81, 82, 83, 84 | Residential |
| 92, 93 | Institutional (education) |
| 94 | Other commercial |
| 95 | TCU |

**Common issues:**
- `parcel_id` not found in `parcels` — orphaned buildings; these break the spatial join chain
- `year_built` after base year — cap to base year
- Buildings with both `residential_units > 0` AND `non_residential_sqft > 0` — mixed-use is valid; verify intentional
- `residential_units > 0` but `sqft_per_unit = 0` — the model uses `sqft_per_unit` for feasibility analysis; assign a reasonable default

---

## `zoning`

Controls what can be built on each parcel. Directly drives the developer/feasibility model.

**Index:** `parcel_id` (unique)

**Columns:**

| Column | Type | Rules | Notes |
|---|---|---|---|
| `parcel_id` | int32 | join → `parcels`, no null | |
| `future_use` | object | no null | Future land use designation |
| `max_dua` | float64 | range [0 – 126], no null | Max dwelling units per acre |
| `max_far` | float64 | no null | Max floor area ratio |
| `max_height` | int16 | no null | Max building height (ft) |
| `max_stories` | float16 | range [1 – 73], no null | Max number of stories |
| `pct_undev` | float64 | range [0 – 100], no null | Percent of parcel undeveloped |
| `is_developable` | int | — | 0=not developable, 1=developable, 2=developable with Underground Storage Tank (UST) present. Value 2 parcels are treated as developable but have 10 percentage points added to `pct_undev`, making them slightly less competitive for new development. |
| `type81` – `type95` | int | — | 1 = building type allowed on this parcel |

**Common issues:**
- `pct_undev = null` — treated as 0% developable in the model; fill with 0 or actual value
- Missing `is_developable` — defaults to not developable; verify coverage
- Any parcel in `zoning` must have a matching `parcel_id` in `parcels`

---

## `land_use_types`

Reference table of land use type definitions.

**Index:** `land_use_type_id` (unique)

| Column | Type | Rules |
|---|---|---|
| `land_use_type_id` | int8 | no null |
| `land_use_name` | object | no null |
| `description` | object | no null |
| `is_residential` | int8 | values: 0, 1 |
| `home_based_status` | int8 | values: 0, 1 |
| `generic_land_use_type_id` | int8 | no null |
| `unit_name` | object | no null |

---

## `building_types`

Maps numeric `building_type_id` codes to names and general categories. Rarely changes.

**Index:** `building_type_id` (unique)

**Required IDs:** All 26 building type codes must be present (11, 13, 14, 21, 23, 31–33, 41–42, 51–53, 61, 63, 65, 71, 81–84, 91–95).

| Column | Type | Rules |
|---|---|---|
| `building_type_id` | int8 | all 26 codes required, no null |
| `building_type_name` | object | no null |
| `building_type_description` | object | no null |
| `generic_building_type_id` | int8 | no null |
| `generic_building_type_name` | object | no null |
| `is_residential` | int8 | values: 0, 1 |
| `naics` | object | no null |
| `unit_name` | object | no null |

---

## `multi_parcel_buildings`

Buildings that span multiple parcels (large industrial complexes, campuses). Only update when new multi-parcel projects are added or removed.

| Column | Rules |
|---|---|
| `building_id` | join → `buildings` |
| `parcel_id` | join → `parcels` |

---

## Update Checklist (New Forecast Cycle)

- [ ] Verify parcel and building tables are updated to the new base year in PostgreSQL
- [ ] Confirm zoning layer reflects the latest approved zoning
- [ ] Re-run SQL extraction; verify row counts are plausible
- [ ] Check `building_type_id` values — all must be in the 23 allowed codes
- [ ] Check `year_built` — cap anything beyond the new base year
- [ ] Check `parcel_id` referential integrity: every building's parcel must exist in `parcels`
- [ ] Check for null `zone_id` values in parcels — every parcel needs a TAZ assignment
- [ ] Check for null `large_area_id` or values outside the 8 valid codes
- [ ] Check `pct_undev` in zoning — nulls default to 0% developable
