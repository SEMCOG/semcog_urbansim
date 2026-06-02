# Table Catalog

Complete alphabetical list of all tables in the model input file. Each entry shows the data source, owning domain, and index column.

---

| Table | Index | Source | Domain |
|---|---|---|---|
| `accessibility_bike_indicator_by_parcel` | `parcel_id` | Accessibility analysis CSV | [Accessibility](../domains/accessibility.md) |
| `accessibility_drive_indicator_by_parcel` | `parcel_id` | Accessibility analysis CSV | [Accessibility](../domains/accessibility.md) |
| `accessibility_walk_indicator_by_parcel` | `parcel_id` | Accessibility analysis CSV | [Accessibility](../domains/accessibility.md) |
| `annual_employment_control_totals` | `year, sector_id, home_based_status` | CSV (controls pipeline) | [Employment](../domains/employment.md) |
| `annual_household_control_totals` | `year` | CSV (controls pipeline) | [Demographics](../domains/demographics.md) |
| `annual_relocation_rates_for_households` | row integer (no named index) | Excel | [Demographics](../domains/demographics.md) |
| `annual_relocation_rates_for_jobs` | `sector_id` | CSV | [Employment](../domains/employment.md) |
| `bg_hh_increase` | `GEOID` (block group) | ACS CSV | [Demographics](../domains/demographics.md) |
| `building_sqft_per_job` | `building_type_id` | CSV | [Employment](../domains/employment.md) |
| `building_types` | `building_type_id` | CSV | [Land Use](../domains/land-use.md) |
| `buildings` | `building_id` | SQL | [Land Use](../domains/land-use.md) |
| `counties` | `county_id` | Constant (prior HDF) | [Geography](../domains/geography.md) |
| `crime_rates` | `parcel_id` | SQL join | [Development & Context](../domains/development-context.md) |
| `demolition_rates` | `city_id` | SQL | [Development & Context](../domains/development-context.md) |
| `employed_workers_rate` | `large_area_id` | CSV | [Employment](../domains/employment.md) |
| `employment_sectors` | `sector_id` | Constant (prior HDF) | [Employment](../domains/employment.md) |
| `events_addition` | `objectid` | SQL | [Development & Context](../domains/development-context.md) |
| `events_deletion` | `objectid` | SQL | [Development & Context](../domains/development-context.md) |
| `group_quarters` | `personid` | CSV (GQ synthesis) | [Demographics](../domains/demographics.md) |
| `group_quarters_control_totals` | `city_id` ¹ | CSV | [Demographics](../domains/demographics.md) |
| `group_quarters_households` | `household_id` | CSV (GQ synthesis) | [Demographics](../domains/demographics.md) |
| `households` | `household_id` | Population synthesis + placement | [Demographics](../domains/demographics.md) |
| `income_growth_rates` | `year` | CSV | [Demographics](../domains/demographics.md) |
| `jobs` | `job_id` | CSV | [Employment](../domains/employment.md) |
| `land_use_types` | `land_use_type_id` | SQL | [Land Use](../domains/land-use.md) |
| `large_areas` | `large_area_id` | Constant (prior HDF) | [Geography](../domains/geography.md) |
| `landmark_worksites` | — | CSV | [Development & Context](../domains/development-context.md) |
| `mcd_total` | `mcd` | CSV (controls pipeline) | [Demographics](../domains/demographics.md) |
| `multi_parcel_buildings` | — | SQL | [Land Use](../domains/land-use.md) |
| `parcels` | `parcel_id` | SQL | [Land Use](../domains/land-use.md) |
| `persons` | `person_id` | Population synthesis | [Demographics](../domains/demographics.md) |
| `poi` | row index | CSV | [Accessibility](../domains/accessibility.md) |
| `refiner_events` | `refinement_id` | CSV (merged) | [Development & Context](../domains/development-context.md) |
| `remi_pop_total` | `large_area_id` | CSV (derived from REMI, prepared separately) | [Demographics](../domains/demographics.md) |
| `schools` | `bcode` | Excel | [Development & Context](../domains/development-context.md) |
| `semmcds` | `semmcd_id` | SQL | [Geography](../domains/geography.md) |
| `target_vacancies` | — | Excel | [Demographics](../domains/demographics.md) |
| `target_vacancies_mcd` | `cityid` | Excel | [Demographics](../domains/demographics.md) |
| `transit_stops` | `stop_id` | CSV | [Accessibility](../domains/accessibility.md) |
| `travel_data` | `from_zone_id, to_zone_id` | CSV (from OMX) | [Travel Data](../domains/travel.md) |
| `travel_data_2030` | `from_zone_id, to_zone_id` | CSV (from OMX) | [Travel Data](../domains/travel.md) |
| `zones` | `zone_id` | SQL | [Geography](../domains/geography.md) |
| `zoning` | `parcel_id` | SQL | [Land Use](../domains/land-use.md) |

¹ `city_id` is used as the index for `group_quarters_control_totals` and many other tables throughout the model. **Outside the City of Detroit**, `city_id` equals `semmcd` and identifies a municipality. **Inside the City of Detroit**, `city_id` is a neighborhood identifier (not a municipal code). This is a legacy naming convention — see the [Glossary](../index.md#glossary) for the full explanation.

---

## Tables NOT in the Final Model Input (intermediate only)

Used internally during the build process but not exported to the model input file:

| Table | Purpose |
|---|---|
| `orig_synthetic_households` | Raw PopulationSim output → input to household transformation |
| `orig_synthetic_persons` | Raw PopulationSim output → input to person transformation |
| `voters_registration` | Used for household-to-building placement |
| `synthetic_households` | Transformed households before placement |
| `emp_refiner_events` | Raw employment events → merged into `refiner_events` |
| `group_quarters_events` | Raw GQ events → merged into `refiner_events` |
| `group_quarters_persons` | Raw GQ persons → transformed into `group_quarters` |
| `buildings_2010_tract` | Buildings with 2010 tract info (internal reference) |
| `block_to_many_mcd` | Block-to-MCD mapping used during placement |
