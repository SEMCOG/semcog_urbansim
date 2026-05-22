# Development & Context

**Primary sources:** PostgreSQL event and demolition tables; CSV files; crime data from PostgreSQL

This domain brings together two related categories:

- **Planned development events** — specific buildings that will be added or removed at known future years, plus background demolition rates
- **Neighborhood context** — school quality and crime rates that influence location choice

Both categories shape *where* development and population locate in the model.

---

## Tables in This Domain

| Table | Description | Source |
|---|---|---|
| `events_addition` | Planned buildings to add at a specific year | SQL |
| `events_deletion` | Planned buildings to demolish at a specific year | SQL |
| `refiner_events` | Flexible policy interventions (add/remove agents) | CSV (merged from emp + GQ events) |
| `demolition_rates` | Annual background demolition counts by city | SQL |
| `landmark_worksites` | Major employers protected from simulation randomness | CSV |
| `schools` | K-12 school locations and quality scores | Excel |
| `crime_rates` | Violent and property crime rates by parcel | SQL join |

---

## Development Events

### `events_addition`

Buildings to be added at a specific future year. Represents known projects (stadiums, hospitals, planned developments) that override the market developer model.

**Index:** `objectid` (unique)  
**SQL table:** configured in `config/sql.yaml`

| Column | Type | Rules | Notes |
|---|---|---|---|
| `objectid` | int | unique, no null | Index |
| `parcel_id` | int32 | no null | Must exist in `parcels` |
| `building_type_id` | int8 | join → `building_types`, no null | |
| `year_built` | int16 | range [base year+1 – forecast end], no null | Year the building is added |
| `residential_units` | int16 | no null | 0 for non-residential |
| `non_residential_sqft` | int32 | no null | 0 for residential |
| `stories` | int | — | |
| `zone_id` | int16 | — | TAZ reference |
| `city_id` | int16 | — | MCD reference |
| `gqcap` | int | — | Group quarters capacity (0 if not GQ) |
| `event_id` | int | — | Project reference ID |

Event buildings receive `sp_filter = -1` (protected from random relocation and demolition).

**Common issues:**
- `parcel_id` not in `parcels` — building can't be placed; fix parcel reference
- `year_built` before or at the base year — these events are filtered out; verify years
- Year range should cover only future years (after the base year)

### `events_deletion`

Buildings to be demolished at a specific future year.

**Index:** `objectid` (unique)  
**SQL table:** configured in `config/sql.yaml`

| Column | Type | Rules | Notes |
|---|---|---|---|
| `objectid` | int | unique, no null | |
| `building_id` | int32 | no null | Building to be demolished |
| `year_built` | int16 | no null | Year of demolition |

**Common issues:**
- `building_id` not in `buildings` — harmless (skipped) but should be cleaned up
- Demolition years before the base year — verify data entry

### `refiner_events`

Flexible event table for complex interventions: adding employment to a specific building, removing households from a zone, adding GQ capacity. Assembled in the build pipeline by merging employment events and GQ events.

**Index:** `refinement_id` (unique)  
**Source:** CSV files — paths in `config/files.yaml` — merged in `main.py`

| Column | Type | Rules | Notes |
|---|---|---|---|
| `refinement_id` | int16 | unique, no null | |
| `transaction_id` | int16 | no null | Groups related events |
| `year` | int16 | forecast year range, no null | |
| `action` | object | no null | `"add"` or `"remove"` |
| `agents` | object | no null | `"jobs"`, `"households"`, or `"group_quarters"` |
| `agent_expression` | object | no null | Filter defining which agents to affect |
| `location_expression` | object | no null | Filter defining target buildings |
| `amount` | int16 | no null | Number of agents |

**Update:** Add rows to the employment events CSV for new employment centers or relocations. Add rows to the GQ events CSV for GQ facility changes. Re-run the build pipeline to merge into `refiner_events`.

### `demolition_rates`

Annual background demolition counts by city and building type. Applied probabilistically each year by `random_demolition_events`.

**Index:** `city_id`  
**SQL table:** configured in `config/sql.yaml`

| Column | Type | Notes |
|---|---|---|
| `city_id` | int16 | MCD reference |
| `type81units` | int32 | Single-family units to demolish annually |
| `type82units` | int32 | |
| `type83units` | int32 | Multi-family |
| `type84units` | int32 | |
| `typenonsqft` | int32 | Non-residential sq ft to demolish annually |

These are **absolute counts** (not rates) applied stochastically — buildings are sampled to meet each city's annual target. Update in PostgreSQL when calibrating to new demolition patterns.

### `landmark_worksites`

Major employers whose jobs are protected from simulation randomness. Large factories, university campuses, hospital complexes.

**Source:** CSV — path in `config/files.yaml`

| Column | Notes |
|---|---|
| `building_id` | Building to protect — receives `sp_filter = -1` at startup |

**How it works:** At startup, buildings in this list get `sp_filter = -1`, preventing the ELCM and relocation models from touching their jobs.

**Update:** Add new anchor employers when major facilities open; remove entries if a facility closes. Keep the list conservative — over-protecting buildings reduces model responsiveness.

---

## Neighborhood Context

### `schools`

K-12 school locations and quality scores. Used as HLCM features — school quality affects where families with children choose to live.

**Index:** `bcode` (unique)  
**Source:** Excel file — path in `config/files.yaml`

| Column | Type | Rules | Notes |
|---|---|---|---|
| `bcode` | int16 | unique, no null | School building code |
| `bname` | object | — | School name |
| `dcode` | int32 | range [47010–84060], no null | School district code |
| `gradelist` | object | no null | Grade levels served |
| `is_grade_school` | int8 | values: 0, 1 | 1=K-8 |
| `point_x` | float64 | no null | State plane X coordinate |
| `point_y` | float64 | no null | State plane Y coordinate |
| `totalachievementindex` | float64 | no null | School achievement score |

**Update:** Refresh from the Michigan Department of Education database when new schools open, schools close, or achievement scores are updated. For new schools without a score, use the district average.

### `crime_rates`

Violent and property crime rates by parcel. Derived from city-level crime data — all parcels in a city get the same rate.

**Index:** `parcel_id` (unique, joins to `parcels`)  
**SQL source:** Join of parcel and crime rate tables — query in `config/sql.yaml`

| Column | Type | Rules | Notes |
|---|---|---|---|
| `parcel_id` | int32 | join → `parcels`, no null | Index |
| `ucr_crime_rate` | float64 | no null | Violent crime rate (UCR) |
| `other_crime_rate` | float64 | no null | Property crime rate |

**Update:** Update the crime rates table in PostgreSQL with new annual data by city. The parcel-level values are derived automatically from city rates when the SQL query runs.

**Common issues:**
- Cities with no data result in null rates — fill with regional average or zero; null values will cause model errors

---

## Update Checklist (New Forecast Cycle)

**Events:**
- [ ] Review planned development projects in PostgreSQL — add new projects, update delayed ones
- [ ] Review demolition records in PostgreSQL
- [ ] Update employment events CSV with new planned employment centers
- [ ] Update GQ events CSV with new facility openings/closings
- [ ] Re-run build pipeline to merge `refiner_events`
- [ ] Review `landmark_worksites.csv` — add/remove anchor employers as needed
- [ ] Review `demolition_rates` in PostgreSQL — update if patterns have changed

**Context:**
- [ ] Update schools from MDE database — new/closed schools, updated scores
- [ ] Update `urbansim_crime_rates` in PostgreSQL with latest annual crime data
- [ ] Run validation: `python -m input_validation.cli --hdf <output hdf>`
