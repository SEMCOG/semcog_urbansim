# Annual Simulation Pipeline

Each simulated year executes the following steps in sequence. One pass through this list = one year of simulation. The pipeline runs from year `base_year + 1` through `final_year` (2021–2050 by default).

---

## Pre-Loop Initialization

Before the annual loop starts, one step runs once:

```
init_taz_hlcm_trend_by_year
```
Loads base-year TAZ-level variables (household counts, population, demographics) that will be used to compute 5-year and 10-year trend features for the HLCM. Only runs once at startup.

---

## Annual Steps (in order)

### Phase 1 — Network & Neighborhood Variables

| Step | Purpose |
|---|---|
| `build_networks_2050` | Builds Pandana walk and drive street networks for the current year. Computes node-level accessibility aggregations (jobs within N minutes, etc.) and registers them on the `nodes_walk` and `nodes_drv` orca tables. |
| `neighborhood_vars` | Computes neighborhood-level variables (crime rates, school proximity, etc.) and attaches them to buildings via the broadcast chain. |
| `update_taz_hlcm_trend` | Updates the TAZ trend table with the current year's household/population counts. Derives 5-year and 10-year change variables per TAZ and registers them as building columns (`*_taz_5yr_change`, `*_taz_10yr_change`). |

---

### Phase 2 — Seeding (First Year Only)

| Step | Purpose |
|---|---|
| `cache_hh_seeds` | Caches a snapshot of the current household table as "seed" data for replication and diagnostics. Only executes in the first simulation year; skipped in all subsequent years. |

---

### Phase 3 — Events & Policy Interventions

| Step | Purpose |
|---|---|
| `scheduled_demolition_events` | Removes buildings listed in the `events_deletion` table for the current year. Updates `dropped_buildings` tracker. |
| `random_demolition_events` | Applies probabilistic demolition rates from the `demolition_rates` table. Samples buildings stochastically by building type and age. |
| `scheduled_development_events` | Adds buildings from the `events_addition` table for the current year (planned developments, major projects). |
| `refiner` | Applies policy-based additions and deletions from the `refiner_events` table. More flexible than the scheduled events — can target specific parcels, building types, or geographic areas. See [Policy Interventions](../models/refiner.md). |

---

### Phase 4 — Demographic Transitions

| Step | Purpose |
|---|---|
| `households_transition` | Adds and removes households to match `annual_household_control_totals`. New households are created by sampling from existing households (demographic replication). Removed households are deleted. New households start with `building_id = -1` (unplaced). See [Demographic Transitions](../models/transitions.md). |
| `fix_lpr` | Adjusts labor participation rates. Corrects the `workers` field in the households table after transition to match target labor force participation by large area and income group. |
| `households_relocation_2050` | Sets a fraction of currently-placed households to `building_id = -1` (unplaced), using rates from `annual_relocation_rates_for_households`. Households in event/special buildings (`sp_filter < 0`) are protected from relocation. |
| `jobs_transition` | Adds and removes jobs to match `annual_employment_control_totals`. New jobs start with `building_id = -1`. See [Demographic Transitions](../models/transitions.md). |

> **Note:** `jobs_relocation_2050` is currently disabled in the pipeline. Jobs are relocated only via the ELCM placement of newly transitioned jobs.

---

### Phase 5 — Pseudo-Building Cleanup

| Step | Purpose |
|---|---|
| `drop_pseudo_buildings` | Removes pseudo-buildings (IDs > 90,000,000) from the buildings table. Any households or jobs still assigned to pseudo-buildings at this point are set to `building_id = -1` and will be placed by the location choice models below. |

---

### Phase 6 — Real Estate Development

| Step | Purpose |
|---|---|
| `feasibility` | Runs pro-forma feasibility analysis on all developable parcels. For each parcel × building form combination, estimates revenue (from REPM prices) vs. construction cost. Returns a table of feasible development proposals. See [Developer & Feasibility](../models/developer.md). |
| `residential_developer` | From the feasible proposals, selects and "builds" residential units. Respects MCD-level housing unit targets and vacancy rate targets. New buildings are added to the buildings table. |
| `non_residential_developer` | Same as above for non-residential forms (office, retail, industrial, medical, entertainment). |
| `update_sp_filter` | Updates the `sp_filter` column on newly added event buildings to mark them as special (protected from random relocation or demolition). |

---

### Phase 7 — Real Estate Price Update

| Steps | Purpose |
|---|---|
| `repm_step_names` (list) | Runs each registered XGBoost REPM model in turn. Each model updates `sqft_price_res` or `sqft_price_nonres` for a segment of buildings (defined by `hedonic_id` = building type × large area combination). See [REPM](../models/repm.md). |

---

### Phase 8 — Housing Unit Allocation

| Step | Purpose |
|---|---|
| `refine_housing_units` | Adjusts the pool of available housing units. Can add or remove units at specific buildings based on known housing unit changes not captured by the developer model. |
| `mcd_hu_sampling` | Sets `mcd_model_quota` on buildings — the number of housing units that the HLCM should fill for each MCD in the current year. Quota is computed from MCD-level household targets (`mcd_total` table) vs. current occupancy. Prioritizes newer buildings in high-growth block groups. |

---

### Phase 9 — Household Location Choice

| Steps | Purpose |
|---|---|
| `hlcm_step_names` (list) | Runs each HLCM model in sequence (sorted alphabetically by large area → demographic segment). Each model places unplaced households from its segment into buildings. Uses PyTorch neural network to score alternatives; respects `mcd_model_quota` capacity. See [HLCM](../models/hlcm.md). |

---

### Phase 10 — Employment Location Choice

| Steps | Purpose |
|---|---|
| `elcm_step_names` (list) | Runs each ELCM model in a sector-dependency order (`[3, 6, 10, 11, 14, 9, 4, 2, 5, 16, 17, 8]`). Each model places unplaced jobs from its segment (large area × sector × home_based_status) into buildings. See [ELCM](../models/elcm.md). |

---

### Phase 11 — Post-Placement Scaling & Cleanup

| Step | Purpose |
|---|---|
| `jobs_scaling_model` | If any jobs remain unplaced after ELCM runs (due to lack of suitable space), assigns them to buildings by scaling up capacity in high-demand areas. Prevents simulation failure due to undersupply. |
| `gq_pop_scaling_model` | Applies group quarters population controls. Adjusts group quarters household counts to match `group_quarters_control_totals` by large area and type. |
| `update_bg_hh_increase` | Updates the `bg_hh_increase` table with the current year's block group household counts. This rolling 3-year window of block group growth trends is used by `mcd_hu_sampling` in the next year. |

---

## Step Execution Summary

```
YEAR Y
├── build_networks_2050
├── neighborhood_vars
├── update_taz_hlcm_trend
├── cache_hh_seeds              ← first year only
├── scheduled_demolition_events
├── random_demolition_events
├── scheduled_development_events
├── refiner
├── households_transition        ← REMI control totals
├── fix_lpr
├── households_relocation_2050
├── jobs_transition              ← REMI control totals
├── drop_pseudo_buildings
├── feasibility
├── residential_developer
├── non_residential_developer
├── update_sp_filter
├── [repm_* × N segments]        ← XGBoost price update
├── refine_housing_units
├── mcd_hu_sampling
├── [hlcm_* × ~72 models]        ← household placement
├── [elcm_* × ~200 models]       ← job placement
├── jobs_scaling_model
├── gq_pop_scaling_model
└── update_bg_hh_increase
                                 → write snapshot to runs/runNNN.h5 / YEAR Y /
```

---

## Output Written Each Year

After all steps complete for year Y, orca writes a snapshot of `out_run_tables` to the HDF5 output file at path `/YEAR/table_name`:

- `buildings` — full building table including new construction
- `jobs` — all jobs with current building assignments
- `base_job_space` — non-home-based job capacity at base year
- `parcels` — parcel attributes
- `households` — all households with current building assignments
- `persons` — all persons
- `group_quarters` — group quarters population
- `dropped_buildings` — buildings demolished this year
- `bg_hh_increase` — block group household growth tracking
