# All Orca Steps Reference

Complete alphabetical reference of all `@orca.step()` functions defined in `models.py`, plus dynamically-registered steps.

---

## Fixed Steps (defined in `models.py`)

| Step | Phase | Description |
|---|---|---|
| `build_networks_2050` | Network | Builds Pandana walk/drive networks; computes node-level accessibility aggregations |
| `cache_hh_seeds` | Seed | Caches household snapshot as seed data (first year only) |
| `diagnostic` | Debug | Dumps entity table info for debugging (not in production pipeline) |
| `drop_pseudo_buildings` | Cleanup | Removes temporary pseudo-buildings (ID > 90M); sets affected agents to `building_id = -1` |
| `elcm_home_based` | Employment | *(Disabled)* Places home-based jobs at household locations |
| `feasibility` | Development | Pro-forma feasibility analysis on all developable parcels |
| `fix_lpr` | Demographics | Corrects household `workers` count to match labor participation targets |
| `gq_pop_scaling_model` | Scaling | Adjusts group quarters population to match control totals |
| `households_relocation` | Demographics | *(Legacy)* Older relocation step — not used in 2050 pipeline |
| `households_relocation_2050` | Demographics | Un-places a fraction of households annually; respects `sp_filter` |
| `households_transition` | Demographics | Adds/removes households to match REMI control totals |
| `increase_property_values` | Pricing | *(On hold)* Applies cumulative income growth to prices |
| `init_taz_hlcm_trend_by_year` | Pre-loop | Initializes TAZ trend data from base year (runs once before loop) |
| `jobs_relocation` | Employment | *(Legacy)* Older job relocation step — not in 2050 pipeline |
| `jobs_relocation_2050` | Employment | *(Disabled)* Would un-place jobs annually |
| `jobs_scaling_model` | Scaling | Places remaining unplaced jobs after ELCM, relaxing capacity constraints |
| `jobs_transition` | Demographics | Adds/removes jobs to match REMI employment control totals |
| `mcd_hu_sampling` | Housing | Sets `mcd_model_quota` on buildings based on MCD growth targets and BG trends |
| `neighborhood_vars` | Network | Computes neighborhood context variables for buildings (crime, schools, etc.) |
| `non_residential_developer` | Development | Selects and builds non-residential structures from feasible proposals |
| `random_demolition_events` | Events | Probabilistic demolition by building age and type |
| `refine_housing_units` | Housing | Adjusts housing unit pool (adds/removes units at specific buildings) |
| `refiner` | Events | Applies policy interventions from `refiner_events` table |
| `repm_comparison_log` | Pricing | *(Disabled)* Logs XGBoost vs. Lasso prediction comparison |
| `residential_developer` | Development | Selects and builds residential units from feasible proposals |
| `scheduled_demolition_events` | Events | Demolishes buildings listed in `events_deletion` for current year |
| `scheduled_development_events` | Events | Adds buildings listed in `events_addition` for current year |
| `update_bg_hh_increase` | Cleanup | Updates block group HH growth rolling window for MCD sampling |
| `update_sp_filter` | Events | Updates `sp_filter` on newly added event buildings |
| `update_taz_hlcm_trend` | Network | Updates TAZ trend table; registers 5yr/10yr change building columns |

---

## Dynamically Registered Steps

### REPM Steps (`repm_step_names`)

One step per trained REPM model segment. Named by the model directory:

```
res_hedonic_XX_laYY
nonres_hedonic_XX_laYY
```

Where `XX` = `hedonic_id` and `YY` = `large_area_id`.

Example names:
- `res_hedonic_81_la161` — residential type 81, Detroit
- `nonres_hedonic_23_la115` — office type 23, Oakland County

### HLCM Steps (`hlcm_step_names`)

One step per segment. Named by the `.pt` filename (without extension).

Naming convention:
```
la{large_area}_{children}_{seniors}_{income}_hlcm.pt
```

Example names:
- `la161_has_children_lowinc_hlcm`
- `la115_no_children_without_seniors_highinc_hlcm`

Sorted alphabetically (effectively: large area → children status → seniors status → income).

### ELCM Steps (`elcm_step_names`)

One step per segment. Named by the `.pt` filename.

Naming convention:
```
la{large_area}_sector{sector}_nonhomebased_elcm.pt
la{large_area}_sector{sector}_homebased_elcm.pt
```

Sorted by `elcm_sector_order = [3, 6, 10, 11, 14, 9, 4, 2, 5, 16, 17, 8]`.

---

## Step Execution Order (Production Pipeline)

See [Annual Simulation Pipeline](../architecture/annual-pipeline.md) for the full ordered list. Summary:

```
1.  build_networks_2050
2.  neighborhood_vars
3.  update_taz_hlcm_trend
4.  cache_hh_seeds              (first year only)
5.  scheduled_demolition_events
6.  random_demolition_events
7.  scheduled_development_events
8.  refiner
9.  households_transition
10. fix_lpr
11. households_relocation_2050
12. jobs_transition
13. drop_pseudo_buildings
14. feasibility
15. residential_developer
16. non_residential_developer
17. update_sp_filter
18. [repm_* × N]
19. refine_housing_units
20. mcd_hu_sampling
21. [hlcm_* × ~72]
22. [elcm_* × ~200]
23. jobs_scaling_model
24. gq_pop_scaling_model
25. update_bg_hh_increase
```
