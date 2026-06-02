# Scenario Controls

## What Is a Scenario?

The **baseline forecast** uses REMI demographic and employment projections that assume historical migration patterns continue. A **scenario** replaces those control totals with alternative assumptions — for example, lower net immigration to the region.

Scenarios allow planners to ask: *What happens to Southeast Michigan's land use pattern if population growth is lower / higher / differently distributed?*

---

## Enabling a Scenario

In `test_forecast_2050.py`, set:

```python
orca.add_injectable('ENABLE_SCENARIO', True)
orca.add_injectable('scenario_hh_control_path',
    '/mnt/hgfs/urbansim/RDF2050/scenarios/controls/low_immigration/annual_household_control_totals_2050_07232024.csv')
orca.add_injectable('scenario_remi_total_pop',
    '/mnt/hgfs/urbansim/RDF2050/scenarios/controls/low_immigration/remi_total_pop_la07232024.csv')
orca.add_injectable('scenario_emp_control_path',
    '/mnt/hgfs/urbansim/RDF2050/scenarios/controls/low_immigration/annual_employment_control_totals.csv')
```

When `ENABLE_SCENARIO = True`, `models.py` replaces the standard orca tables at import time:

```python
if orca.get_injectable('ENABLE_SCENARIO'):
    orca.add_table('annual_household_control_totals', 
                   pd.read_csv(hh_controls_path, index_col=0))
    orca.add_table('remi_pop_total', 
                   pd.read_csv(remi_total_pop_path, index_col=0))
    orca.add_table('annual_employment_control_totals', 
                   pd.read_csv(emp_controls_path, index_col=0))
```

All subsequent simulation steps read from these replaced tables — no other changes are needed.

---

## Available Scenarios

### Low Immigration (default scenario)

**Location:** `/mnt/hgfs/urbansim/RDF2050/scenarios/controls/low_immigration/`

Assumes reduced net in-migration to Southeast Michigan based on alternative REMI projections. Results in lower overall population and household growth vs. the baseline, particularly in the outer counties.

Files:
- `annual_household_control_totals_2050_07232024.csv`
- `remi_total_pop_la07232024.csv`
- `annual_employment_control_totals.csv`

---

## Creating a New Scenario

To create a custom scenario:

1. Prepare alternative control total CSV files matching the schema of `annual_household_control_totals`, `remi_pop_total`, and `annual_employment_control_totals`
2. Place them in a new subdirectory under `.../scenarios/controls/`
3. Update the three `scenario_*_path` injectables in `test_forecast_2050.py`
4. Set `ENABLE_SCENARIO = True`

### Control Total File Schemas

**`annual_household_control_totals`:**
```
year, large_area_id, age_of_head_min, age_of_head_max, persons_min, persons_max,
workers_min, workers_max, income_min, income_max, children, cars, race_id,
total_number_of_households
```

**`annual_employment_control_totals`:**
```
year, large_area_id, sector_id, home_based_status, number_of_jobs
```

**`remi_pop_total`:**
```
large_area_id, 2021, 2022, ..., 2050
```

---

## Scenario Output Identification

The `run_config.yaml` written to `runs/runNNN/` records `ENABLE_SCENARIO` and the scenario file paths, so outputs are always traceable to their assumptions.

It is good practice to use a descriptive run output directory name or to note the run number alongside the scenario name when reporting results.
