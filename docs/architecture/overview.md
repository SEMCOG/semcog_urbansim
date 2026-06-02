# System Architecture Overview

## High-Level Structure

The SEMCOG UrbanSim model is organized as a set of interacting components, all wired together by the **orca** dependency injection framework. Each component is a Python module that registers tables, columns, or simulation steps with orca. When the simulation runs, orca resolves dependencies automatically and calls steps in the declared order.

```
┌─────────────────────────────────────────────────────────────┐
│                    test_forecast_2050.py                    │
│              (entry point, run configuration)               │
└───────────────────────────┬─────────────────────────────────┘
                            │ imports & configures
          ┌─────────────────┼─────────────────────┐
          ▼                 ▼                     ▼
    assumptions.py      dataset.py           models.py
    (injectables,       (HDF5 tables,        (all @orca.step
     constants,          broadcasts,          functions)
     random seed)        checkpoints)
          │                 │                     │
          └─────────────────┴──────────────────── ┘
                            │
                    variables/__init__.py
                    (derived orca columns
                     for all entity types)
                            │
                    lcm_utils.py
                    (PyTorch model loading
                     & step registration)
```

---

## Core Modules

### [`test_forecast_2050.py`](../../test_forecast_2050.py)
The simulation entry point. Sets run-level configuration (paths, year range, scenario toggles), imports all modules, then calls `orca.run()` with the ordered list of annual steps. After the run completes, calls `output_indicators.main()` to aggregate and upload results.

### [`assumptions.py`](../../assumptions.py)
Registers global orca injectables: the `year` injectable (advances each iteration), `building_type_map`, `form_to_btype`, accessibility variable lists (`NEAR_MAX_VARS`, `CUMULATIVE_VARS`), random seeds, and the travel survey path. Also contains `load_latest_input_hdf()` and `verify()` which open the input HDF5 store.

### [`dataset.py`](../../dataset.py)
Loads all HDF5 tables into orca at startup. Defines the primary entity tables (`buildings`, `households`, `jobs`, `parcels`, `persons`) with initial data cleaning and derived columns. Establishes **broadcasts** (orca's foreign-key relationships) between tables. Handles checkpoint resume by overwriting tables from a prior run's last completed year.

### [`models.py`](../../models.py)
The largest file (~3,000 lines). Defines every `@orca.step()` function that executes during simulation. On import, also loads all HLCM and ELCM PyTorch models and registers them as injectable step functions. See [Annual Pipeline](annual-pipeline.md) for the step sequence.

### [`variables/`](../../variables/)
A package of orca column definitions organized by entity type. These are computed on demand when a step requests a column that doesn't exist in the base table.

| File | Covers |
|---|---|
| `variables_parcel.py` | `acres`, `allowed`, zoning compliance |
| `variables_building.py` | `hedonic_id`, `general_type`, `is_residential`, etc. |
| `variables_zone.py` | Zone-level aggregations (employment density, population) |
| `variables_demographic.py` | Household flags: `has_children`, `with_seniors`, `income_quartile` |
| `variables_employment.py` | Employment sector breakdowns |
| `variables_access.py` | Walk / bike / drive accessibility metrics |
| `variables_tract.py` | Census tract aggregations |

### [`lcm_utils.py`](../../lcm_utils.py)
Utilities for loading PyTorch-based location choice models (HLCM and ELCM) from `.pt` files, registering them as orca steps, and running simulations with capacity constraints. Key functions: `load_torch_lcm()`, `load_hlcm_model_configs_from_path()`, `register_hlcm_model_step()`, `register_elcm_model_step()`.

### [`utils.py`](../../utils.py)
General simulation utilities: `get_run_filename()`, `hedonic_simulate()`, `lcm_simulate()`, `SimulationChoiceModel`, `run_log()`, and various transition/relocation helpers.

### [`output_indicators.py`](../../output_indicators.py)
Post-simulation aggregation. Reads the completed run HDF5, computes indicators at multiple geographies (large area, MCD, zone, legislative district), and optionally uploads to PostgreSQL and CartoDB.

---

## Orca Framework Concepts

The model uses **orca** (formerly known as urbansim3) as its simulation backbone. Key concepts:

**Tables** — pandas DataFrames registered with `orca.add_table()` or `@orca.table`. The primary tables are `households`, `jobs`, `buildings`, `parcels`, `persons`.

**Injectables** — scalar values or objects registered with `orca.add_injectable()`. Examples: `year`, `hlcm_model_path`, `repm_step_names`.

**Columns** — derived columns registered with `@orca.column("table_name", "column_name")`. Computed on demand, can depend on other tables/columns.

**Broadcasts** — foreign key relationships declared with `orca.broadcast()`. Allow columns from one table (e.g., `parcels`) to be accessed on a related table (e.g., `buildings`) via a join.

**Steps** — simulation functions decorated with `@orca.step()`. Steps declare their dependencies as function arguments; orca injects them automatically.

**`orca.run()`** — executes a list of step names in order, for each value in `iter_vars`. Each iteration is one simulated year.

---

## Geographic Hierarchy

The model operates across several nested geographic levels:

```
Region (Southeast Michigan, 7 counties)
  └── County (7: Wayne, Oakland, Macomb, Washtenaw, Livingston, Monroe, St. Clair)
        └── Large Area (8 modeling zones, align roughly to counties + Detroit)
              └── MCD / City (semmcd — municipal entities)
                    └── TAZ (Traffic Analysis Zone — travel demand zones)
                          └── Census Block Group
                                └── Parcel (finest unit, ~1.9M parcels)
                                      └── Building (multiple per parcel possible)
```

Households and jobs are assigned to **buildings**. Most model decisions (location choice, developer feasibility) operate at the building or parcel level. Control totals and calibration targets operate at the large area or MCD level.

---

## Key Design Patterns

### Dependency Injection via Orca
Steps are stateless functions. They request data by declaring it as a function argument. This makes steps easy to test in isolation and allows orca to manage caching and invalidation.

### Annual Time Steps
The simulation runs `orca.run()` over `range(base_year+1, final_year+1)`. The `year` injectable updates each iteration. Every step runs once per year in the declared order.

### Segmented Models
Both HLCM and ELCM consist of many smaller models, each covering a specific segment (geographic area × demographic or sector group). Models run sequentially, each consuming unplaced agents from its segment.

### Pseudo-Buildings
Households and jobs with invalid `building_id` values (e.g., -1) are temporarily assigned to pseudo-buildings (IDs > 90,000,000) at startup. These are removed by `drop_pseudo_buildings()` after ELCM placement.

### Event-Driven Policy
Planned developments and demolitions are stored as rows in `events_addition`, `events_deletion`, and `refiner_events` tables in the HDF5 store. The `refiner` step reads these each year and applies them before market simulation.
