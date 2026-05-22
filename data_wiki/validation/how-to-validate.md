# How to Validate the HDF

## Overview

The `input_validation` module validates the assembled HDF against a formal specification. It checks:

- **Table presence** — every required table exists
- **Column dtypes** — columns have the expected data type
- **Null values** — columns flagged `no_null` have no missing values
- **Allowed values** — categorical columns only contain expected codes
- **Value ranges** — numeric columns are within expected bounds
- **Required coverage** — e.g., all forecast years are present
- **Referential integrity** — foreign keys exist in the referenced table
- **Extra tables/columns** — warns about items not covered by the spec

The validation spec is `input_validation/hdf_check.yaml`. The HDF path and other options are set at runtime — no file changes are needed to run validation.

---

## Running Validation

### Basic check after building the HDF

```bash
python -m input_validation.cli --hdf <path-to-hdf-output>
```

### Save results to file

```bash
python -m input_validation.cli --hdf <path-to-hdf> --output validation_report.yaml
```

### Strict mode (non-zero exit code on errors — useful for automated checks)

```bash
python -m input_validation.cli --hdf <path-to-hdf> --strict
```

### As part of the main build pipeline

Pass `--run-validation` when running `main.py`:

```bash
python main.py --run-validation --validation-output validation_report.yaml
```

This runs validation automatically after export and prints a summary. Recommended for production builds.

---

## Validation Rule Reference

Rules are defined per-column in `input_validation/hdf_check.yaml`:

| Rule | Meaning |
|---|---|
| `data_type: int8` | Column must be this dtype |
| `no_null: true` | No null values allowed |
| `match_vals: [1, 2, 3]` | All values must be in this list |
| `within_range: [0, 100]` | All values must fall in this range (inclusive) |
| `contains_all: [2020, 2050]` | Column must contain at least these values |
| `contains_all: range(2020, 2050)` | Column must contain all integers from 2020 to 2049 |
| `join_check: [buildings.building_id, "m:1"]` | All values must exist in the referenced table column |
| `idx_names: [parcel_id]` | Index levels must have these names |
| `idx_unique: true` | Index must be unique |

---

## Interpreting Results

### Issue levels

| Level | Meaning | Action Required |
|---|---|---|
| `error` | Data fails a required rule | **Fix before using in simulation** |
| `warning` | Extra tables/columns not in spec | Review — usually benign but worth noting |

### Common error patterns and fixes

**Missing table:**
```
ERROR table parcels: table is missing from HDF
```
→ The table was not exported. Check the export list in `main.py` and re-run.

**Wrong value:**
```
ERROR check annual_employment_control_totals.large_area_id: failure cases: [0, 200]
```
→ Invalid large area codes. Check for unmapped geographies in control totals source data.

**Orphaned foreign key:**
```
ERROR check buildings.parcel_id: failure cases: [123456, 789012] (150 total)
```
→ Buildings whose parcel doesn't exist in `parcels`. These break spatial joins in the simulation.

Quick diagnostic:
```python
import pandas as pd
store = pd.HDFStore('path/to/hdf_out.h5', 'r')
buildings = store['buildings']
parcels = store['parcels']
orphans = buildings[~buildings.parcel_id.isin(parcels.index)]
print(f"{len(orphans)} orphaned buildings")
store.close()
```

**Missing years in control totals:**
```
ERROR check annual_household_control_totals.year: failure cases: [2045, 2046, 2047, 2048, 2049]
```
→ Run the forecast controls pipeline for the missing years and re-export.

**Out-of-range values:**
```
ERROR check buildings.year_built: failure cases: [2025, 2030]
```
→ Buildings with year_built after the base year. Cap to the base year in the source data or SQL query.

---

## Adding New Validation Rules

To add rules for a new table or column, edit `input_validation/hdf_check.yaml`:

```yaml
my_new_table:
  index:
    idx_names: [my_id]
    idx_unique: true
  my_id:
    data_type: int32
    no_null: true
  value_col:
    data_type: float64
    within_range: [0.0, 1.0]
    no_null: true
```

Run validation after adding rules to confirm they work as expected. See [Table Catalog](../reference/table-catalog.md) for tables currently missing validation rules.
