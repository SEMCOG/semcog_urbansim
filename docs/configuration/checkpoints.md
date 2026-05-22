# Checkpoints & Resume

## What Is a Checkpoint?

A checkpoint allows you to **resume a simulation from where it stopped**, rather than re-running from the base year. This is useful when:

- A long run was interrupted (power loss, disk full, process killed)
- You want to branch from a partially-completed run with different settings
- You need to re-run only the final years after updating a model

The simulation writes a full snapshot of all entity tables to the output HDF5 at the end of each year. A checkpoint resume reads the last completed year's snapshot as the new starting state.

---

## Configuring a Resume

In `test_forecast_2050.py`:

```python
orca.add_injectable('use_checkpoint', True)
orca.add_injectable('runnum_to_resume', 'run1365.h5')
```

The `runnum_to_resume` value must be the filename (not full path) of an existing run HDF5 in the `runs/` directory.

---

## How It Works

### At Startup (`dataset.py`)

When `use_checkpoint = True`, `assumptions.verify()`:

1. Copies the input HDF5 to `data/checkpoint_store.h5` (working copy)
2. Opens the checkpoint run file (`runs/run1365.h5`)
3. Calls `update_store_from_checkpoint()` which:
   - Finds the **last completed year** in the checkpoint HDF5 (max year key)
   - Sets `checkpoint_year` injectable to that year
   - Overwrites these tables in the working store with checkpoint values:
     - `buildings`, `jobs`, `parcels`, `households`, `persons`
     - `group_quarters`, `dropped_buildings`, `bg_hh_increase`
   - Preserves base-store dtypes for common columns

### At Run Start (`test_forecast_2050.py`)

```python
run_start = base_year if not orca.get_injectable('use_checkpoint') \
            else orca.get_injectable('checkpoint_year')
```

The simulation loop starts from `run_start + 1` (the year after the last completed checkpoint year).

### After Completion

After the new run finishes, the checkpoint merge step copies all year groups from the old run that are **not** in the new run:

```python
if orca.get_injectable('use_checkpoint'):
    # Add prior years from checkpoint run into new output HDF5
    for k in old_result:
        if '/base/' in k or k in store_la.keys():
            continue
        store_la[k] = old_result[k]
```

This produces a **complete output HDF5** covering all years from base year through `final_year`, with prior years coming from the checkpoint run and later years from the new run.

---

## Important Notes

### `cache_hh_seeds` Behavior

The `cache_hh_seeds` step only runs in the **first year of the loop** (not the first year of the full simulation). When resuming, this step runs in the first year of the resumed portion. This is by design.

### Column Dtype Consistency

When resuming, the code enforces that common columns match the base-store dtypes:
```python
common_cols = cols[cols.isin(ckpt_df.columns)]
ckpt_df[common_cols] = ckpt_df[common_cols].astype(dtypes[common_cols])
```
This prevents dtype mismatches from checkpoint-computed columns (e.g., `sqft_price_res` which exists in the checkpoint but not the raw HDF5).

### Working Store

The checkpoint creates a copy of the input HDF5 at `data/checkpoint_store.h5`. This file can be large (several GB). It is overwritten on each checkpoint resume.

### Checkpoint Table Skipping

Tables that appear in `buildings` but not in `tbs_to_update` list are **not** restored from checkpoint — they retain their base-year values. This includes most reference/lookup tables which should not change.

---

## Verifying a Checkpoint Resume

After starting a resumed run, check the console output for:
```
Loading table buildings from checkpoint year 2035...
Loading table households from checkpoint year 2035...
...
```

And verify that `run_start` is the expected year (should equal `checkpoint_year`, not `base_year`).
