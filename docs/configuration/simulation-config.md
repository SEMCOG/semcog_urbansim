# Simulation Configuration

All run-level configuration is set at the top of [`test_forecast_2050.py`](../../test_forecast_2050.py) before `import models`. Changes here affect the current run only — they are not persisted to any config file except `run_config.yaml` (written to the run output directory).

---

## Run Parameters

```python
base_year = 2020          # Base year (data vintage)
final_year = 2050         # Last year to simulate
indicator_spacing = 5     # Compute indicators every N years
upload_to_carto = True    # Upload results to CartoDB after run
run_debug = False         # Enable debug logging
add_2019 = True           # Include 2019 data in indicator output
```

---

## Model Paths

```python
orca.add_injectable('hlcm_model_path',
    '/mnt/hgfs/RDF2050/estimation/models/models_survey_finetune')

orca.add_injectable('elcm_model_path',
    '/mnt/hgfs/RDF2050/estimation/models/elcm_models_25May30/')

orca.add_injectable('yaml_configs', 'yaml_configs_elcm_hlcm.yaml')
```

| Injectable | Description |
|---|---|
| `hlcm_model_path` | Directory containing HLCM `.pt` files in a `pts/` subdirectory |
| `elcm_model_path` | Directory containing ELCM `.pt` files in a `pts/` subdirectory |
| `yaml_configs` | YAML file listing all model filenames (relative to project root) |

To switch to a different set of models (e.g., after re-estimation), update these paths. The `yaml_configs` file must list the `.pt` filenames present in the `pts/` subdirectory.

---

## Scenario Controls

```python
orca.add_injectable('ENABLE_SCENARIO', False)

# Only used when ENABLE_SCENARIO = True:
orca.add_injectable('scenario_hh_control_path',
    '.../low_immigration/annual_household_control_totals_2050_07232024.csv')
orca.add_injectable('scenario_remi_total_pop',
    '.../low_immigration/remi_total_pop_la07232024.csv')
orca.add_injectable('scenario_emp_control_path',
    '.../low_immigration/annual_employment_control_totals.csv')
```

Set `ENABLE_SCENARIO = True` to replace the baseline control totals with scenario-specific ones. See [Scenario Controls](scenarios.md).

---

## Checkpoint / Resume

```python
orca.add_injectable('use_checkpoint', False)
orca.add_injectable('runnum_to_resume', 'run1365.h5')
```

Set `use_checkpoint = True` to resume from the last completed year of `runnum_to_resume`. See [Checkpoints & Resume](checkpoints.md).

---

## Run Output Flags

```python
RUN_OUTPUT_INDICATORS = True   # Run output_indicators.py after simulation
```

Set to `False` to skip indicator aggregation (useful for quick test runs where you only need the HDF5 output).

---

## Auto-Saved Config

At the start of each run, a `run_config.yaml` is written to `runs/runNNN/`. This captures:

```yaml
RUN NUMBER: runs/runNNN.h5
hlcm_model_path: /mnt/hgfs/RDF2050/estimation/models/models_survey_finetune
elcm_model_path: /mnt/hgfs/RDF2050/estimation/models/elcm_models_25May30/
yaml_configs: yaml_configs_elcm_hlcm.yaml
base_year: 2020
final_year: 2050
ENABLE_SCENARIO: false
use_checkpoint: false
runnum_to_resume: run1365.h5
repm_model_type: XGBoost
git_branch_name: forecast_2055
git_commit_id: <full SHA>
```

This makes every run **reproducible** — you can always trace which model version and configuration produced a given output file.

---

## Random Seed

The random seed is set in `assumptions.py`:

```python
seed = 271828
random.seed(seed)
np.random.seed(seed)
```

Using the same seed with the same model files and input data will produce identical results across runs. The seed value is logged via `utils.run_log()`.

---

## Injectables Set at Import Time

When `models.py` is imported, additional injectables are set that affect simulation behavior:

| Injectable | Set By | Default |
|---|---|---|
| `hlcm_step_names` | `models.py` | Sorted list of HLCM model names |
| `elcm_step_names` | `models.py` | ELCM model names sorted by sector order |
| `repm_step_names` | `models.py` | Sorted list of REPM model directory names |
| `hh_location_choice_models` | `models.py` | Dict of loaded PyTorch HLCM models |
| `emp_location_choice_models` | `models.py` | Dict of loaded PyTorch ELCM models |
| `mcd_hu_sampling_config` | `models.py` | Contents of `configs/mcd_hu_sampling.yaml` |

These are derived from the path injectables above — changing `hlcm_model_path` will cause different models to be loaded and registered.
