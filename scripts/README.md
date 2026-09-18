# scripts/

Scripts that are not part of a forecast run. The live pipeline stays in the
repository root (`models.py`, `dataset.py`, `assumptions.py`, `utils.py`,
`lcm_utils.py`, `input_paths.py`, `output_indicators.py`, `simulation_2055.py`).

Configs were sorted the same way: `configs/` holds only what the pipeline loads
at runtime, and superseded ones moved to `configs/legacy/` (the MNL-era
`hlcm/`, `elcm/`, `repm/`, `hlcm_calib/`, `hlcm_2050/`, `elcm_2050/`, the unused
`yaml_configs*` variants, and the LCM/REPM constraint files).

## Running them

Python puts the *script's own directory* on `sys.path`, not the repository root,
so these need the root on the path explicitly. Run from the repository root:

```bash
PYTHONPATH=. python scripts/testing/test_developer.py
```

## `testing/`

Harnesses and one-off checks that exercise the current pipeline.

- `test_forecast_2050.py` — the 2050-round run script (`base_year` 2020). Kept
  as a harness; `simulation_2055.py` in the root is the entry point for the
  2055 forecast.
- `test_developer.py`, `test_lcm_simulation.py`, `test_mcd_hu_samping.py`,
  `test_travel_survey_vars.py`, `test_MNLDiscreteChoiceModel.py`,
  `proforma_run_test.py`, `refiner_run_test.py`, `output_indicators_run_test.py`

## `legacy/`

Superseded or broken; kept for reference, not expected to run.

- `Simulation.py`, `Simulation_no_scaling.py`, `Cost_Shift_test.py`, `test.py`,
  `Refiner.py`, `test_forecast_2050_base.py` — all reference tables and steps
  that no longer exist (`poi`, `pseudo_building`, `drop_pseudo_buildings`), so
  they cannot run against the 2055 inputs.
- `dcm_ard_libs.py`, `fit_large_MNL_LCM.py`, `testing_LCM_with_demo_data.py`,
  `location_choice/` — the MNL/ARD location-choice estimation stack, superseded
  by the PyTorch LCMs. Moved together because they import one another.
- `app.py`, `whatnots_2015.py`, `whatnots_2020.py`, `transcad.py`,
  `convert_pickle5_to_pickle4.py`, `copy_run_summary_tables.py`,
  `generating_testing_mcd_vacancy_table.py`, `proforma_generate_settings.py`,
  `variable_used_stats.py` — standalone utilities, untouched since 2020–2024.
