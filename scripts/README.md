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
- pytest suites from the demolition review: `test_demolition_calibration.py`
  (calibration/simulation feature parity), `test_demolition_stock.py`
  (stock-rate volume targets), `test_city_households_for_developer.py`
  (residential demand counts unplaced households). These need `pytest`, which
  is not in the `forecast_py312` environment.
- `replay_utils.py` — shared helpers for in-process replay and paired tests
  (on/off, factor A/B from one saved state). Use them in every such test:
  `hold_data_tables()` first, then `snapshot()` / `restore()`, and
  `scale_column()` / `scale_by_group()` / `assert_changed()` so that each change
  is checked and the run stops if a change is lost. Self-test:
  `python -m pytest scripts/testing/test_replay_utils.py`.

## `estimation/`

Scripts that estimate or calibrate model inputs. They are not part of a forecast run;
the run reads only their output in `configs/`. Run them from the repository root
(they read `configs/` from the working directory):

- `demolition_calibration.py`: demolition score models and rates. Writes
  `configs/demolition_model.yaml`, `configs/demolition_rates.csv` and, with
  `--stock-rates-only`, only `configs/demolition_stock_rates.csv`.
  `PYTHONPATH=. python scripts/estimation/demolition_calibration.py [--stock-rates-only]`
- `estimate_developer_selection.py`: residential developer site-selection model.
  Writes `configs/developer_selection_coefs.yaml`. Its features must match
  `models.py` (the site-selection feature code).
  `PYTHONPATH=. python scripts/estimation/estimate_developer_selection.py`
- `make_city_factors.py`: HLCM city factors (calibration of each city's household net
  change in the first simulated years). Writes `configs/hlcm_city_factors.csv`, which
  `configs/model_structure.yaml` (`models.hlcm.calibration.city_factors`) names; an empty
  setting runs without factors. Recalibrate after every HLCM re-estimation:
  1. Run the model for the calibration window (for example 2026–2030), with the current
     factors or with the setting empty.
  2. `PYTHONPATH=. python scripts/estimation/make_city_factors.py --runs <run h5>[,<run h5>]
     --first-year <first> --last-year <last> --hlcm-model <HLCM model folder>
     [--previous configs/hlcm_city_factors.csv]`
  3. Repeat until the change is within the seed noise.

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
