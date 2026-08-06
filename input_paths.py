"""Central registry of external input (and a couple of output) locations.

Every external file/dir the **forecast simulation** reads is defined here once,
as a list of candidate locations in priority order; the first that exists is
used (`utils.first_existing_path`). This keeps all data paths in one place and
makes the model portable across environments.

How to point at different data:
  - Production = the mounted network drives (`/mnt/hgfs/...`), listed first.
  - Local copy = `_LOCAL` (e.g. a `d_drive/forecast_inputs/base_year` copy used
    when the network mounts are absent), listed as the fallback.
  - To move the local copy, set the `SEMCOG_INPUT_DIR` env var or edit `_LOCAL`.
  - To add another environment, add a candidate to the relevant entry.

Out of scope (left as-is): the alternate estimation / test / notebook scripts
(`HLCM_estimation.py`, `test_developer.py`, `notebooks/*`, …) use a different
`/home/da/share/...` layout and their own data versions.
"""
import os

import utils

# Local fallback copy of the inputs (override per machine).
_LOCAL = os.environ.get(
    "SEMCOG_INPUT_DIR", "/mnt/D/RDF2055/forecast_inputs/base_year"
)


def _p(*candidates):
    """First existing candidate, else the first (so errors name the canonical path)."""
    return utils.first_existing_path(*candidates)


# ---------------------------------------------------------------------------
# Core run inputs (required for a forecast run)
# ---------------------------------------------------------------------------
BASE_HDF = _p(
    "/mnt/hgfs/urbansim/RDF2055/model_inputs/base_hdf/main_080526.h5",
    f"{_LOCAL}/main_080526.h5",
    # 2050 base, fallback only
    "/mnt/hgfs/urbansim/RDF2050/model_inputs/base_hdf/forecast_data_input.h5",
    f"{_LOCAL}/forecast_data_input.h5",
)

# 2020 base- RDF2050 — used only to derive the 2020->2025 block-group
# household base trend for bg_hh_increase (see dataset.bg_hh_increase).
BG_HH_2020_HDF = _p(
    "/mnt/hgfs/urbansim/RDF2050/model_inputs/base_hdf/forecast_data_input.h5",
    f"{_LOCAL}/forecast_data_input.h5",
    "/mnt/hgfs/urbansim/RDF2050/model_inputs/base_hdf/forecast_data_input_031523.h5",
    f"{_LOCAL}/forecast_data_input_031523.h5",
)

HLCM_MODEL_DIR = _p(
    "/mnt/hgfs/RDF2050/estimation/models/models_survey_finetune",
    f"{_LOCAL}/models/models_survey_finetune",
)
ELCM_MODEL_DIR = _p(
    "/mnt/hgfs/RDF2050/estimation/models/elcm_models_25May30/",
    f"{_LOCAL}/models/elcm_models_25May30/",
)

ACCESS_INDICATORS_H5 = _p(
    "/mnt/hgfs/urbansim/RDF2055/model_inputs/base_hdf/access_indicators.h5",
    f"{_LOCAL}/access_indicators.h5",
)

# legacy CSVs, fallback only -- superseded by ACCESS_INDICATORS_H5 (same data,
# documented fill values applied: 95/125/155 min near-max, 0 cumulative/gravity)
_ACCESS = "/mnt/hgfs/urbansim/Accessibility/access_to_core_2024/outputs_model/indicators"
ACCESS_WALK_CSV = _p(
    f"{_ACCESS}/walk/walk_indicators_by_parcel_20251111.csv",
    f"{_LOCAL}/indicators/walk/walk_indicators_by_parcel_20251111.csv",
)
ACCESS_BIKE_CSV = _p(
    f"{_ACCESS}/bike/bike_indicators_by_parcel_20251111.csv",
    f"{_LOCAL}/indicators/bike/bike_indicators_by_parcel_20251111.csv",
)
ACCESS_DRIVE_CSV = _p(
    f"{_ACCESS}/drive/drive_indicators_by_parcel_20251111.csv",
    f"{_LOCAL}/indicators/drive/drive_indicators_by_parcel_20251111.csv",
)

# Pandana network bundle — normally in the local `data/` dir; fall back to copy.
NETWORKS_2050_H5 = _p(
    "/mnt/hgfs/urbansim/RDF2055/model_inputs/base_hdf/semcog_networks.h5",
    os.path.join(os.path.dirname(__file__), "data", "semcog_networks.h5"),
    f"{_LOCAL}/semcog_networks.h5",
    # legacy filename, fallback only -- identical content
    os.path.join(os.path.dirname(__file__), "data", "semcog_2050_networks.h5"),
    f"{_LOCAL}/semcog_2050_networks.h5",
)

# ---------------------------------------------------------------------------
# Travel survey (block-group variable build; not part of every run)
# ---------------------------------------------------------------------------
TRAVEL_SURVEY_DIR = _p(
    "/mnt/D/RDF2055/input_data/travel_survey/Full_Dataset_HTS_Uni_2026-06-11",
    f"{_LOCAL}/travel_survey/Full_Dataset_HTS_Uni_2026-06-11",
)

# ---------------------------------------------------------------------------
# Scenario controls (only used when ENABLE_SCENARIO is True)
# ---------------------------------------------------------------------------
_SCEN = "/mnt/hgfs/urbansim/RDF2050/scenarios/controls/low_immigration"
_SCEN_LOCAL = f"{_LOCAL}/scenarios/low_immigration"
SCENARIO_HH_CONTROL_CSV = _p(
    f"{_SCEN}/annual_household_control_totals_2050_07232024.csv",
    f"{_SCEN_LOCAL}/annual_household_control_totals_2050_07232024.csv",
)
SCENARIO_REMI_POP_CSV = _p(
    f"{_SCEN}/remi_total_pop_la07232024.csv",
    f"{_SCEN_LOCAL}/remi_total_pop_la07232024.csv",
)
SCENARIO_EMP_CONTROL_CSV = _p(
    f"{_SCEN}/annual_employment_control_totals.csv",
    f"{_SCEN_LOCAL}/annual_employment_control_totals.csv",
)

# ---------------------------------------------------------------------------
# Historical / back-cast (optional analysis, not the forward run)
# ---------------------------------------------------------------------------
HDF_INPUT_2045 = _p(
    "/mnt/hgfs/urbansim/RDF2045/data/base_year/all_semcog_data_02-02-18-final-forecast-pd3.h5",
    f"{_LOCAL}/all_semcog_data_02-02-18-final-forecast.h5",
)
FORECAST_INPUT_2040 = _p(
    "/mnt/hgfs/urbansim/RDF2050/model_improvements/2024_spring/2010_data",
    f"{_LOCAL}/model_improvements/2024_spring/2010_data",
)

BUILDING_TO_ZONE_CSV = _p(
    os.path.join(os.path.dirname(__file__), "data", "building_to_zone_baseyear_2020_shrink.csv"),
    f"{_LOCAL}/building_to_zone_baseyear_2020_shrink.csv",
)

ACS_BG_HH_CSV = _p(
    os.path.join(os.path.dirname(__file__), "data", "ACS_HH_14_19_BG.csv"),
    f"{_LOCAL}/ACS_HH_14_19_BG.csv",
)

# ---------------------------------------------------------------------------
# Output destination (optional run-archive copy; consumer guards on existence)
# ---------------------------------------------------------------------------
MODEL_RUNS_DIR = _p(
    "/mnt/hgfs/urbansim/RDF2050/model_runs",
    f"{_LOCAL}/model_runs",
)
