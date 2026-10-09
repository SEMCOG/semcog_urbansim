"""Central registry of external input (and a couple of output) locations.

Every external file/dir the **forecast simulation** reads is defined here once,
as a list of candidate locations in priority order; the first that exists is
used (`utils.first_existing_path`). This keeps all data paths in one place and
makes the model portable across environments.

How to point at different data:
  - Local copy = `_LOCAL` (e.g. a `d_drive/forecast_inputs/base_year` copy),
    listed first.
  - Mounted network drives (`/mnt/...`) are the fallback when the local copy is
    not available.
  - To move the local copy, set the `SEMCOG_INPUT_DIR` env var or edit `_LOCAL`.
  - To add another environment, add a candidate to the relevant entry.

Out of scope (left as-is): the alternate estimation / test / notebook scripts
(`HLCM_estimation.py`, `test_developer.py`, `notebooks/*`, …) use a different
`/home/da/share/...` layout and their own data versions.
"""
import hashlib
import os
from pathlib import Path

import utils

# Local primary copy of the inputs (override per machine).
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
    "/mnt/hgfs/urbansim/RDF2055/model_inputs/base_hdf/main_100926.h5",
    f"{_LOCAL}/main_100926.h5",
)

# RDF2055 estimation
HLCM_MODEL_DIR = _p(
    "/mnt/D/RDF2055/estimation/models/models_26Oct08",
    "/mnt/hgfs/urbansim/RDF2055/estimation/models/models_26Oct08",
    f"{_LOCAL}/models/models_26Oct08",
)
ELCM_MODEL_DIR = _p(
    "/mnt/D/RDF2055/estimation/models/elcm_models_26Oct02",
    "/mnt/hgfs/urbansim/RDF2055/estimation/models/elcm_models_26Oct02",
    f"{_LOCAL}/models/elcm_models_26Oct02",
)
REPM_MODEL_DIR = _p(
    "/mnt/hgfs/urbansim/RDF2055/model_inputs/estimation/repm_oct03",
    f"{_LOCAL}/models/repm_oct03",
)


ACCESS_INDICATORS_H5 = _p(
    f"{_LOCAL}/access_indicators.h5",
    "/mnt/hgfs/urbansim/RDF2055/model_inputs/base_hdf/access_indicators.h5",
)

# Pandana network bundle — normally in the local `data/` dir; fall back to copy.
NETWORKS_2050_H5 = _p(
    f"{_LOCAL}/semcog_networks.h5",
    "/mnt/hgfs/urbansim/RDF2055/model_inputs/base_hdf/semcog_networks.h5",
)

# ---------------------------------------------------------------------------
# Travel survey (block-group variable build; not part of every run)
# ---------------------------------------------------------------------------
TRAVEL_SURVEY_DIR = _p(
    f"{_LOCAL}/travel_survey/Full_Dataset_HTS_Uni_2026-06-11",
    "/mnt/D/RDF2055/input_data/travel_survey/Full_Dataset_HTS_Uni_2026-06-11",
)

# ---------------------------------------------------------------------------
# Scenario controls (only used when ENABLE_SCENARIO is True)
# ---------------------------------------------------------------------------
_SCEN = "/mnt/hgfs/urbansim/RDF2050/scenarios/controls/low_immigration"
_SCEN_LOCAL = f"{_LOCAL}/scenarios/low_immigration"
SCENARIO_HH_CONTROL_CSV = _p(
    f"{_SCEN_LOCAL}/annual_household_control_totals_2050_07232024.csv",
    f"{_SCEN}/annual_household_control_totals_2050_07232024.csv",
)
SCENARIO_REMI_POP_CSV = _p(
    f"{_SCEN_LOCAL}/remi_total_pop_la07232024.csv",
    f"{_SCEN}/remi_total_pop_la07232024.csv",
)
SCENARIO_EMP_CONTROL_CSV = _p(
    f"{_SCEN_LOCAL}/annual_employment_control_totals.csv",
    f"{_SCEN}/annual_employment_control_totals.csv",
)

# ---------------------------------------------------------------------------
# Output destination (optional run-archive copy; consumer guards on existence)
# ---------------------------------------------------------------------------
MODEL_RUNS_DIR = _p(
    f"{_LOCAL}/model_runs",
    "/mnt/hgfs/urbansim/RDF2055/model_runs",
)


# These pairs are checked on demand rather than at every simulation startup.
# A full SHA-256 pass over large HDFs is intentionally a preflight operation.
MIRROR_PAIRS = {
    "base HDF": (
        f"{_LOCAL}/main_100926.h5",
        ["/mnt/hgfs/urbansim/RDF2055/model_inputs/base_hdf/main_100926.h5"],
    ),
    "accessibility HDF": (
        f"{_LOCAL}/access_indicators.h5",
        ["/mnt/hgfs/urbansim/RDF2055/model_inputs/base_hdf/access_indicators.h5"],
    ),
    "network bundle": (
        f"{_LOCAL}/semcog_networks.h5",
        ["/mnt/hgfs/urbansim/RDF2055/model_inputs/base_hdf/semcog_networks.h5"],
    ),
    "HLCM package": (
        f"{_LOCAL}/models/models_26Oct08",
        [
            "/mnt/D/RDF2055/estimation/models/models_26Oct08",
            "/mnt/hgfs/RDF2055/estimation/models/models_26Oct08",
        ],
    ),
    "ELCM package": (
        f"{_LOCAL}/models/elcm_models_26Sep11",
        [
            "/mnt/D/RDF2055/estimation/models/elcm_models_26Sep11",
            "/mnt/hgfs/RDF2055/estimation/models/elcm_models_26Sep11",
        ],
    ),
}


def _sha256(path):
    """Hash a file or a directory's sorted relative-file manifest."""
    path = Path(path)
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for file_path in files:
        digest.update(str(file_path.relative_to(path.parent)).encode())
        with file_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def verify_input_mirrors():
    """Return False when an available local/remote mirror differs by SHA-256."""
    consistent = True
    for label, (local_path, remote_paths) in MIRROR_PAIRS.items():
        available = [path for path in [local_path, *remote_paths] if os.path.exists(path)]
        if len(available) < 2:
            print(f"SKIP {label}: fewer than two mirrors are available")
            continue
        local_hash = _sha256(local_path)
        for remote_path in remote_paths:
            if not os.path.exists(remote_path):
                continue
            matches = local_hash == _sha256(remote_path)
            print(f"{'OK' if matches else 'MISMATCH'} {label}: {remote_path}")
            consistent &= matches
    return consistent


if __name__ == "__main__" or os.environ.get("SEMCOG_VERIFY_INPUT_MIRRORS") == "1":
    if not verify_input_mirrors():
        raise SystemExit("Input mirror verification failed")
