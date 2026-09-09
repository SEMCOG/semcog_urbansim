"""Cache for the REPM training matrix produced by build_networks + neighborhood_vars + _load_data.

REPM estimation/diagnostic scripts re-run build_networks and neighborhood_vars
against the *same* base-year data on every iteration, just to test one model
spec. That step is a pandana network computation (precompute() on the walk
network's ~936k nodes dominates it) costing on the order of 30-45 minutes and
~90GB RAM -- expensive to redo per experiment. This caches the
(vars_used, mat) pair that xgb_training._load_data() produces from it,
and reuses that cache as long as the inputs feeding the computation haven't
changed.

Scope: REPM estimation / diagnostic scripts only. Do NOT use this in the
annual forecast simulation (test_forecast_2050.py) -- there, accessibility
variables must be recomputed every simulated year as buildings/jobs/
households change.
"""
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import scipy.sparse

import input_paths

CACHE_DIR = Path("/home/da/RDF2055/d_drive/runs/cache")
CACHE_DATA_PATH = CACHE_DIR / "repm_training_matrix.h5"
CACHE_META_PATH = CACHE_DIR / "repm_training_matrix.meta.json"

# Bump this by hand if build_networks / neighborhood_vars / _load_data logic
# changes in a way not captured by the file fingerprint below (e.g. a new
# variable_definitions entry added inline in models.py, a change to
# _should_skip_var, a change to how nodeid_walk/nodeid_drv are assigned).
# The fingerprint below only watches *data and config* files, not code.
CACHE_SCHEMA_VERSION = 5

_CONFIG_YAMLS = [
    "configs/networks_walk.yaml",
    "configs/networks_drv.yaml",
    "configs/available_networks_2050.yaml",
]

# Set to True to bypass the cache and always recompute (also overwrites it).
FORCE_RECOMPUTE = False


def _file_stat_fingerprint(path):
    st = os.stat(path)
    return {"path": str(path), "mtime": st.st_mtime, "size": st.st_size}


def _file_hash_fingerprint(path):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest}


def _current_fingerprint():
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "data_files": [
            _file_stat_fingerprint(input_paths.BASE_HDF),
            _file_stat_fingerprint(input_paths.NETWORKS_2050_H5),
        ],
        "config_files": [_file_hash_fingerprint(p) for p in _CONFIG_YAMLS],
    }


def is_valid():
    """Whether a cached matrix exists and matches the current inputs."""
    if FORCE_RECOMPUTE:
        return False
    if not (CACHE_DATA_PATH.exists() and CACHE_META_PATH.exists()):
        return False
    try:
        cached_fp = json.loads(CACHE_META_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return cached_fp == _current_fingerprint()


def load():
    """Load a previously cached (vars_used, mat) pair. Call is_valid() first."""
    df = pd.read_hdf(CACHE_DATA_PATH, key="mat")
    vars_used = list(df.index)
    mat = scipy.sparse.csr_matrix(df.values)
    return vars_used, mat


def save(vars_used, mat):
    """Cache a (vars_used, mat) pair as produced by xgb_training._load_data()."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dense = pd.DataFrame(mat.toarray(), index=vars_used)
    dense.to_hdf(CACHE_DATA_PATH, key="mat", mode="w", format="fixed")
    CACHE_META_PATH.write_text(json.dumps(_current_fingerprint(), indent=2))
    print(f"Cached REPM training matrix to {CACHE_DATA_PATH}")
