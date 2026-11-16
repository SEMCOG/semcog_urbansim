"""Unit tests for the HLCM tract pattern prior (lcm_utils._hlcm_pattern_prior, 2026-11-16),
its checkpoint storage, and the city-factor copy in configs_used.

Run from the repo root: PYTHONPATH=. python -m pytest scripts/testing/test_pattern_prior.py -q
"""
import os

import numpy as np
import orca
import pandas as pd
import pytest
import yaml

import checkpoint_targets
import utils
from lcm_utils import _hlcm_pattern_prior

SEG = "tract_hh_type_ratio_children_has_children_senior_without_seniors_income_lowinc"


@pytest.fixture
def shares():
    orca.add_injectable("hlcm_la_segment_base_shares", pd.Series({(3, SEG): 0.10, (5, SEG): 0.0}))
    yield
    orca.orca._INJECTABLES.pop("hlcm_la_segment_base_shares", None)  # a plain value, not a cached function


def harness(base_ratios, la_share, beta):
    """The formula of the test-harness wrapper that run5175 used (sim_prior.py, margin '')."""
    r = np.nan_to_num(np.asarray(base_ratios, dtype="float64"), nan=0.0) / la_share
    return np.clip(r, 0.5, 2.0) ** beta


def test_off_or_missing_is_neutral(shares):
    b = np.array([0.02, 0.1, 0.4])
    assert _hlcm_pattern_prior(None, b, 3, SEG) == 1.0
    assert _hlcm_pattern_prior({"enabled": False, "beta": 1.0}, b, 3, SEG) == 1.0


def test_same_as_harness(shares):
    b = np.array([0.0, 0.02, 0.05, 0.1, 0.15, 0.4, np.nan])
    for beta in (1.0, 0.5, 2.0):
        got = _hlcm_pattern_prior({"enabled": True, "beta": beta}, b, 3, SEG)
        np.testing.assert_allclose(got, harness(b, 0.10, beta))
    got = _hlcm_pattern_prior({"enabled": True, "beta": 1.0}, b, 3, SEG)
    np.testing.assert_allclose(got, [0.5, 0.5, 0.5, 1.0, 1.5, 2.0, 0.5])  # clip 0.5..2, nan -> 0


def test_segment_without_households_is_neutral(shares):
    assert _hlcm_pattern_prior({"enabled": True}, np.array([0.1]), 5, SEG) == 1.0
    assert _hlcm_pattern_prior({"enabled": True}, np.array([0.1]), 99, SEG) == 1.0


def test_beta_must_be_positive(shares):
    with pytest.raises(ValueError):
        _hlcm_pattern_prior({"enabled": True, "beta": 0}, np.array([0.1]), 3, SEG)


def _targets():
    return {"tract_hh_type_base_ratios": pd.DataFrame({"a": [0.1, 0.2]}),
            "new_unit_fill_shares": pd.DataFrame({"x": [1.0]}),
            "new_unit_fill_ratios": pd.DataFrame({"y": [2.0]}),
            "sqft_band_cuts": {3: np.array([1.0, 2.0])},
            "job_btype_baseyear_prob_matrix": pd.DataFrame({1: [0.5, 0.5]})}


def test_checkpoint_keeps_the_shares(tmp_path):
    p = str(tmp_path / "run.h5")
    s = pd.Series({(3, SEG): 0.1, (5, SEG): 0.2})
    checkpoint_targets.save(p, {**_targets(), "hlcm_la_segment_base_shares": s})
    with pd.HDFStore(p, "r") as st:
        back = checkpoint_targets.read(st)
    pd.testing.assert_series_equal(back["hlcm_la_segment_base_shares"], s)


def test_older_checkpoint_without_the_shares_still_reads(tmp_path):
    p = str(tmp_path / "old.h5")
    checkpoint_targets.save(p, _targets())
    with pd.HDFStore(p, "r") as st:
        back = checkpoint_targets.read(st)
    assert back["hlcm_la_segment_base_shares"] is None


def test_configs_used_copies_the_named_city_factors(tmp_path):
    cfg = tmp_path / "configs"
    cfg.mkdir()
    ms = {"models": {"hlcm": {"calibration": {"city_factors": "my_factors.csv"}}}}
    (cfg / "model_structure.yaml").write_text(yaml.safe_dump(ms))
    assert utils._city_factors_path(str(cfg)) == os.path.join(str(cfg), "my_factors.csv")
    ms["models"]["hlcm"]["calibration"]["city_factors"] = "/abs/f.csv"
    (cfg / "model_structure.yaml").write_text(yaml.safe_dump(ms))
    assert utils._city_factors_path(str(cfg)) == "/abs/f.csv"
    ms["models"]["hlcm"]["calibration"]["city_factors"] = None
    (cfg / "model_structure.yaml").write_text(yaml.safe_dump(ms))
    assert utils._city_factors_path(str(cfg)).endswith("hlcm_city_factors.csv")
