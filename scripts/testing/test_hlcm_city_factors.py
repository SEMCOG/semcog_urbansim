"""HLCM city factors (net-change calibration): off unless configured; read from config or env."""

import pandas as pd

import lcm_utils


def test_off_without_config(monkeypatch):
    monkeypatch.delenv("SEMCOG_HLCM_CITY_FACTORS", raising=False)
    assert lcm_utils._hlcm_city_factors({}) is None


def test_read_from_env_and_config(tmp_path, monkeypatch):
    f = tmp_path / "factors.csv"
    pd.DataFrame({"city_id": [4130, 4005], "factor": [0.9, 1.2]}).to_csv(f, index=False)
    monkeypatch.setenv("SEMCOG_HLCM_CITY_FACTORS", str(f))
    assert lcm_utils._hlcm_city_factors({}).to_dict() == {4130: 0.9, 4005: 1.2}
    monkeypatch.delenv("SEMCOG_HLCM_CITY_FACTORS")
    assert lcm_utils._hlcm_city_factors({"city_factors": str(f)})[4130] == 0.9


def test_city_factor_in_log_scores_moves_stratified_counts():
    # stratified placement: the per-tract counts follow the log-score mass, so the
    # city factor (added to the log scores) must shift slots toward the favored city
    from lcm_utils import _hlcm_stratified_counts
    import numpy as np
    tracts = np.array([1] * 50 + [2] * 50)          # tract 1 in city A, tract 2 in city B
    ls = np.zeros(100)
    base, _ = _hlcm_stratified_counts(tracts, ls, pd.Series({1: 1.0, 2: 1.0}), 20)
    city = np.where(tracts == 1, np.log(2.0), 0.0)  # city A factor 2
    tilt, _ = _hlcm_stratified_counts(tracts, ls + city, pd.Series({1: 1.0, 2: 1.0}), 20)
    assert base[1] == 10 and tilt[1] > base[1]
