"""HLCM city factors (net-change calibration): a csv in configs named by
models.hlcm.calibration.city_factors; off when the setting is empty."""

import orca
import pandas as pd
import pytest

import lcm_utils


@pytest.fixture(autouse=True)
def fresh(monkeypatch, tmp_path):
    lcm_utils._CITY_FACTORS.clear()
    monkeypatch.setattr(lcm_utils.misc, "configs_dir", lambda: str(tmp_path))
    monkeypatch.setattr(lcm_utils.utils, "run_log", lambda *a, **k: None)
    orca.add_injectable("hlcm_model_path", "/models/models_A/")
    yield
    lcm_utils._CITY_FACTORS.clear()


def _write(tmp_path, model="models_A"):
    pd.DataFrame({"city_id": [4130, 4005], "factor": [0.9, 1.2], "hlcm_model": model}).to_csv(
        tmp_path / "hlcm_city_factors.csv", index=False)


def test_off_when_setting_empty(monkeypatch):
    monkeypatch.setenv("SEMCOG_HLCM_CITY_FACTORS", "/no/such/file.csv")   # no longer read
    assert lcm_utils._hlcm_city_factors({}) is None
    assert lcm_utils._hlcm_city_factors({"city_factors": ""}) is None


def test_read_from_configs_folder(tmp_path, capsys):
    _write(tmp_path)
    f = lcm_utils._hlcm_city_factors({"city_factors": "hlcm_city_factors.csv"})
    assert f.to_dict() == {4130: 0.9, 4005: 1.2}
    assert "WARNING" not in capsys.readouterr().out


def test_warns_when_model_set_differs(tmp_path, capsys):
    _write(tmp_path, model="models_OLD")
    lcm_utils._hlcm_city_factors({"city_factors": "hlcm_city_factors.csv"})
    assert "WARNING: factors calibrated for HLCM models_OLD, this run uses models_A" in capsys.readouterr().out


def test_missing_named_file_stops():
    with pytest.raises(FileNotFoundError):
        lcm_utils._hlcm_city_factors({"city_factors": "no_such.csv"})


def test_repo_file_matches_model_structure():
    # the shipped setting names a file that exists in the repo configs, with the required columns
    import os
    import yaml
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg = yaml.safe_load(open(os.path.join(repo, "configs", "model_structure.yaml")))
    name = cfg["models"]["hlcm"]["calibration"].get("city_factors")
    if name:
        df = pd.read_csv(os.path.join(repo, "configs", name))
        assert {"city_id", "factor", "hlcm_model"} <= set(df.columns)
        assert df.city_id.is_unique and (df.factor > 0).all()


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
