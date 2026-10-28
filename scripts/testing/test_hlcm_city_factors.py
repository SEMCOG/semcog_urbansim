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
