"""Regression test: residential developer city demand counts unplaced households.

households_transition adds tens of thousands of households each year that stay
unplaced (no city) until HLCM. Counting only placed households at developer time
inflated city vacancy and drove the raw demand target negative almost everywhere.
"""

import os
from pathlib import Path

os.environ.setdefault(
    "SEMCOG_INPUT_DIR",
    str(Path(__file__).resolve().parents[1] / "d_drive/forecast_inputs/base_year"),
)

import orca
import pandas as pd

orca.add_injectable("data_out_dir", "/tmp/city_households_for_developer_test")

import models

CITY_TO_LA = pd.Series({10: 3, 20: 3, 30: 5})
SNAPSHOT = pd.Series({10: 60, 20: 40, 30: 50})


def test_la_change_spread_by_snapshot_share():
    # LA 3 grew from 100 to 110 households (10 unplaced); LA 5 lost 5
    hh_la = pd.Series([3] * 110 + [5] * 45)
    out = models.city_households_for_developer(hh_la, CITY_TO_LA, SNAPSHOT)

    assert out[10] == 66 and out[20] == 44
    assert out[30] == 45


def test_large_area_totals_match_current_households():
    hh_la = pd.Series([3] * 97 + [5] * 58)
    out = models.city_households_for_developer(hh_la, CITY_TO_LA, SNAPSHOT)

    assert out.groupby(CITY_TO_LA).sum().to_dict() == {3: 97, 5: 58}


def test_no_change_returns_snapshot():
    hh_la = pd.Series([3] * 100 + [5] * 50)
    out = models.city_households_for_developer(hh_la, CITY_TO_LA, SNAPSHOT)

    assert out.equals(SNAPSHOT.astype(float))


def test_trend_with_large_area_total_kept():
    # The worked example: LA 3 loses 100 households; city trends +300, +100, -200
    city_to_la = pd.Series({1: 3, 2: 3, 3: 3})
    snapshot = pd.Series({1: 50_000, 2: 30_000, 3: 20_000})
    trend = pd.Series({1: 300.0, 2: 100.0, 3: -200.0})
    hh_la = pd.Series([3] * 99_900)
    out = models.city_households_for_developer(hh_la, city_to_la, snapshot, trend)

    assert out.round().to_dict() == {1: 50_150, 2: 30_010, 3: 19_740}
    assert round(out.sum()) == 99_900


def test_trend_history_starts_from_2020():
    hh_2020 = pd.read_csv("configs/households_by_city_2020.csv", index_col="city_id")["households"]
    city = hh_2020.index[0]
    if orca.is_injectable("city_hh_history"):
        orca.add_injectable("city_hh_history", None)

    snap_2025 = pd.Series({city: hh_2020[city] + 500.0})
    t_2026 = models.city_household_trend(2026, snap_2025)
    assert t_2026[city] == 100.0  # (2025 - 2020) / 5

    snap_2026 = pd.Series({city: hh_2020[city] + 700.0})
    t_2027 = models.city_household_trend(2027, snap_2026)
    assert t_2027[city] == (700.0 - 100.0) / 5  # 2021 is the straight-line value: 2020 + 100
