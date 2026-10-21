"""Regression test: residential developer city demand counts unplaced households.

households_transition adds tens of thousands of households each year that stay
unplaced (no city) until HLCM. Counting only placed households at developer time
inflated city vacancy and drove the raw demand target negative almost everywhere.
"""

import os
from pathlib import Path

os.environ.setdefault(
    "SEMCOG_INPUT_DIR",
    str(Path(__file__).resolve().parents[3] / "d_drive/forecast_inputs/base_year"),
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


def _two_city_momentum(year, add_a, add_b, fade=0.8):
    """Momentum for two cities of one large area that gained add_a / add_b since 2020."""
    hh_2020 = pd.read_csv("configs/households_by_city_2020.csv", index_col="city_id")["households"]
    a, b = hh_2020.index[:2]
    snap = pd.Series({a: hh_2020[a] + add_a, b: hh_2020[b] + add_b})
    out = models.city_excess_momentum(year, snap, pd.Series({a: 3, b: 3}), fade)
    return out, a, b, hh_2020


def _reset_momentum():
    if orca.is_injectable("city_excess_momentum"):
        orca.add_injectable("city_excess_momentum", None)


def test_momentum_is_excess_over_large_area_share():
    _reset_momentum()
    out, a, b, hh_2020 = _two_city_momentum(2026, 500.0, 0.0, fade=1.0)
    share_a = hh_2020[a] / (hh_2020[a] + hh_2020[b])
    # city a: 100/yr raw; large area: 100/yr; excess = 100 - share_a * 100
    assert abs(out[a] - (100.0 - share_a * 100.0)) < 1e-9
    assert abs(out.sum()) < 1e-9  # zero-sum in the large area


def test_growth_at_large_area_share_has_no_momentum():
    _reset_momentum()
    hh_2020 = pd.read_csv("configs/households_by_city_2020.csv", index_col="city_id")["households"]
    a, b = hh_2020.index[:2]
    out, _, _, _ = _two_city_momentum(2026, hh_2020[a] * 0.05, hh_2020[b] * 0.05)
    assert out.abs().max() < 1e-9


def test_momentum_fades_and_ignores_simulated_growth():
    _reset_momentum()
    first, a, b, _ = _two_city_momentum(2026, 500.0, 0.0)
    # later snapshots (simulated growth) must not renew the momentum
    later, _, _, _ = _two_city_momentum(2028, 5_000.0, 0.0)
    assert abs(later[a] - first[a] * 0.8 ** 2) < 1e-9
    _reset_momentum()
