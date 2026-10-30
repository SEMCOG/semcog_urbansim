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


def _reset(name):
    if orca.is_injectable(name):
        orca.add_injectable(name, None)


def test_base_share_is_frozen_at_first_call():
    _reset("city_base_hh_share")
    first = models.city_base_share(SNAPSHOT, CITY_TO_LA)
    assert first.round(4).to_dict() == {10: 0.6, 20: 0.4, 30: 1.0}
    # city 10 doubles; the frozen share must not follow it
    grown = pd.Series({10: 120, 20: 40, 30: 50})
    assert models.city_base_share(grown, CITY_TO_LA).equals(first)
    _reset("city_base_hh_share")


def test_frozen_share_spreads_change_not_the_live_share():
    _reset("city_base_hh_share")
    share = models.city_base_share(SNAPSHOT, CITY_TO_LA)
    # city 10 has since taken all of LA 3's growth; LA 3 now 140, LA 5 unchanged
    grown = pd.Series({10: 90, 20: 40, 30: 50})
    hh_la = pd.Series([3] * 140 + [5] * 50)
    out = models.city_households_for_developer(hh_la, CITY_TO_LA, grown, None, share)
    # the 10 unplaced go 60/40 by base share, not 69/31 by the live one
    assert out[10] == 96 and out[20] == 44
    assert out.groupby(CITY_TO_LA).sum().to_dict() == {3: 140, 5: 50}
    _reset("city_base_hh_share")


def test_share_renormalizes_and_falls_back_where_absent():
    _reset("city_base_hh_share")
    hh_la = pd.Series([3] * 110 + [5] * 50)
    # LA 5 missing from the frozen share: city 30 falls back to the live share
    share = pd.Series({10: 0.3, 20: 0.1})
    out = models.city_households_for_developer(hh_la, CITY_TO_LA, SNAPSHOT, None, share)
    assert out[10] == 67.5 and out[20] == 42.5   # 0.3/0.4 and 0.1/0.4 of +10
    assert out[30] == 50
    _reset("city_base_hh_share")


def test_share_cap_holds_a_city_to_its_band():
    _reset("city_base_units")
    units = pd.Series({10: 600, 20: 400, 30: 500})
    cap = models.city_share_cap(units, CITY_TO_LA, 0.15)
    assert cap.round(2).to_dict() == {10: 690.0, 20: 460.0, 30: 575.0}
    # LA 3 stock grows to 1,200: the cap grows with it, the share does not
    grown = pd.Series({10: 800, 20: 400, 30: 500})
    assert models.city_share_cap(grown, CITY_TO_LA, 0.15).round(2)[10] == 828.0
    _reset("city_base_units")


def test_share_cap_off_when_band_is_zero():
    _reset("city_base_units")
    units = pd.Series({10: 600, 20: 400, 30: 500})
    assert models.city_share_cap(units, CITY_TO_LA, 0) is None
    assert models.city_share_cap(units, CITY_TO_LA, None) is None


def test_share_cap_never_below_the_stock_a_city_started_with():
    _reset("city_base_units")
    units = pd.Series({10: 600, 20: 400, 30: 500})
    models.city_share_cap(units, CITY_TO_LA, 0.15)
    # LA 3 stock halves; the band alone would cap city 10 at 345, below its 600
    shrunk = pd.Series({10: 600, 20: 0, 30: 500})
    assert models.city_share_cap(shrunk, CITY_TO_LA, 0.15)[10] == 600.0
    _reset("city_base_units")


def test_share_cap_leaves_a_city_with_no_base_stock_uncapped():
    _reset("city_base_units")
    units = pd.Series({10: 600, 20: 0, 30: 500})
    cap = models.city_share_cap(units, CITY_TO_LA, 0.15)
    assert pd.isna(cap[20])          # no base stock: uncapped, not frozen at zero
    assert cap[10] == 690.0
    _reset("city_base_units")
