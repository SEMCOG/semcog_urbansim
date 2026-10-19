"""Unit tests for the stock-based demolition targets (demolition_stock.py).

Plain pandas, no orca and no input data: fast. The orca-level checks are in
test_demolition_calibration.py.
"""
import os

import numpy as np
import pandas as pd
import pytest

import demolition_stock as ds

CFG = {"detroit": {"large_area_ids": [5], "rate_per_vacant_unit": 0.042},
       "min_age_eligible": 10}
P = ds.stock_rate_cfg(CFG)
RATES_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "demolition_stock_rates.csv")


def _buildings(**overrides):
    row = dict(city_id=501, building_type_id=81, residential_units=1, non_residential_sqft=0,
               households=0, building_age=80, sp_filter=0, event_id=0, year_built=1940)
    row.update(overrides)
    return pd.DataFrame([row])


@pytest.mark.parametrize("change, expected", [
    ({}, True),                                   # vacant Detroit house
    ({"households": 1}, False),                   # occupied
    ({"residential_units": 5}, False),            # 5+ units
    ({"building_type_id": 84}, False),            # not a pool type
    ({"non_residential_sqft": 500}, False),       # mixed use
    ({"building_age": 5}, False),                 # too new
    ({"sp_filter": -1}, False),                   # protected
    ({"event_id": 12}, False),                    # event building
])
def test_vacant_pool_mask(change, expected):
    b = _buildings(**change)
    assert bool(ds.vacant_pool_mask(b, pd.Series([5]), P).iat[0]) is expected


def test_vacant_pool_only_in_detroit():
    assert not ds.vacant_pool_mask(_buildings(), pd.Series([125]), P).iat[0]


def test_stock_by_city_uses_pool_in_detroit_and_old_units_elsewhere():
    b = pd.concat([
        _buildings(city_id=501, households=0),                 # Detroit, pool
        _buildings(city_id=501, households=1),                 # Detroit, occupied: not pool
        _buildings(city_id=2005, households=1),                # other: old unit counts
        _buildings(city_id=2005, year_built=1990, households=1),  # other: new, not counted
        _buildings(city_id=2005, building_type_id=31, residential_units=0,
                   non_residential_sqft=1000, year_built=1960),
    ], ignore_index=True)
    la = pd.Series([5, 5, 125, 125, 125])
    s = ds.stock_by_city(b, la, P)
    assert s.at[501, "type81units"] == 1
    assert s.at[2005, "type81units"] == 1
    assert s.at[2005, "typenonsqft"] == 1000


def _targets_inputs():
    cities = pd.Index([1, 2, 3], name="city_id")
    rates = pd.DataFrame(0.0, index=cities, columns=ds.CATEGORIES)
    rates["type81units"] = [0.01, 0.02, 0.042]
    stock = pd.DataFrame(0.0, index=cities, columns=ds.CATEGORIES)
    stock["type81units"] = [1000, 500, 2000]
    return stock, rates


def test_stock_targets_rate_times_stock():
    t = ds.stock_targets(*_targets_inputs())
    assert t["type81units"].tolist() == pytest.approx([10, 10, 84])


def test_rate_multiplier():
    t = ds.stock_targets(*_targets_inputs(), rate_mult=2.0)
    assert t["type81units"].sum() == pytest.approx(208)


def test_floor_cap_formula():
    # V=200, U=1000, f=0.10 -> (200-100)/0.9 = 111.1 -> 111; vacancy after stays >= 10%
    cap = float(ds.floor_cap(200, 1000, 0.10))
    assert cap == 111
    assert (200 - cap) / (1000 - cap) >= 0.10
    assert (200 - cap - 1) / (1000 - cap - 1) < 0.10
    assert float(ds.floor_cap(50, 1000, 0.10)) == 0  # already below the floor


def test_allocate_cap_no_change_when_targets_fit():
    t = pd.Series({"type81units": 10.0, "type82units": 2.0, "type83units": 0.0})
    pd.testing.assert_series_equal(ds.allocate_cap(t, 100, t), t)


def test_allocate_cap_moves_unused_share():
    t = pd.Series({"type81units": 150.0, "type82units": 50.0, "type83units": 0.0})
    avail = pd.Series({"type81units": 500.0, "type82units": 0.0, "type83units": 0.0})
    out = ds.allocate_cap(t, 100, avail)
    assert out.sum() == pytest.approx(100)
    assert out["type82units"] == 0
    assert out["type81units"] == pytest.approx(100)


def test_round_target_keeps_expected_volume():
    rng = np.random.default_rng(1)
    draws = [ds.round_target(0.6, rng.random()) for _ in range(20000)]
    assert set(draws) == {0.0, 1.0}
    assert np.mean(draws) == pytest.approx(0.6, abs=0.01)
    assert ds.round_target(3.0, 0.99) == 3.0
    assert ds.round_target(3.25, 0.1) == 4.0 and ds.round_target(3.25, 0.5) == 3.0


def test_shrunk_rates_extremes():
    dc = pytest.importorskip("demolition_calibration")
    df = pd.DataFrame({"city_id": [1, 2, 3], "large_area_id": [3, 3, 3],
                       "demolished": [0, 50, 0], "stock": [0, 10000, 10000]})
    c = dc._shrunk_rates(df, 1000)
    m = c["r_city"] / c["r_la"]
    assert m.at[1] == pytest.approx(1.0)                   # no history -> large-area rate
    long = dc._shrunk_rates(df.assign(stock=df.stock * 1000), 1000)
    assert long.at[2, "r_city"] == pytest.approx(50 / 1e7, rel=1e-3)  # long history -> own rate
    assert dc._shrunk_rates(df, np.inf)["own_weight"].eq(0).all()


@pytest.mark.skipif(not os.path.exists(RATES_CSV), reason="rates file not built")
def test_calibrated_levels_reproduce_volume():
    r = pd.read_csv(RATES_CSV)
    other = r[r.large_area_id != 5]
    g = other.assign(t=other.rate * other.stock).groupby(["large_area_id", "category"])
    got, want = g["t"].sum(), g["level_volume"].sum()
    has_stock = g["stock"].sum() > 0
    assert np.allclose(got[has_stock], want[has_stock], rtol=1e-6, atol=1e-6)
    det = r[(r.large_area_id == 5) & r.category.isin(ds.RES_CATEGORIES)]
    assert (det.rate * det.stock).sum() == pytest.approx(0.042 * det.stock.sum(), rel=1e-6)
