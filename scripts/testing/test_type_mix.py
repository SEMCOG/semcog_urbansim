"""Unit tests for type_mix.py (residential developer type-mix quotas)."""
import pandas as pd
import pytest

import type_mix


def test_split_units_sums_and_largest_remainder():
    out = type_mix.split_units(10, {"a": 1, "b": 1, "c": 1})
    assert sum(out.values()) == 10
    assert sorted(out.values()) == [3, 3, 4]
    assert type_mix.split_units(0, {"a": 1}) == {"a": 0}
    assert type_mix.split_units(5, {"a": 0, "b": 0}) == {"a": 0, "b": 0}


def test_developer_quota_fills_gap_after_events():
    shares = {"single-family": 0.5, "condo": 0.1, "apartment": 0.4}
    # events already deliver 300 apartments; total = 1000 -> apartment target 400, gap 100
    q = type_mix.developer_quota(700, {"apartment": 300}, shares)
    assert sum(q.values()) == 700
    assert q == {"single-family": 500, "condo": 100, "apartment": 100}


def test_developer_quota_events_above_target_give_zero():
    shares = {"single-family": 0.5, "condo": 0.1, "apartment": 0.4}
    q = type_mix.developer_quota(100, {"apartment": 1000}, shares)
    assert q["apartment"] == 0 and sum(q.values()) == 100


def test_developer_quota_no_gap_falls_back_to_shares():
    shares = {"single-family": 0.5, "condo": 0.5}
    q = type_mix.developer_quota(10, {"single-family": 100, "condo": 100}, shares)
    assert q == {"single-family": 5, "condo": 5}


def test_allocate_to_cities_follows_targets():
    out = type_mix.allocate_to_cities(9, {101: 2, 102: 1, 103: 0})
    assert out == {101: 6, 102: 3, 103: 0}


def test_load_shares_and_cfg(tmp_path):
    p = tmp_path / "mix.csv"
    pd.DataFrame({"scenario": ["base", "base", "other"], "large_area_id": [3, 5, 3], "year": [2026] * 3,
                  "share_sf": [0.6, 0.1, 1.0], "share_condo": [0.1, 0.1, 0.0], "share_apt": [0.3, 0.8, 0.0]}
                 ).to_csv(p, index=False)
    s = type_mix.load_shares(p, 2026, "base")
    assert list(s.index) == [3, 5]
    assert s.loc[5, "apartment"] == pytest.approx(0.8)
    assert type_mix.type_mix_cfg({}) is None
    assert type_mix.type_mix_cfg({"type_mix": {"file": "x.csv", "scenario": "base"}}) == {"file": "x.csv", "scenario": "base"}


def test_allocate_min_units_reallocates_small_shares():
    # 12 units over targets 10/1/1: plain split gives 10/1/1; the 1-unit cities are
    # dropped one at a time and their units go to the city that can take a building
    out = type_mix.allocate_to_cities(12, {101: 10, 102: 1, 103: 1}, min_units=5)
    assert out == {101: 12, 102: 0, 103: 0}


def test_allocate_min_units_keeps_cities_at_or_above_minimum():
    out = type_mix.allocate_to_cities(20, {101: 1, 102: 1, 103: 2}, min_units=5)
    assert sum(out.values()) == 20
    assert all(u == 0 or u >= 5 for u in out.values())
    assert out == type_mix.allocate_to_cities(20, {101: 1, 102: 1, 103: 2}, min_units=5)  # deterministic


def test_allocate_min_units_quota_below_minimum_stays_unmet():
    assert type_mix.allocate_to_cities(4, {101: 3, 102: 1}, min_units=5) == {101: 0, 102: 0}


def test_allocate_min_units_zero_target_city_gets_nothing():
    out = type_mix.allocate_to_cities(10, {101: 0, 102: 5}, min_units=5)
    assert out == {101: 0, 102: 10}
