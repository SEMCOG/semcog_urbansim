"""Unit tests for the feasible residential capacity (res_capacity.py). Fast, no orca."""
import pandas as pd

import res_capacity

PS = pd.Series({1: 50_000.0, 2: 50_000.0, 3: 2_000_000.0, 4: 50_000.0})
AVE = pd.Series({1: 2_000.0, 2: 500.0, 3: 2_000.0, 4: 2_000.0})
CUR = pd.Series({1: 0, 2: 4, 3: 0, 4: 0})
MAX_PS = 1_100_000


def _feas(rows):
    return pd.DataFrame(rows, columns=["parcel_id", "form", "residential_sqft", "max_profit",
                                       "max_profit_far"]).set_index("parcel_id").assign(
        building_sqft=lambda d: d.residential_sqft)


def _cap(f, allowed=None):
    return res_capacity.feasible_capacity(f, PS, AVE, CUR, 1_000, MAX_PS, allowed)


def test_one_value_per_parcel_over_far_and_form_rows():
    # parcel 1: three FAR variants and two forms -> its largest, 20 units (not 10+20+15)
    f = _feas([(1, "single-family", 20_000, 5, 0.4), (1, "single-family", 40_000, 3, 0.8),
               (1, "condo", 30_000, 1, 0.6)])
    assert _cap(f) == 20


def test_net_of_current_units_and_min_unit_size():
    # parcel 2: 10,000 sqft / max(500, 1,000) = 10 units, minus 4 existing = 6
    assert _cap(_feas([(2, "condo", 10_000, 5, 0.2)])) == 6


def test_rows_pick_cannot_build_are_not_counted():
    f = _feas([(1, "single-family", 20_000, 0, 0.4),      # not profitable
               (4, "single-family", 20_000, 5, 0.0),      # max_profit_far 0
               (3, "single-family", 20_000, 5, 0.4),      # parcel_size >= max_parcel_size
               (2, "condo", 3_000, 5, 0.2)])              # net units 3 - 4 < 0
    assert _cap(f) == 0


def test_max_parcel_size_is_strict():
    ps = pd.Series({1: float(MAX_PS)})
    f = _feas([(1, "single-family", 20_000, 5, 0.4)])
    assert res_capacity.feasible_capacity(f, ps, AVE, CUR, 1_000, MAX_PS) == 0
    assert res_capacity.feasible_capacity(f, ps - 1, AVE, CUR, 1_000, MAX_PS) == 10


def test_allowed_mask_drops_rows():
    f = _feas([(1, "apartment", 20_000, 5, 0.4), (4, "single-family", 10_000, 5, 0.2)])
    assert _cap(f, allowed=[False, True]) == 5


def test_empty_table():
    assert _cap(_feas([])) == 0


def test_loss_rows_counted_when_not_profitable_only():
    # parcel 1: loss-making 10 units (and a profitable 5); parcel 4: loss-making, max_profit_far 0
    f = _feas([(1, "single-family", 20_000, -5, 0.4), (1, "single-family", 10_000, 2, 0.2),
               (4, "single-family", 20_000, -5, 0.0)])
    assert _cap(f) == 5
    assert res_capacity.feasible_capacity(f, PS, AVE, CUR, 1_000, MAX_PS,
                                          profitable_only=False) == 10


def test_loss_rows_use_the_least_loss_per_sqft_variant():
    # parcel 1: the least-loss pass keeps the 10-unit variant (-2 / 20,000 sqft), not the
    # larger 20-unit one (-40 / 40,000 sqft); the profitable 5 units are smaller
    f = _feas([(1, "single-family", 20_000, -2, 0.4), (1, "single-family", 40_000, -40, 0.8),
               (1, "single-family", 10_000, 2, 0.2)])
    assert res_capacity.feasible_capacity(f, PS, AVE, CUR, 1_000, MAX_PS,
                                          profitable_only=False) == 10


def test_gap_floor_keeps_the_price_order():
    # the threshold rule lifted 50 to 160 but left 100 at 100; the gap share keeps 50 < 100
    p = pd.Series([50.0, 100.0, 150.0, 200.0])
    out = res_capacity.gap_floor_price(p, pd.Series(160.0, index=p.index), 0.5)
    assert out.tolist() == [105.0, 130.0, 155.0, 200.0]
    assert out.is_monotonic_increasing


def test_gap_floor_share_limits():
    p = pd.Series([50.0, 200.0])
    rep = pd.Series(160.0, index=p.index)
    assert res_capacity.gap_floor_price(p, rep, 0.0).tolist() == [50.0, 200.0]
    assert res_capacity.gap_floor_price(p, rep, 1.0).tolist() == [160.0, 200.0]


def test_gap_floor_missing_price_gets_the_replacement():
    p = pd.Series([float("nan"), 80.0])
    rep = pd.Series([160.0, 160.0])
    assert res_capacity.gap_floor_price(p, rep, 0.25).tolist() == [160.0, 100.0]


def test_net_gap_cap_surplus_leaves_only_the_mismatch_share():
    # city A short 1,000, city B surplus 3,000: net -2,000, so only the share of A's 1,000
    assert res_capacity.net_gap_cap([1000, -3000], [1000], 0.5) == 500.0
    assert res_capacity.net_gap_cap([1000, -3000], [1000], 0.0) == 0.0


def test_net_gap_cap_shortage_adds_the_net_gap():
    assert res_capacity.net_gap_cap([1000, 500, -300], [1000, 500], 0.5) == 1200 + 750
    # share 1 never binds below the positive targets
    assert res_capacity.net_gap_cap([1000, -300], [1000], 1.0) >= 1000
