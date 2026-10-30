"""Unit tests for the HLCM tract calibrator helpers (lcm_utils).

Plain pandas, no orca and no input data. Run from the repo root with PYTHONPATH=.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import lcm_utils
from lcm_utils import (_hlcm_calibration_target, _hlcm_shrink_target,
                       _hlcm_tract_adjustment, _hlcm_stratified_counts,
                       _largest_remainder, _pick_stratified, assign_sqft_band)


def test_target_is_base_over_current():
    y = _hlcm_calibration_target([0.4, 0.2], [0.2, 0.4])
    assert y.tolist() == pytest.approx([2.0, 0.5])


@pytest.mark.parametrize("mode, expected", [("floor", 0.0), ("neutral", 1.0)])
def test_zero_base_share(mode, expected):
    assert _hlcm_calibration_target([0.0], [0.2], mode)[0] == expected


def test_missing_current_is_neutral_and_empty_tract_is_capped():
    y = _hlcm_calibration_target([0.0, 0.3], [0.0, 0.0])
    assert y.tolist() == [1.0, 2.0]  # 0/0 -> neutral; base with no current -> clip ceiling


def test_shrink_pulls_thin_cells_to_one_and_leaves_thick_ones():
    t = pd.Series({1: 2.0, 2: 2.0})
    out, w = _hlcm_shrink_target(t, pd.Series({1: 3.0, 2: 3000.0}), k=30.0)
    assert out[1] == pytest.approx(1.0 + 1.0 * 3 / 33)      # 1.09
    assert out[2] == pytest.approx(1.0 + 1.0 * 3000 / 3030)  # 1.99
    assert w == pytest.approx((3 / 33 + 3000 / 3030) / 2)


def test_shrink_is_symmetric_about_one():
    n = pd.Series({1: 30.0, 2: 30.0})
    out, _ = _hlcm_shrink_target(pd.Series({1: 2.0, 2: 0.5}), n, k=30.0)
    assert out[1] == pytest.approx(1.5) and out[2] == pytest.approx(0.75)


def test_shrink_disabled_and_missing_counts():
    t = pd.Series({1: 2.0})
    pd.testing.assert_series_equal(_hlcm_shrink_target(t, pd.Series(dtype=float), k=0)[0], t)
    # a tract with no base-year mass goes fully neutral, it does not keep the raw ratio
    assert _hlcm_shrink_target(t, pd.Series(dtype=float), k=30.0)[0][1] == 1.0


def test_adjustment_clips_then_applies_strength():
    w, at = _hlcm_tract_adjustment(pd.Series({1: 4.0, 2: 1.0}), [1, 2, 3],
                                   clip=(0.5, 2.0), strength=0.5)
    assert w.tolist() == pytest.approx([2.0 ** 0.5, 1.0, 1.0])  # 3 has no weight -> 1
    assert at == 0.5


def test_full_strength_passes_the_clipped_ratio_through():
    w, _ = _hlcm_tract_adjustment(pd.Series({1: 1.6}), [1], clip=(0.5, 2.0), strength=1.0)
    assert w.tolist() == pytest.approx([1.6])


CUTS = {5: np.array([1000.0, 2000.0]), 125: np.array([1500.0])}


def test_assign_sqft_band_uses_each_large_area_own_cuts():
    b = assign_sqft_band([999, 1000, 2500, 1400, 1600],
                         [5, 5, 5, 125, 125], CUTS)
    assert b.tolist() == [0, 1, 2, 0, 1]  # a value on a cut falls in the upper band


def test_assign_sqft_band_defaults_to_zero():
    b = assign_sqft_band([5000, np.nan], [99, 5], {**CUTS, 99: np.array([])})
    assert b.tolist() == [0, 0]  # unbanded large area, and a missing size


# --- stratified placement -----------------------------------------------------

def _slots(per_tract):
    """tract_ids for a choice set with `per_tract[t]` slots in each tract t."""
    return np.concatenate([np.full(k, t) for t, k in per_tract.items()])


def test_largest_remainder_sums_exactly_and_favours_big_fractions():
    out = _largest_remainder([0.6, 0.6, 0.8], 2)
    assert out.sum() == 2
    assert out.tolist() == [0, 0, 1] or out.tolist() == [1, 0, 1]
    assert _largest_remainder([1.5, 1.5, 1.0], 4).sum() == 4


def test_counts_follow_the_model_mass_when_the_weight_is_neutral():
    t = _slots({1: 100, 2: 100})
    ls = np.concatenate([np.zeros(100), np.full(100, np.log(3.0))])
    counts, d = _hlcm_stratified_counts(t, ls, pd.Series({1: 1.0, 2: 1.0}), 40)
    assert counts.sum() == 40
    # tract 2 carries 3x the mass, so it takes 3x the slots
    assert counts[2] / counts[1] == pytest.approx(3.0, rel=0.1)
    assert d['at_capacity'] == 0


def test_the_weight_tilts_the_allocation_exactly():
    t = _slots({1: 100, 2: 100})
    ls = np.zeros(200)            # model indifferent between the two tracts
    counts, _ = _hlcm_stratified_counts(t, ls, pd.Series({1: 2.0, 2: 1.0}), 30)
    assert counts.sum() == 30
    assert counts[1] == 20 and counts[2] == 10     # 2:1, landed exactly


def test_neutral_weight_follows_capacity_not_an_even_split():
    # uniform scores put mass in proportion to slots, so a 5-slot tract takes
    # 40 * 5/105 and never overflows on its own
    counts, d = _hlcm_stratified_counts(_slots({1: 5, 2: 100}), np.zeros(105),
                                        pd.Series({1: 1.0, 2: 1.0}), 40)
    assert counts[1] == 2 and counts[2] == 38
    assert d['at_capacity'] == 0


def test_overflow_water_fills_into_tracts_with_room():
    t = _slots({1: 5, 2: 100})
    # a 20x weight asks the 5-slot tract for 20 of 40; it can only take 5
    counts, d = _hlcm_stratified_counts(t, np.zeros(105),
                                        pd.Series({1: 20.0, 2: 1.0}), 40)
    assert counts[1] == 5 and counts[2] == 35
    assert counts.sum() == 40 and d['at_capacity'] == 1


def test_overflow_cascades_through_several_capped_tracts():
    t = _slots({1: 2, 2: 3, 3: 200})
    counts, d = _hlcm_stratified_counts(
        t, np.zeros(205), pd.Series({1: 50.0, 2: 50.0, 3: 1.0}), 60)
    assert counts[1] == 2 and counts[2] == 3 and counts[3] == 55
    assert d['at_capacity'] == 2


def test_allocation_never_exceeds_capacity_or_the_request():
    rng = np.random.default_rng(0)
    t = _slots({i: int(rng.integers(1, 30)) for i in range(40)})
    cap = pd.Series(t).value_counts()
    w = pd.Series(np.exp(rng.normal(0, 1.0, 40)), index=range(40))
    for n in (1, 17, 200, len(t), len(t) + 500):
        counts, _ = _hlcm_stratified_counts(t, rng.normal(size=len(t)), w, n)
        assert (counts >= 0).all()
        assert (counts <= cap.reindex(counts.index)).all()
        assert counts.sum() == min(n, len(t))


def test_picks_match_the_allocation_and_are_distinct():
    t = _slots({1: 50, 2: 50, 3: 50})
    ls = np.arange(150, dtype=float) / 50.0
    counts = pd.Series({1: 10, 2: 5, 3: 0})
    idx = _pick_stratified(np.exp(ls), ls, t, counts, 'slot_sampling', seed=7)
    assert len(idx) == 15 and len(set(idx.tolist())) == 15
    got = pd.Series(t[idx]).value_counts()
    assert got[1] == 10 and got[2] == 5 and 3 not in got


def test_top_n_within_tract_takes_the_best_slots():
    t = _slots({1: 10, 2: 10})
    scores = np.concatenate([np.arange(10.0), np.arange(10.0)])
    idx = _pick_stratified(scores, np.log(scores + 1e-9), t,
                           pd.Series({1: 2, 2: 3}), 'top_n')
    assert sorted(idx[t[idx] == 1].tolist()) == [8, 9]
    assert sorted(idx[t[idx] == 2].tolist()) == [17, 18, 19]


def test_empty_and_degenerate_inputs():
    t = _slots({1: 10})
    counts, d = _hlcm_stratified_counts(t, np.zeros(10), pd.Series({1: 1.0}), 0)
    assert counts.empty and d == {}
    # all slots scored -inf: fall back to capacity shares rather than divide by zero
    counts, _ = _hlcm_stratified_counts(_slots({1: 10, 2: 30}),
                                        np.full(40, -np.inf),
                                        pd.Series(dtype='float64'), 20)
    assert counts.sum() == 20 and counts[2] > counts[1]


CATS = {'children': ['children_no_children', 'children_has_children'],
        'senior': ['senior_with_seniors', 'senior_without_seniors'],
        'income': ['income_lowinc', 'income_midinc', 'income_highinc']}
SEG = ('tract_hh_type_ratio_children_no_children_senior_with_seniors'
       '_income_lowinc')


def test_marginal_var_picks_the_dimension_of_a_segment():
    seg = ('tract_hh_type_ratio_children_no_children_senior_with_seniors'
           '_income_lowinc')
    mv = lambda d: lcm_utils._hlcm_marginal_var(seg, d, CATS)
    assert mv('income') == 'tract_hh_type_ratio_income_lowinc'
    assert mv('children') == 'tract_hh_type_ratio_children_no_children'
    assert mv('senior') == 'tract_hh_type_ratio_senior_with_seniors'
    # not a household category of the model
    assert mv('tenure') is None


def test_marginal_weight_multiplies_into_the_joint_weight():
    tract_ids = np.array([1, 1, 2, 2])
    joint = pd.Series({1: 2.0, 2: 0.5})
    marg = pd.Series({1: 1.5, 2: 0.8})
    j, _ = _hlcm_tract_adjustment(joint, tract_ids, clip=(0.1, 10.0), strength=1.0)
    m, _ = _hlcm_tract_adjustment(marg, tract_ids, clip=(0.1, 10.0), strength=1.0)
    assert (j * m).tolist() == pytest.approx([3.0, 3.0, 0.4, 0.4])
    # a neutral marginal leaves the joint weight alone
    one, _ = _hlcm_tract_adjustment(pd.Series({1: 1.0, 2: 1.0}), tract_ids,
                                    clip=(0.1, 10.0), strength=1.0)
    assert (j * one).tolist() == pytest.approx(j.tolist())


def test_marginals_resolve_every_requested_dimension():
    cal = {'clip': [0.5, 2.0], 'shrink_k': 5,
           'marginal': {'dimensions': {'income': 4, 'children': 2.5, 'senior': 0}}}
    out = lcm_utils._hlcm_marginals(SEG, cal, CATS)
    assert [(d, v, st) for d, v, st, _, _ in out] == [
        ('income', 'tract_hh_type_ratio_income_lowinc', 4.0),
        ('children', 'tract_hh_type_ratio_children_no_children', 2.5)]
    # senior at strength 0 is dropped; the others inherit the block's clip/shrink_k
    assert all((c, k) == ((0.5, 2.0), 5.0) for _, _, _, c, k in out)


def test_marginal_dimension_can_override_clip_and_shrink():
    cal = {'clip': [0.5, 2.0], 'shrink_k': 5,
           'marginal': {'shrink_k': 3,
                        'dimensions': {'income': {'strength': 6, 'clip': [0.4, 2.5]},
                                       'senior': 1.5}}}
    got = {d: (st, c, k) for d, _, st, c, k in lcm_utils._hlcm_marginals(SEG, cal, CATS)}
    assert got['income'] == (6.0, (0.4, 2.5), 3.0)
    assert got['senior'] == (1.5, (0.5, 2.0), 3.0)   # block default, not the joint's


def test_marginals_off_by_default_and_at_zero_strength():
    for cal in ({}, {'marginal': {}}, {'marginal': {'dimensions': {}}},
                {'marginal': {'dimensions': {'income': 0, 'children': 0, 'senior': 0}}}):
        assert lcm_utils._hlcm_marginals(SEG, cal, CATS) == []


def test_marginals_ignore_a_dimension_the_model_does_not_have():
    cal = {'marginal': {'dimensions': {'tenure': 5}}}
    assert lcm_utils._hlcm_marginals(SEG, cal, CATS) == []
