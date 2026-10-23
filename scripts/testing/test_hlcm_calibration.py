"""Unit tests for the HLCM tract calibrator helpers (lcm_utils).

Plain pandas, no orca and no input data. Run from the repo root with PYTHONPATH=.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lcm_utils import (_hlcm_calibration_target, _hlcm_shrink_target,
                       _hlcm_tract_adjustment, assign_sqft_band)


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
