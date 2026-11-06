"""Checks on the HLCM MCD vacancy weight (lcm_utils._hlcm_mcd_vacancy_weights).

Builds a small fake buildings/target pair in orca rather than loading a run, so
it stays a seconds-long check. Run from the repository root:
    PYTHONPATH=. python scripts/testing/test_hlcm_mcd_vacancy.py
"""
import os
import sys

import numpy as np
import orca
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import lcm_utils
from lcm_utils import _hlcm_mcd_vacancy_weights, _hlcm_tract_adjustment


def setup(rows, targets, year=2030):
    """rows: (semmcd, city_id, la, units, vacant). targets: city_id -> rate."""
    b = pd.DataFrame(rows, columns=['semmcd', 'city_id', 'large_area_id',
                                    'residential_units', 'vacant_residential_units'])
    orca.add_table('buildings', b)
    orca.add_table('target_res_vacancies_mcd',
                   pd.DataFrame({str(year): pd.Series(targets)}))
    orca.add_injectable('year', year)
    lcm_utils._mcd_vac_cache.clear()
    return _hlcm_mcd_vacancy_weights(year)


def t_neutral_when_vacancy_equals_target():
    w = setup([(1, 11, 3, 1000, 50), (2, 12, 3, 1000, 50)], {11: 0.05, 12: 0.05})
    assert np.allclose(w.values, 1.0), w
    print('  PASS vacancy == target everywhere -> all weights 1')


def t_above_target_outranks_below():
    # mcd 1 is 2x its target, mcd 2 is at target
    w = setup([(1, 11, 3, 1000, 100), (2, 12, 3, 1000, 50)], {11: 0.05, 12: 0.05})
    assert w[1] > 1 > w[2], w
    # LA centering is unit-weighted, so the pair's weights bracket 1
    assert np.isclose(w[1] / w[2], 2.0), w[1] / w[2]
    print('  PASS above target outranks at-target, ratio 2.0 preserved')


def t_la_centering_does_not_mix_large_areas():
    # LA 3 sits far above target everywhere, LA 5 at target; the within-LA
    # ordering of LA 5 must not move because of LA 3
    rows = [(1, 11, 3, 1000, 400), (2, 12, 3, 1000, 200),
            (3, 13, 5, 1000, 60), (4, 14, 5, 1000, 40)]
    tg = {11: 0.05, 12: 0.05, 13: 0.05, 14: 0.05}
    w = setup(rows, tg)
    assert np.isclose(w[1] / w[2], 2.0) and np.isclose(w[3] / w[4], 1.5), w
    # each large area's unit-weighted mean weight is 1
    assert np.isclose((w[1] + w[2]) / 2, 1.0) and np.isclose((w[3] + w[4]) / 2, 1.0), w
    print('  PASS large areas centered independently, within-LA ratios intact')


def t_clip_keeps_within_la_order_after_centering():
    # uncentered these are 8x and 4x target and would both hit a 2.0 clip,
    # flattening them; centered they stay apart
    rows = [(1, 11, 3, 1000, 400), (2, 12, 3, 1000, 200)]
    w = setup(rows, {11: 0.05, 12: 0.05})
    a, _ = _hlcm_tract_adjustment(w, np.array([1, 2]), clip=(0.5, 2.0), strength=1.0)
    assert a[0] > a[1], a
    raw, _ = _hlcm_tract_adjustment(pd.Series({1: 8.0, 2: 4.0}), np.array([1, 2]),
                                    clip=(0.5, 2.0), strength=1.0)
    assert raw[0] == raw[1] == 2.0, raw
    print('  PASS centering saves the within-LA split that clipping would flatten')


def t_strength_scales_on_the_log_scale():
    w = setup([(1, 11, 3, 1000, 100), (2, 12, 3, 1000, 50)], {11: 0.05, 12: 0.05})
    ids = np.array([1, 2])
    full, _ = _hlcm_tract_adjustment(w, ids, clip=(0.5, 2.0), strength=1.0)
    half, _ = _hlcm_tract_adjustment(w, ids, clip=(0.5, 2.0), strength=0.5)
    off, _ = _hlcm_tract_adjustment(w, ids, clip=(0.5, 2.0), strength=0.0)
    assert np.allclose(half, np.sqrt(full)), (half, full)
    assert np.allclose(off, 1.0), off
    print('  PASS strength 0.5 is the square root of the full tilt, 0 switches it off')


def t_target_weighted_from_cities():
    # one MCD, two cities with different targets and sizes: 0.02 on 3000 units
    # and 0.10 on 1000 gives a unit-weighted target of 0.04
    rows = [(1, 11, 3, 3000, 120), (1, 12, 3, 1000, 40), (2, 13, 3, 1000, 40)]
    w = setup(rows, {11: 0.02, 12: 0.10, 13: 0.04})
    # mcd 1: vacancy 160/4000 = 0.04 against a 0.04 target -> ratio 1, same as mcd 2
    assert np.allclose(w.values, 1.0), w
    print('  PASS city targets weighted into the MCD by units')


def t_missing_target_year_skips():
    setup([(1, 11, 3, 1000, 50)], {11: 0.05}, year=2030)
    lcm_utils._mcd_vac_cache.clear()
    assert _hlcm_mcd_vacancy_weights(2099) is None
    print('  PASS a year the target table lacks returns None')


def main():
    fails = 0
    for fn in [t_neutral_when_vacancy_equals_target, t_above_target_outranks_below,
               t_la_centering_does_not_mix_large_areas,
               t_clip_keeps_within_la_order_after_centering,
               t_strength_scales_on_the_log_scale, t_target_weighted_from_cities,
               t_missing_target_year_skips]:
        try:
            fn()
        except AssertionError as e:
            fails += 1
            print('  FAIL %s: %s' % (fn.__name__, e))
    print('%d of 7 passed' % (7 - fails))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
