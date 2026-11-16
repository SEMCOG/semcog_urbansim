"""lcm_utils._elcm_event_boost: the lift on scheduled-event non-residential slots.

Run from the repository root:
    PYTHONPATH=. python scripts/testing/test_elcm_event_boost.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def alts(rows):
    """rows: (building_id, event_id, non_residential_sqft, year_built)."""
    df = pd.DataFrame(rows, columns=["bid", "event_id", "non_residential_sqft", "year_built"])
    return df.set_index("bid")


def slots(a, per):
    """One row per vacant job space, as the non-home-based path builds them."""
    return a.loc[np.repeat(a.index.to_numpy(), per)]


def t_factor_one_is_a_bitwise_noop():
    from lcm_utils import _elcm_event_boost
    a = alts([(1, 900001, 50000, 2030), (2, 0, 50000, 2030)])
    arr, n = _elcm_event_boost({"factor": 1.0}, a, slots(a, [2, 2]), 2035)
    assert n == 0 and np.array_equal(arr, np.ones(4)), (arr, n)
    print("  PASS factor 1.0 lifts nothing")


def t_only_event_nonres_is_lifted():
    from lcm_utils import _elcm_event_boost
    a = alts([(1, 900001, 50000, 2030),   # event, non-res   -> lifted
              (2, 0, 50000, 2030),        # developer built  -> not
              (3, 900002, 0, 2030)])      # event, no nonres -> not
    s = slots(a, [1, 1, 1])
    arr, n = _elcm_event_boost({"factor": 1.2, "max_age": 0}, a, s, 2035)
    assert n == 1 and np.allclose(arr, [1.2, 1.0, 1.0]), arr
    print("  PASS only event non-residential slots lifted")


def t_max_age_expires_the_lift():
    from lcm_utils import _elcm_event_boost
    a = alts([(1, 900001, 50000, 2030)])
    s = slots(a, [1])
    for yr, want in ((2035, 1.2), (2040, 1.2), (2041, 1.0)):
        arr, _ = _elcm_event_boost({"factor": 1.2, "max_age": 10}, a, s, yr)
        assert arr[0] == want, (yr, arr[0])
    arr, _ = _elcm_event_boost({"factor": 1.2, "max_age": 0}, a, s, 2055)
    assert arr[0] == 1.2, arr
    print("  PASS max_age 10 expires after 10 years; 0 keeps it for the whole run")


def t_one_weight_per_slot_not_per_building():
    from lcm_utils import _elcm_event_boost
    a = alts([(1, 900001, 50000, 2030), (2, 0, 50000, 2030)])
    s = slots(a, [3, 2])                      # duplicate index, 5 slots
    arr, n = _elcm_event_boost({"factor": 1.2, "max_age": 0}, a, s, 2035)
    assert len(arr) == 5 and n == 3 and np.allclose(arr, [1.2, 1.2, 1.2, 1.0, 1.0]), arr
    print("  PASS duplicate index gives one weight per slot")


def t_missing_column_degrades_to_noop():
    from lcm_utils import _elcm_event_boost
    a = alts([(1, 900001, 50000, 2030)]).drop(columns=["event_id"])
    arr, n = _elcm_event_boost({"factor": 1.2}, a, slots(a, [2]), 2035)
    assert n == 0 and np.array_equal(arr, np.ones(2)), (arr, n)
    print("  PASS a frame without event_id is a no-op, not an error")


def t_odds_shift_matches_the_factor():
    """The lift multiplies the odds of an event slot by exactly the factor."""
    from lcm_utils import _elcm_event_boost
    a = alts([(1, 900001, 50000, 2030), (2, 0, 50000, 2030)])
    s = slots(a, [10, 90])                    # events are 10% of slots
    arr, _ = _elcm_event_boost({"factor": 1.2, "max_age": 0}, a, s, 2035)
    w = np.ones(len(s)) * arr                 # flat model scores
    share = w[:10].sum() / w.sum()
    want = 1.2 * 0.10 / (1.2 * 0.10 + 0.90)
    assert abs(share - want) < 1e-12, (share, want)
    print("  PASS event share of draw weight 10.0%% -> %.2f%% at factor 1.2" % (100 * share))


def main():
    tests = [t_factor_one_is_a_bitwise_noop, t_only_event_nonres_is_lifted,
             t_max_age_expires_the_lift, t_one_weight_per_slot_not_per_building,
             t_missing_column_degrades_to_noop, t_odds_shift_matches_the_factor]
    fails = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as e:
            fails += 1
            print("  FAIL %s: %s" % (fn.__name__, e))
    print("%d of %d passed" % (len(tests) - fails, len(tests)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
