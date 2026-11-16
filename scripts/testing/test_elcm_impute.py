"""Unit tests for the ELCM new-building job-stock imputation (lcm_utils, 2026-11-15).

Run from the repo root: PYTHONPATH=. python -m pytest scripts/testing/test_elcm_impute.py -q
"""
import numpy as np
import pandas as pd
import pytest

from lcm_utils import ELCM_STOCK_COLS, impute_new_building_stock, new_building_stock_profile

SEC = ["bldg_jobs_sector_%d" % s for s in range(1, 19)]


def make_alts():
    """Six old occupied buildings of type 21 and 23, one old empty building, three new buildings."""
    rows = []
    for i, (bt, js, nh, hb, s3) in enumerate([(21, 100, 50, 5, 30), (21, 200, 120, 10, 60),
                                               (21, 50, 20, 2, 10), (21, 80, 40, 4, 20),
                                               (21, 60, 30, 3, 15), (23, 400, 300, 0, 100),
                                               (21, 100, 0, 0, 0)]):
        r = dict(building_id=i + 1, building_type_id=bt, year_built=1990, job_spaces=js,
                 jobs_non_home_based=nh, jobs_home_based=hb, b_total_jobs=nh + hb)
        r.update({c: 0.0 for c in SEC})
        r["bldg_jobs_sector_3"] = s3
        r["bldg_jobs_sector_5"] = nh + hb - s3
        rows.append(r)
    for bid, yb, nh in [(101, 2026, 0), (102, 2030, 0), (103, 2026, 90)]:
        r = dict(building_id=bid, building_type_id=21, year_built=yb, job_spaces=100,
                 jobs_non_home_based=nh, jobs_home_based=0, b_total_jobs=nh)
        r.update({c: 0.0 for c in SEC})
        r["bldg_jobs_sector_5"] = nh
        rows.append(r)
    a = pd.DataFrame(rows).set_index("building_id")
    a["vacant_job_spaces"] = (a.job_spaces - a.b_total_jobs).clip(lower=0)
    a["other_feature"] = np.arange(len(a), dtype=float)
    return a


def slots(a):
    """Feature rows of sampled slots: two slots per building, as the slot sampler repeats ids."""
    return a.loc[np.repeat(a.index.to_numpy(), 2), ELCM_STOCK_COLS + ["other_feature"]].astype(float)


def test_profile_uses_old_occupied_buildings_only():
    prof, allp = new_building_stock_profile(make_alts(), base_year=2025)
    assert list(prof.index) == [21]  # type 23 has one building < min_buildings
    assert prof.loc[21, "n"] == 5    # the empty old building and the new ones are excluded
    assert prof.loc[21, "occupancy"] == pytest.approx(0.5)  # median of 0.5, 0.6, 0.4, 0.5, 0.5
    assert prof.loc[21, ["bldg_jobs_sector_%d" % s for s in range(1, 19)]].sum() == pytest.approx(1)
    assert allp.n == 6


def test_imputed_columns_are_consistent():
    a = make_alts()
    X, info = impute_new_building_stock(slots(a), a, 2026, 2025, 10, new_building_stock_profile(a, 2025))
    new = X.loc[[101]]
    assert np.allclose(new.jobs_non_home_based, 50.0)  # occupancy 0.5 x 100 job spaces
    assert np.allclose(new[SEC].sum(axis=1), new.b_total_jobs)
    assert np.allclose(new.jobs_non_home_based + new.jobs_home_based, new.b_total_jobs)
    assert np.allclose(new.vacant_job_spaces, 100 - new.b_total_jobs)
    assert info["buildings"] == 2 and info["slots"] == 4  # 101 and 102; 103 has more than the imputed stock


def test_old_buildings_and_other_features_unchanged():
    a = make_alts()
    X0 = slots(a)
    X, _ = impute_new_building_stock(X0, a, 2030, 2025, 10, new_building_stock_profile(a, 2025))
    old = X.index <= 7
    pd.testing.assert_frame_equal(X[old], X0[old])
    pd.testing.assert_series_equal(X.other_feature, X0.other_feature)
    assert X0.loc[101, "b_total_jobs"].eq(0).all()  # the input frame is not changed in place


def test_fade_and_never_below_real():
    a = make_alts()
    prof = new_building_stock_profile(a, 2025)
    nh = {}
    for year in (2026, 2031, 2036, 2040):
        X, _ = impute_new_building_stock(slots(a), a, year, 2025, 10, prof)
        nh[year] = X.loc[101, "jobs_non_home_based"].iloc[0]
        assert X.loc[103, "jobs_non_home_based"].iloc[0] == 90  # real stock above imputed: no change
    assert nh[2026] == pytest.approx(50.0)                     # age 0: full imputed stock
    assert nh[2031] == pytest.approx(25.0)                     # age 5 of 10: half
    assert nh[2036] == 0 and nh[2040] == 0                     # faded to the real stock
    X, _ = impute_new_building_stock(slots(a), a, 2030, 2025, 10, prof)
    assert X.loc[102, "jobs_non_home_based"].iloc[0] == pytest.approx(50.0)  # built 2030: age 0


def test_partly_occupied_new_building_blends_up():
    a = make_alts()
    a.loc[101, ["jobs_non_home_based", "b_total_jobs", "bldg_jobs_sector_5"]] = 10
    a.loc[101, "vacant_job_spaces"] = 90
    X, _ = impute_new_building_stock(slots(a), a, 2031, 2025, 10, new_building_stock_profile(a, 2025))
    r = X.loc[101].iloc[0]
    assert r.jobs_non_home_based == pytest.approx(10 + 0.5 * (50 - 10))
    assert r[SEC].sum() == pytest.approx(r.b_total_jobs)


def test_type_without_profile_uses_all_types():
    a = make_alts()
    a.loc[101, "building_type_id"] = 99
    X, _ = impute_new_building_stock(slots(a), a, 2026, 2025, 10, new_building_stock_profile(a, 2025))
    assert X.loc[101, "jobs_non_home_based"].iloc[0] > 0


def test_no_new_buildings_returns_input():
    a = make_alts().loc[1:7]
    X0 = slots(a)
    X, info = impute_new_building_stock(X0, a, 2026, 2025, 10, new_building_stock_profile(a, 2025))
    assert X is X0 and info["slots"] == 0
