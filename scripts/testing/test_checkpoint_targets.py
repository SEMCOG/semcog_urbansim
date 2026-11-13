"""Base-year targets saved in a run h5 and read back on a resume (checkpoint_targets).

Plain pandas, no orca and no input data. Run from the repo root with PYTHONPATH=.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import checkpoint_targets as ct


def _targets():
    key = pd.MultiIndex.from_tuples([(3, 81, 0), (3, 81, 1), (5, 83, 4)],
                                    names=["large_area_id", "building_type_id", "band"])
    return {
        "tract_hh_type_base_ratios": pd.DataFrame(
            {"tract_hh_type_ratio_a": [0.2, 0.5], "tract_hh_type_total": [10, 40]},
            index=pd.Index([2610001, 2610002], name="tract_id")),
        "new_unit_fill_shares": pd.DataFrame({"tract_hh_type_ratio_a": [0.1, 0.3, 0.0]}, index=key),
        "new_unit_fill_ratios": pd.DataFrame({"tract_hh_type_ratio_a": [0.9, 1.2, 1.0]}, index=key),
        "sqft_band_cuts": {3: np.array([900.0, 1400.0, 2100.0]), 5: np.array([700.0, 1100.0])},
        "job_btype_baseyear_prob_matrix": pd.DataFrame({1: [0.7, 0.3], 2: [0.0, 1.0]},
                                                       index=pd.Index([21, 23], name="building_type_id")),
    }


def test_round_trip(tmp_path):
    path = str(tmp_path / "run.h5")
    t = _targets()
    ct.save(path, t)
    with pd.HDFStore(path, "r") as st:
        back = ct.read(st)
        assert all(k.startswith("/%s/" % ct.GROUP) for k in st.keys())
    for k in ("tract_hh_type_base_ratios", "new_unit_fill_shares", "new_unit_fill_ratios",
              "job_btype_baseyear_prob_matrix"):
        pd.testing.assert_frame_equal(back[k], t[k])
    assert sorted(back["sqft_band_cuts"]) == [3, 5]
    for la, cuts in t["sqft_band_cuts"].items():
        np.testing.assert_array_equal(back["sqft_band_cuts"][la], cuts)


def test_store_without_targets_reads_none(tmp_path):
    path = str(tmp_path / "old.h5")
    with pd.HDFStore(path, "w") as st:
        st["2034/households"] = pd.DataFrame({"building_id": [1]})
        assert ct.read(st) is None


def test_incomplete_targets_raise(tmp_path):
    path = str(tmp_path / "part.h5")
    with pd.HDFStore(path, "w") as st:
        st["%s/tract_hh_type_base_ratios" % ct.GROUP] = _targets()["tract_hh_type_base_ratios"]
        with pytest.raises(ValueError, match="missing"):
            ct.read(st)


def test_group_is_not_a_year_or_base_key():
    # update_store_from_checkpoint finds the year from purely numeric groups, and the
    # resume copy in simulation_2055 skips keys holding '/base/'
    k = "/%s/sqft_band_cuts" % ct.GROUP
    assert not k.split("/")[1].isdigit() and "/base/" not in k
