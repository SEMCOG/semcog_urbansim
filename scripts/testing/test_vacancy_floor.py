"""vacancy_floor.py: demolished space is not rebuilt, and the target never goes below the floor."""
import numpy as np
import pandas as pd
import pytest

import demolition_stock
import vacancy_floor as vf


def test_floor_rate_housing_and_nonres():
    assert vf.floor_rate([0.203, 0.04], 0.5, 0.05).tolist() == pytest.approx([0.1015, 0.05])
    assert vf.floor_rate([0.63], 0.5).tolist() == pytest.approx([0.315])


def test_highland_park_demolished_units_not_rebuilt_above_the_floor():
    # 2041 (run5079, no demolition floor): 4,486 households, 4,746 units (5.5% vacant),
    # 868 units demolished, target 20.3%. Without the floor the gap is about 15 units;
    # the vacancy is below the 10.15% floor, so the developer builds back to the floor.
    no_floor = vf.target_space(4486, 0.203, 868, 0.0)
    assert no_floor - 4746 == pytest.approx(4486 / 0.797 - 868 - 4746)
    assert no_floor - 4746 < 20
    t = vf.target_space(4486, 0.203, 868, 0.1015)
    assert t == pytest.approx(4486 / (1 - 0.1015))     # about 247 units above 4,746


def test_sterling_heights_retail_mall_not_rebuilt():
    # 2027: 7,650 jobs, 16,339 job spaces after 4,854 were demolished; target 63%
    t = vf.target_space(7650, 0.63, 4854, 0.315)
    assert t < 16339                                   # no gap: nothing is rebuilt
    assert 1 - 7650 / t == pytest.approx(0.5165, abs=1e-3)


def test_floor_binds_when_demolition_would_leave_no_vacancy():
    # 1,000 households; demolition removed so much that the target space falls below them
    t = vf.target_space(1000, 0.20, 1000, 0.10)
    assert t == pytest.approx(1000 / 0.9)              # held at the 10% floor


def test_floor_above_target_does_not_raise_the_target():
    # tight city: target 4%, floor 5% (min); the target, not the floor, applies
    assert vf.target_space(1000, 0.04, 0, 0.05) == pytest.approx(1000 / 0.96)


def test_demolished_space_counts_demolition_steps_only():
    dropped = pd.DataFrame({
        "step": ["scored_demolition_events", "scheduled_demolition_events", "developer_redevelopment",
                 "scored_demolition_events"],
        "city_id": [1, 1, 1, 2], "residential_units": [10, 5, 7, 0],
        "building_type_id": [81, 21, 21, 23], "non_residential_sqft": [0.0, 5000.0, 9000.0, 4000.0]})
    assert vf.demolished_units(dropped).to_dict() == {1: 15, 2: 0}
    js = vf.demolished_job_spaces(dropped, pd.Series({21: 500.0, 23: 400.0}),
                                  {21: "retail", 23: "office"})
    assert js.to_dict() == {(1, "retail"): 10.0, (2, "office"): 10.0}


def test_demolition_cap_stops_at_the_floor():
    # 4,746 units, 260 vacant (5.5%), floor 10.15%: no unit may be removed
    assert demolition_stock.floor_cap(260, 4746, 0.1015) == 0
    # 5,614 units, 1,141 vacant (20.3%): removing x vacant units stops at the floor
    x = demolition_stock.floor_cap(1141, 5614, 0.1015)
    assert (1141 - x) / (5614 - x) == pytest.approx(0.1015, abs=1e-3)
