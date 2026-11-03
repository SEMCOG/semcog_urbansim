"""Nonres developer calls (run_developer without `profit`): the long-form feasibility table
holds every form and Developer.pick ignores `forms`, so run_developer restricts the pool of
each call to its forms. Parcels built by a call leave the table for the later calls.

Developer.pick is replaced by a stub that records the pool it gets, so no model state is needed.
"""

import os
from pathlib import Path

os.environ.setdefault(
    "SEMCOG_INPUT_DIR",
    str(Path(__file__).resolve().parents[3] / "d_drive/forecast_inputs/base_year"),
)

import orca
import pandas as pd
import pytest

orca.add_injectable("data_out_dir", "/tmp/nonres_form_filter_test")

import models
from developer import develop

PARCELS = pd.Index([1, 2, 3, 3], name="parcel_id")
# parcel 3 has two proposals: warehouse and retail
FEAS = pd.DataFrame({
    "form": ["retail", "office", "warehouse", "retail"],
    "non_residential_sqft": [20_000.0, 90_000.0, 300_000.0, 15_000.0],
    "building_sqft": [20_000.0, 90_000.0, 300_000.0, 15_000.0],
    "residential_sqft": 0.0,
    "max_profit": [1e5, 9e5, 5e6, 2e5],
    "max_profit_far": 1.0,
}, index=PARCELS)


@pytest.fixture
def pool(monkeypatch):
    seen = []
    build = {"parcel": None}

    def fake_pick(self, profit_to_prob_func=None, custom_selection_func=None):
        seen.append(self.feasibility.copy())
        if build["parcel"] is None:
            return None
        row = self.feasibility.loc[[build["parcel"]]].iloc[[0]]
        return row.assign(parcel_id=row.index, job_spaces=10, current_units=0)

    monkeypatch.setattr(develop.Developer, "pick", fake_pick)
    monkeypatch.setattr(models.parcel_utils, "add_buildings", lambda *a, **k: None)
    monkeypatch.setattr(models, "_label_developer_removals", lambda: None)
    monkeypatch.setitem(orca.orca._TABLES, "parcels", orca.orca.DataFrameWrapper(
        "parcels", pd.DataFrame({"parcel_sqft": 10_000.0}, index=PARCELS.unique())))  # not land-phased
    orca.add_injectable("year", 2031)
    orca.add_table("feasibility_city_1", FEAS)
    return seen, build


def _run(forms):
    ones = pd.Series(1.0, index=PARCELS.unique())
    return models.run_developer(
        100, "city_1", forms, pd.DataFrame({"parcel_id": []}), "job_spaces",
        ones * 100_000, ones * 1000, ones * 0, "nonres_developer.yaml")


def test_retail_call_gets_retail_proposals_only(pool):
    seen, _ = pool
    _run(["retail"])
    assert set(seen[0]["form"]) == {"retail"}
    assert sorted(seen[0].index.tolist()) == [1, 3]


def test_group_call_keeps_all_its_forms(pool):
    seen, _ = pool
    _run(["manufacturing", "wholesale", "warehouse"])
    assert seen[0]["form"].tolist() == ["warehouse"]


def test_nothing_built_puts_full_table_back(pool):
    _run(["retail"])
    assert len(orca.get_table("feasibility_city_1").to_frame()) == len(FEAS)


def test_built_parcel_leaves_table_for_later_calls(pool):
    seen, build = pool
    build["parcel"] = 3
    _run(["retail"])
    left = orca.get_table("feasibility_city_1").to_frame()
    assert 3 not in left.index                     # its warehouse proposal is gone too
    assert sorted(left["form"]) == ["office", "retail"]
    build["parcel"] = None
    _run(["office"])                               # a later call sees the other forms only
    assert seen[1]["form"].tolist() == ["office"]
