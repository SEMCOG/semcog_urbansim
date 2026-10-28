"""Type-mix least-loss pass (run_developer(profit="unprofitable")): no loss-making proposal
larger than the remaining target of the form (+ the overshoot tolerance) reaches the selection.

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

orca.add_injectable("data_out_dir", "/tmp/loss_pool_size_test")

import models
from developer import develop

PARCELS = pd.Index([1, 2], name="parcel_id")
# parcel 1: 500 units, the smallest loss per sqft (ranked first by the least-loss sort)
# parcel 2: 10 units, a larger loss per sqft
FEAS = pd.DataFrame({
    "form": ["apartment", "apartment"],
    "residential_sqft": [500_000.0, 10_000.0],
    "building_sqft": [600_000.0, 12_000.0],
    "max_profit": [-6_000_000.0, -600_000.0],
    "max_profit_far": [2.0, 0.5],
    "non_residential_sqft": [0.0, 0.0],
}, index=PARCELS)


@pytest.fixture
def pool(monkeypatch):
    seen = []

    def fake_pick(self, profit_to_prob_func=None, custom_selection_func=None):
        seen.append(self.feasibility.copy())
        return None

    monkeypatch.setattr(develop.Developer, "pick", fake_pick)
    orca.add_injectable("year", 2026)
    orca.add_table("feasibility_city_1", FEAS)
    return seen


def _run(target):
    ones = pd.Series(1.0, index=PARCELS)
    return models.run_developer(
        target, "city_1", ["apartment"], None, "residential_units",
        ones * 100_000, ones * 1000, ones * 0, "res_developer.yaml",
        custom_selection_func=lambda *a: None, profit="unprofitable", exclude_parcels=set())


def test_fitting_loss_proposal_replaces_larger_better_ranked_one(pool):
    # target 10, tolerance 0.25: the 500-unit proposal is out, although it ranks first
    _run(10)
    assert len(pool) == 1
    assert pool[0].index.tolist() == [2]


def test_large_target_keeps_both(pool):
    _run(1000)
    assert sorted(pool[0].index.tolist()) == [1, 2]


def test_empty_loss_pool_returns_zero_without_selection(pool):
    # target 5: both proposals (500 and 10 units) exceed 5 x 1.25
    assert _run(5) == (0, [])
    assert pool == []  # pick() was not called
    # the full table is back in orca for the next pass
    assert len(orca.get_table("feasibility_city_1").to_frame()) == 2


def _run_feas(feas, target, profit="unprofitable", min_units=None, funnel=None,
              size_limit=False, forms=("apartment",)):
    orca.add_table("feasibility_city_1", feas)
    idx = feas.index.unique()
    ones = pd.Series(1.0, index=idx)
    return models.run_developer(
        target, "city_1", list(forms), None, "residential_units",
        ones * 100_000, ones * 1000, ones * 0, "res_developer.yaml",
        custom_selection_func=lambda *a: None, profit=profit, exclude_parcels=set(),
        min_units_by_form=min_units, funnel=funnel, size_limit=size_limit)


def _rows(parcels, units, profit_per_sqft, form="apartment"):
    sqft = [u * 1000.0 for u in units]
    return pd.DataFrame({
        "form": form, "residential_sqft": sqft, "building_sqft": sqft,
        "max_profit": [p * q for p, q in zip(profit_per_sqft, sqft)],
        "max_profit_far": 0.5, "non_residential_sqft": 0.0,
    }, index=pd.Index(parcels, name="parcel_id"))


def test_min_apartment_units_drops_small_buildings(pool):
    feas = _rows([1, 2, 3], [1, 4, 6], [-10, -10, -10])
    _run_feas(feas, 20, min_units={"apartment": 5})
    assert pool[0].index.tolist() == [3]


def test_min_apartment_units_also_in_profitable_pass(pool):
    feas = _rows([1, 2], [2, 8], [10, 10])
    _run_feas(feas, 20, profit="profitable", min_units={"apartment": 5})
    assert pool[0].index.tolist() == [2]


def test_one_variant_per_parcel_least_loss_per_sqft(pool):
    # parcel 1 has three variants; only the one with the least loss per sqft stays
    feas = _rows([1, 1, 1, 2], [6, 8, 10, 7], [-30, -10, -20, -15])
    _run_feas(feas, 100)
    p = pool[0]
    assert not p.index.duplicated().any()
    assert p.loc[1, "residential_sqft"] == 8000.0


def test_pool_counted_in_net_units(pool):
    # target 10: the size limit (12.5) admits 12-unit buildings; the pool keeps rows while
    # the net units before them are below 3 x 10 = 30 (0, 12, 24 -> parcels 1-3)
    feas = _rows([1, 2, 3, 4, 5], [12, 12, 12, 12, 12], [-1, -2, -3, -4, -5])
    _run_feas(feas, 10)
    assert pool[0].index.tolist() == [1, 2, 3]


def test_funnel_records_each_step(pool):
    fun = []
    feas = _rows([1, 1, 2], [6, 8, 3], [-30, -10, -20])
    _run_feas(feas, 10, min_units={"apartment": 5}, funnel=fun)
    f = fun[0]
    assert f["target"] == 10
    assert f["parcels_min_units"] == 1 and f["cap_min_units"] == 8
    assert f["rows_one_per_parcel"] == 1 and f["cap_pool"] == 8


# D1: size limit also in the profitable passes of the type mix (size_limit=True)

def test_profitable_size_limit_keeps_eligible_proposal(pool):
    # target 4, tolerance 0.25: limit 5 units; the 225-unit condo is out, the 5-unit one stays
    feas = _rows([1, 2], [225, 5], [20, 10], form="condo")
    _run_feas(feas, 4, profit="profitable", size_limit=True, forms=["condo"])
    assert pool[0].index.tolist() == [2]


def test_profitable_oversized_only_pool_builds_nothing(pool):
    feas = _rows([1, 2], [225, 6], [20, 10], form="condo")
    assert _run_feas(feas, 4, profit="profitable", size_limit=True, forms=["condo"]) == (0, [])
    assert pool == []  # pick() was not called; the next pass (least loss) can try
    assert len(orca.get_table("feasibility_city_1").to_frame()) == 2


def test_profitable_without_size_limit_unchanged(pool):
    # the path without type mix keeps every profitable proposal
    feas = _rows([1, 2], [225, 5], [20, 10], form="condo")
    _run_feas(feas, 4, profit="profitable", forms=["condo"])
    assert sorted(pool[0].index.tolist()) == [1, 2]


def test_remainder_limit_uses_remaining_city_units(pool):
    # remainder call: all forms, target = remaining city units (40 -> limit 50)
    feas = pd.concat([_rows([1], [60], [10], form="single-family"),
                      _rows([2], [50], [10], form="condo"),
                      _rows([3], [8], [10], form="apartment")])
    fun = []
    _run_feas(feas, 40, profit="profitable", size_limit=True, funnel=fun,
              forms=["single-family", "condo", "apartment"])
    assert sorted(pool[0].index.tolist()) == [2, 3]
    assert fun[0]["cap_size_limit"] == 58


# min_loss_condo_units: residential_developer adds {"condo": n} to min_units_by_form in the
# least-loss pass only

def test_condo_minimum_in_loss_pass(pool):
    feas = _rows([1, 2, 3], [1, 4, 6], [-10, -10, -10], form="condo")
    _run_feas(feas, 20, min_units={"condo": 5}, forms=["condo"])
    assert pool[0].index.tolist() == [3]


def test_no_condo_minimum_keeps_small_profitable_condo(pool):
    feas = _rows([1, 2], [1, 6], [10, 10], form="condo")
    _run_feas(feas, 20, profit="profitable", forms=["condo"])
    assert sorted(pool[0].index.tolist()) == [1, 2]
