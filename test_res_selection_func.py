"""Regression test: residential site selection must score proposals by parcel_id.

With keep_suboptimal=True, Developer.pick() resets the proposal index to row
numbers and moves parcel_id into a column. Scoring by df.index then matched no
parcels, so every proposal got a uniform probability and the demolition-rebuild
boost never applied.
"""

import os
from pathlib import Path

os.environ.setdefault(
    "SEMCOG_INPUT_DIR",
    str(Path(__file__).resolve().parents[1] / "d_drive/forecast_inputs/base_year"),
)

import numpy as np
import orca
import pandas as pd
from developer import proposal_select

orca.add_injectable("data_out_dir", "/tmp/res_selection_func_test")

import models

MODEL = {
    "features": ["x"],
    "coef": [2.0],
    "intercept": 0.0,
    "scaler_mean": [0.0],
    "scaler_std": [1.0],
}
FEATURES = pd.DataFrame(
    {"x": [1.0, 0.0, 0.0], "land_use_type_id": [11, 11, 11]},
    index=pd.Index([1000001, 1000002, 1000003], name="parcel_id"),
)


def _probs(df, demo_boost=None, monkeypatch=None):
    """Run the selection closure and capture the probabilities it hands to the sampler."""
    captured = {}

    def fake_choice(d, p, target_units):
        captured["p"] = p
        return d.index.values[:1]

    monkeypatch.setattr(proposal_select, "weighted_random_choice_multiparcel", fake_choice)
    score = models.make_res_selection_func(
        {"fallback": MODEL, 11: MODEL}, FEATURES, demo_boost=demo_boost
    )
    score(None, df, None, 10)
    return captured["p"]


def _suboptimal_frame():
    # Two proposals per parcel, shaped as pick() passes them (RangeIndex + parcel_id column)
    return pd.DataFrame({
        "parcel_id": [1000001, 1000001, 1000002, 1000002],
        "net_units": [10, 20, 10, 20],
    })


def test_scores_by_parcel_id_column(monkeypatch):
    df = _suboptimal_frame()
    p = _probs(df, monkeypatch=monkeypatch)

    assert p.index.equals(df.index)
    assert np.isclose(p.sum(), 1.0)
    # parcel 1000001 has the higher utility, so its proposals must outrank 1000002's
    assert p.iloc[0] > p.iloc[2]
    assert np.isclose(p.iloc[0], p.iloc[1])


def test_demo_boost_applies_by_parcel_id(monkeypatch):
    df = _suboptimal_frame()
    base = _probs(df, monkeypatch=monkeypatch)
    boosted = _probs(df, demo_boost=pd.Series({1000002: 20.0}), monkeypatch=monkeypatch)

    assert boosted.iloc[2:].sum() > base.iloc[2:].sum()


def test_parcel_id_index_still_supported(monkeypatch):
    # keep_suboptimal=False: one proposal per parcel, indexed by parcel_id
    df = pd.DataFrame(
        {"net_units": [10, 10]},
        index=pd.Index([1000001, 1000002], name="parcel_id"),
    )
    p = _probs(df, monkeypatch=monkeypatch)

    assert p.index.equals(df.index)
    assert p.iloc[0] > p.iloc[1]
