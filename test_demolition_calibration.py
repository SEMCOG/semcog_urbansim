"""Parity tests: demolition calibration features must equal the simulation features.

The calibration trains on historical buildings, and the simulation scores the
forecast buildings. A coefficient is only valid if both calculate each feature
the same way.
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
import pytest
import yaml

orca.add_injectable("data_out_dir", "/tmp/demolition_calibration_test")

import demolition_calibration as dc
import models  # also registers the building columns


def test_feature_formulas_match_simulation():
    orca.add_injectable("year", 2025)
    b = orca.get_table("buildings").to_frame([
        "parcel_id", "year_built", "residential_units", "sqft_per_unit",
        "non_residential_sqft", *dc.RES_FEATURES,
    ])
    p = orca.get_table("parcels").to_frame(["bldgimprval", "landvalue"])
    b = b.join(p, on="parcel_id")

    inputs = pd.DataFrame({
        "year_built_imputed":   b["year_built"],
        "res_sqft":             b["residential_units"] * b["sqft_per_unit"],
        "non_residential_sqft": b["non_residential_sqft"],
        "building_value":       b["bldgimprval"].fillna(1),
        "land_value":           b["landvalue"].fillna(0),
    }, index=b.index)
    calib = dc.compute_features(inputs, 2025)

    for f in dc.RES_FEATURES:
        assert np.allclose(calib[f], b[f], rtol=1e-6, atol=1e-6), f



def test_yaml_features_exist_in_simulation():
    with open(dc.CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    columns = set(orca.get_table("buildings").columns)
    for key in dc.MODELS:
        assert set(cfg[key]["coef"]) <= columns, key


def test_odds_score_rejects_missing_feature():
    section = {"coef": {"x": 1.0}, "feature_mean": {"x": 0.0}, "feature_std": {"x": 1.0}}
    with pytest.raises(KeyError):
        models._odds_score(pd.DataFrame({"y": [1.0]}), section)


def test_volume_table_has_sampled_categories():
    rates = orca.get_table("demolition_rates").to_frame()
    assert list(rates.columns) == ["typenonsqft", "type81units", "type82units", "type83units"]
    assert rates.index.name == "city_id"


# ── Stock-based targets and prerequisites (PLAN_demolition_stock_rates.md) ──

import assumptions
import demolition_stock
import variables


def test_protected_types_blocked_at_data_load():
    b = orca.get_table("buildings").to_frame(["building_type_id", "sp_filter"])
    protected = b["building_type_id"].isin(assumptions.PROTECTED_BTYPES)
    assert protected.any()
    assert (b.loc[protected, "sp_filter"] == -1).all()


def test_big_nonres_parcels_protected_after_2050():
    orca.add_injectable("year", 2051)
    try:
        allowed = variables.parcel_is_allowed_2055()
        big = orca.get_table("parcels").non_residential_sqft >= 50000
        assert big.any()
        assert not allowed.reindex(big[big].index).fillna(False).any()
    finally:
        orca.add_injectable("year", 2025)


def test_detroit_vacant_houses_skip_value_rule():
    orca.add_injectable("year", 2026)
    with open(dc.CONFIG_PATH) as f:
        sp = demolition_stock.stock_rate_cfg(yaml.safe_load(f))
    b = orca.get_table("buildings").to_frame([
        "parcel_id", "building_type_id", "residential_units", "non_residential_sqft",
        "b_total_households", "building_age", "sp_filter", "event_id"])
    la = b["parcel_id"].map(orca.get_table("parcels").large_area_id)
    pool = demolition_stock.vacant_pool_mask(b.assign(households=b["b_total_households"]), la, sp)
    assert 20000 < b.loc[pool, "residential_units"].sum() < 40000
    with_rule = variables.parcel_is_allowed_2055(
        impr_to_land_ratio=5, purpose="demolition")
    without = variables.parcel_is_allowed_2055(
        impr_to_land_ratio=np.inf, purpose="demolition")
    pcl = b.loc[pool, "parcel_id"]
    # most pool houses are blocked by the value rule and open without it
    assert with_rule.reindex(pcl).fillna(False).mean() < 0.2
    assert without.reindex(pcl).fillna(False).mean() > 0.8


def test_developer_removals_labeled():
    keep = orca.get_table("dropped_buildings").local.copy() if orca.is_table("dropped_buildings") else None
    orca.add_injectable("year", 2030)
    orca.add_table("dropped_buildings", pd.DataFrame(
        {"year_demo": [2030, 2030, 2029], "step": [np.nan, "scored_demolition_events", np.nan]},
        index=pd.Index([1, 2, 3], name="building_id")))
    try:
        models._label_developer_removals()
        steps = orca.get_table("dropped_buildings").local["step"]
        assert steps.tolist()[:2] == ["developer_redevelopment", "scored_demolition_events"]
        assert pd.isna(steps.iat[2])  # an earlier year is not relabeled
    finally:
        if keep is not None:
            orca.add_table("dropped_buildings", keep)
        orca.add_injectable("year", 2025)
