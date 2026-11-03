"""Integration test: land phase values reach feasibility and the developer only.

The pro forma and the developer see one phase of land and no building purchase on
phased parcels (dev_parcel_size, dev_land_cost). The parcels.parcel_size and
parcels.land_cost columns, which HLCM reads as features, keep the real values.
"""

import os
from pathlib import Path

os.environ.setdefault(
    "SEMCOG_INPUT_DIR",
    str(Path(__file__).resolve().parents[3] / "d_drive/forecast_inputs/base_year"),
)

import orca
import pandas as pd

orca.add_injectable("data_out_dir", "/tmp/land_phasing_columns_test")

import models  # noqa: E402  (registers the parcel columns)
import urbansim_parcels.utils as parcel_utils  # noqa: E402

AC = 43560
# 1: phased (40 ac, RESIDENTIAL); 2: small; 3: large but not RESIDENTIAL
PARCELS = pd.DataFrame({"parcel_sqft": [40 * AC, 2 * AC, 40 * AC], "pct_undev": [25, 0, 0]},
                       index=pd.Index([1, 2, 3], name="parcel_id"))
ZONING = pd.DataFrame({"future_use": ["RESIDENTIAL", "RESIDENTIAL", "AGRICULTURAL"],
                       "max_far": 1.0, "max_height": 30.0}, index=PARCELS.index)
LAND_COST = pd.Series([250_000.0, 100_000.0, 50_000.0], index=PARCELS.index)


def _setup():
    orca.add_table("parcels", PARCELS.copy())
    orca.add_table("zoning", ZONING.copy())
    orca.add_column("parcels", "land_cost", LAND_COST)
    orca.clear_cache()
    return orca.get_table("parcels")


def test_hlcm_columns_keep_real_values():
    p = _setup()
    assert p.land_phased.tolist() == [True, False, False]
    assert (p.parcel_size / AC).tolist() == [30, 2, 40]          # 40 ac x (1 - 25%)
    assert (p.dev_parcel_size / AC).tolist() == [15, 2, 40]      # one 15-acre phase
    assert p.land_cost.tolist() == LAND_COST.tolist()
    assert p.dev_land_cost.tolist() == [0, 100_000, 50_000]


def test_feasibility_sites_get_phase_values(monkeypatch):
    p = _setup()
    seen = {}

    def capture(sites, *a, **k):
        seen["sites"] = sites.copy()
        return sites

    monkeypatch.setattr(parcel_utils, "apply_parcel_callbacks", capture)
    monkeypatch.setattr(parcel_utils, "lookup_by_form", lambda *a, **k: pd.DataFrame())
    parcel_utils.run_feasibility(
        p, None, None, cfg="proforma.yaml", site_overrides={"parcel_size": p.dev_parcel_size,
                                       "land_cost": p.dev_land_cost})
    sites = seen["sites"]
    assert (sites.parcel_size / AC).tolist() == [15, 2, 40]
    assert sites.land_cost.tolist() == [0, 100_000, 50_000]
    # the parcels table is not changed by the feasibility run
    assert (orca.get_table("parcels").parcel_size / AC).tolist() == [30, 2, 40]
    assert orca.get_table("parcels").land_cost.tolist() == LAND_COST.tolist()


def test_developer_rules_on_phased_parcels():
    p = _setup()
    f = pd.DataFrame({"form": ["apartment", "apartment", "apartment", "single-family"]},
                     index=pd.Index([1, 1, 2, 1], name="parcel_id"))
    ok = models._land_phase_ok(f, [9, 60, 9, 31], p.dev_parcel_size)
    # phased: 9-unit apartment too small, 60 allowed, 31 single-family > 15 ac x 2.0;
    # unphased parcel 2: the 9-unit apartment stays
    assert ok.tolist() == [False, True, True, False]
