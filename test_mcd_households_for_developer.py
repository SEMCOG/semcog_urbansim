"""Regression test: residential developer demand counts unplaced households.

households_transition adds tens of thousands of households each year that stay
unplaced (no MCD) until HLCM. Counting only placed households at developer time
inflated MCD vacancy and drove the raw demand target negative almost everywhere.
"""

import os
from pathlib import Path

os.environ.setdefault(
    "SEMCOG_INPUT_DIR",
    str(Path(__file__).resolve().parents[1] / "d_drive/forecast_inputs/base_year"),
)

import orca
import pandas as pd

orca.add_injectable("data_out_dir", "/tmp/mcd_households_for_developer_test")

import models

MCD_TO_LA = pd.Series({10: 3, 20: 3, 30: 5})
SNAPSHOT = pd.Series({10: 60, 20: 40, 30: 50})


def test_la_change_spread_by_snapshot_share():
    # LA 3 grew from 100 to 110 households (10 unplaced); LA 5 lost 5
    hh_la = pd.Series([3] * 110 + [5] * 45)
    out = models.mcd_households_for_developer(hh_la, MCD_TO_LA, SNAPSHOT)

    assert out[10] == 66 and out[20] == 44
    assert out[30] == 45


def test_large_area_totals_match_current_households():
    hh_la = pd.Series([3] * 97 + [5] * 58)
    out = models.mcd_households_for_developer(hh_la, MCD_TO_LA, SNAPSHOT)

    assert out.groupby(MCD_TO_LA).sum().to_dict() == {3: 97, 5: 58}


def test_no_change_returns_snapshot():
    hh_la = pd.Series([3] * 100 + [5] * 50)
    out = models.mcd_households_for_developer(hh_la, MCD_TO_LA, SNAPSHOT)

    assert out.equals(SNAPSHOT.astype(float))
