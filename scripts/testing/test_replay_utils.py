"""Self-test for replay_utils.py. Fast: a tiny orca setup, no model data."""
import os
import sys

import numpy as np
import orca
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_utils as ru  # noqa: E402


@pytest.fixture(autouse=True)
def tables():
    orca.clear_all()
    base = pd.DataFrame({"price": [100.0, 200.0, 300.0, 0.0], "la": [3, 5, 5, np.nan]},
                        index=[10, 11, 12, 13])

    @orca.table(cache=True)
    def buildings():
        return base.copy()

    @orca.table(cache=True)
    def summary(buildings):  # a function table derived from buildings
        return pd.DataFrame({"mean_price": [buildings.price.mean()]})

    yield
    orca.clear_all()


def test_edit_of_function_table_is_lost_without_hold():
    # the pitfall itself: get_table() hides the function table, the edit is dropped
    assert isinstance(orca.get_table("buildings"), orca.DataFrameWrapper)
    assert orca.table_type("buildings") == "function"
    w = orca.get_table("buildings")
    w.update_col_from_series("price", w.price * 0.5)
    orca.clear_cache()
    assert orca.get_table("buildings").price.iloc[0] == 100.0


def test_scale_refuses_function_table():
    with pytest.raises(ru.ReplayCheckError, match="table function"):
        ru.scale_column("buildings", "price", pd.Series(0.5, index=[10]))


def test_hold_then_scale_survives_clear_cache():
    assert ru.hold_data_tables(["buildings"]) == ["buildings"]
    ru.scale_by_group("buildings", ["price"], "la", {3: 0.5, 5: 0.25})
    orca.clear_cache()
    assert orca.get_table("buildings").price.tolist() == [50.0, 50.0, 75.0, 0.0]
    # the derived function table follows the edit
    assert orca.get_table("summary").mean_price.iloc[0] == pytest.approx(175 / 4)


def test_scale_by_group_raises_on_missing_group():
    ru.hold_data_tables(["buildings"])
    with pytest.raises(ru.ReplayCheckError, match="without a factor"):
        ru.scale_by_group("buildings", ["price"], "la", {3: 0.5})


def test_snapshot_restore_round_trip():
    ru.hold_data_tables(["buildings"])
    saved = ru.snapshot()
    assert isinstance(saved["tables"]["buildings"], pd.DataFrame)
    assert orca.table_type("summary") == "function"
    ru.scale_by_group("buildings", ["price"], "la", {3: 0.5, 5: 0.5})
    ru.restore(saved, seed=1, injectables={"flag": True})
    assert orca.get_table("buildings").price.tolist() == [100.0, 200.0, 300.0, 0.0]
    assert orca.get_injectable("flag") is True


def test_assert_changed_and_ratio():
    a = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 1.0]})
    b = pd.DataFrame({"x": [0.5, 1.0], "y": [1.0, 1.0]})
    assert ru.assert_changed(a, b, ["x"])["x"] == pytest.approx(0.5)
    with pytest.raises(ru.ReplayCheckError, match="did not change"):
        ru.assert_changed(a, b, ["y"])
    ru.assert_ratio(0.5, 0.5)
    with pytest.raises(ru.ReplayCheckError):
        ru.assert_ratio(0.6, 0.5)


def test_restore_drops_later_tables_and_injectables():
    # a step that extends a log table and sets an injectable on its first call
    ru.hold_data_tables(["buildings"])
    orca.add_injectable("counter", {"n": 0})
    saved = ru.snapshot()
    orca.add_table("log", pd.DataFrame({"case": [1]}))
    orca.add_injectable("first_call_state", 5)
    orca.get_injectable("counter")["n"] = 9          # in-place edit of a container
    orca.add_table("summary", pd.DataFrame({"mean_price": [0.0]}))  # function table replaced
    ru.restore(saved)
    assert not orca.is_table("log")
    assert not orca.is_injectable("first_call_state")
    assert orca.get_injectable("counter") == {"n": 0}
    assert orca.table_type("summary") == "function"
    assert orca.get_table("summary").mean_price.iloc[0] == pytest.approx(150.0)
