"""Shared helpers for in-process replay / paired tests of the forecast pipeline.

A paired test runs the steps up to a point once, saves the orca state, and then runs the
same steps for several cases (on/off, factor A/B) from that saved state. Two things can go
wrong without any error message, and the helpers here check both:

1. A table is still a cached orca table function (buildings, households, ... until a step
   calls orca.add_table). orca.get_table() hides this: it returns a DataFrameWrapper either
   way. An in-place edit then changes only the cached copy, orca.clear_cache() drops it, and
   the next access loads the table from the store again. hold_data_tables() turns the data
   tables into plain tables; snapshot()/restore() work on plain tables only.
2. An intervention does not reach the step that reads it. scale_column() reads the column
   back after a cache clear and raises if the values are not the expected ones;
   assert_changed() raises if a derived table (e.g. nodes_walk) did not change.

Typical use:

    import replay_utils as ru
    ru.hold_data_tables()
    ...pre-steps...
    saved = ru.snapshot()
    for case in cases:
        ru.restore(saved, seed=SEED, injectables={"remi_price_growth": True})
        before = orca.get_table("nodes_walk").to_frame()
        ru.scale_by_group("buildings", ["sqft_price_res"], "large_area_id", {3: 1 / 1.38, ...})
        orca.run(["neighborhood_vars"], iter_vars=[year])
        ru.assert_changed(before, orca.get_table("nodes_walk").to_frame(), ["residential"])

Import with scripts/testing on sys.path (sys.path.insert or PYTHONPATH). Self-test:
pytest scripts/testing/test_replay_utils.py
"""
import numpy as np
import orca
import pandas as pd

DATA_TABLES = ("buildings", "households", "jobs", "persons", "parcels", "group_quarters")


class ReplayCheckError(AssertionError):
    """An intervention or a saved state is not what the test expects."""


def _require_plain(table):
    if orca.table_type(table) != "dataframe":
        raise ReplayCheckError(
            "%r is an orca table function: in-place edits are lost on clear_cache(). "
            "Call hold_data_tables() first." % table)


def hold_data_tables(names=DATA_TABLES):
    """Register each named table as a plain table (DataFrameWrapper). Returns the names changed."""
    held = []
    for t in names:
        if orca.is_table(t) and orca.table_type(t) != "dataframe":
            orca.add_table(t, orca.get_table(t).local)
            held.append(t)
    for t in names:
        if orca.is_table(t):
            _require_plain(t)
    print("replay_utils: held as plain tables:", held)
    return held


# Module-level caches in the pipeline that orca.clear_cache() does not reach. A case must not
# reuse them: e.g. the REPM feature frame is cached per YEAR (utils.get_cached_buildings_df), so
# without a reset a second case in the same year predicts from the first case's features.
MODULE_CACHES = {"utils": {"_xgb_features_cache": {"year": None, "df": None, "all_features": None}}}


def _reset_module_caches():
    import sys
    for mod, caches in MODULE_CACHES.items():
        m = sys.modules.get(mod)
        for name, empty in caches.items():
            if m is not None and hasattr(m, name):
                getattr(m, name).clear()
                getattr(m, name).update(empty)


def _copy_plain(v):
    if isinstance(v, (pd.Series, pd.DataFrame, np.ndarray)):
        return v.copy()
    if isinstance(v, (dict, list, set)):
        return type(v)(v) if not isinstance(v, dict) else dict(v)
    return v


def snapshot():
    """Save the orca state: a copy of every plain table, the registered function tables
    (re-run after restore()), and every injectable (plain values deep-copied).

    Steps also keep state outside the data tables, e.g. log tables they extend
    (debug_res_developer) or injectables they set on the first call (parcel_impr_shares).
    restore() puts all of it back, so a case cannot see an earlier case. Injectables are
    kept by reference (a step that replaces one is undone); plain containers are copied
    one level deep (a step that edits one in place is undone too). Objects such as the
    HDF store and the loaded models are not copied.
    """
    reg = orca.orca
    tables = {}
    for t in orca.list_tables():
        raw = reg._TABLES[t]
        tables[t] = (orca.get_table(t).local.copy() if orca.table_type(t) == "dataframe" else raw)
    inj = {k: _copy_plain(v) for k, v in reg._INJECTABLES.items()}
    return {"tables": tables, "injectables": inj}


def restore(saved, seed=None, injectables=None):
    """Put the saved tables and injectables back (and drop those added since), clear the
    cache, then set the given injectables and the numpy seed."""
    reg = orca.orca
    for t in [t for t in list(reg._TABLES) if t not in saved["tables"]]:
        del reg._TABLES[t]
    for t, v in saved["tables"].items():
        if isinstance(v, pd.DataFrame):
            orca.add_table(t, v.copy())
        else:
            reg._TABLES[t] = v
    for k in [k for k in list(reg._INJECTABLES) if k not in saved["injectables"]]:
        del reg._INJECTABLES[k]
    for k, v in saved["injectables"].items():
        reg._INJECTABLES[k] = _copy_plain(v)
    orca.clear_cache()
    _reset_module_caches()
    for k, v in (injectables or {}).items():
        orca.add_injectable(k, v)
    if seed is not None:
        np.random.seed(seed)
    for t, v in saved["tables"].items():
        if isinstance(v, pd.DataFrame):
            cur = orca.get_table(t).local
            if cur.shape != v.shape or not cur.index.equals(v.index):
                raise ReplayCheckError("restore: table %r differs from the snapshot" % t)


def scale_column(table, col, factor, round_values=False):
    """Multiply table[col] by factor (a Series on the table index or a subset of it).

    After the update the cache is cleared and the column is read back: raises if the values
    are not the expected ones (lost edit) or if a factor other than 1 changed nothing.
    Returns (mean before, mean after) over the rows in factor.
    """
    _require_plain(table)
    factor = pd.Series(factor, dtype="float64")
    if factor.isna().any():
        raise ReplayCheckError("scale_column %s.%s: %d rows without a factor"
                               % (table, col, int(factor.isna().sum())))
    w = orca.get_table(table)
    old = w.local.loc[factor.index, col].copy()
    new = old * factor
    if round_values:
        new = new.round()
    w.update_col_from_series(col, new, cast=True)
    orca.clear_cache()
    back = orca.get_table(table).local.loc[factor.index, col]
    expected = new.astype(back.dtype)
    if not np.allclose(back.to_numpy("float64"), expected.to_numpy("float64"),
                       rtol=1e-6, atol=0.5 if round_values else 1e-9, equal_nan=True):
        raise ReplayCheckError("scale_column %s.%s: the column does not hold the new values "
                               "after clear_cache()" % (table, col))
    moved = (factor != 1.0) & (old != 0) & old.notna()
    if moved.any() and np.allclose(back[moved].to_numpy("float64"), old[moved].to_numpy("float64")):
        raise ReplayCheckError("scale_column %s.%s: no value changed" % (table, col))
    print("replay_utils: %s.%s scaled on %d rows, mean %.4g -> %.4g"
          % (table, col, len(factor), old.mean(), back.mean()))
    return old.mean(), back.mean()


def scale_by_group(table, cols, group_col, factor_by_group, round_values=False):
    """Multiply each column in cols by factor_by_group[group] (e.g. large_area_id -> 1/ratio).
    Rows whose group has no factor raise, unless the group value is NaN (they keep their value)."""
    g = orca.get_table(table)[group_col]
    g = g[g.notna()]
    f = g.map(factor_by_group)
    if f.isna().any():
        raise ReplayCheckError("scale_by_group %s: groups without a factor: %s"
                               % (table, sorted(g[f.isna()].unique().tolist())[:10]))
    return {c: scale_column(table, c, f, round_values) for c in cols}


def assert_changed(before, after, cols, min_rel_change=1e-4):
    """Raise unless every column in cols changed by a mean relative amount >= min_rel_change.
    Returns {col: mean(after) / mean(before)}."""
    out = {}
    for c in cols:
        a, b = before[c].astype("float64"), after[c].astype("float64").reindex(before.index)
        rel = float(np.nanmean(np.abs(b - a)) / max(np.nanmean(np.abs(a)), 1e-12))
        if not rel >= min_rel_change:
            raise ReplayCheckError("assert_changed: %r did not change (mean relative change %.2g)"
                                   % (c, rel))
        out[c] = float(np.nanmean(b) / np.nanmean(a))
    print("replay_utils: changed as required:", {k: round(v, 4) for k, v in out.items()})
    return out


def assert_ratio(observed, expected, rtol=0.02, what="ratio"):
    """Raise unless observed is within rtol of expected (e.g. a mean price ratio vs 1/factor)."""
    if not abs(observed / expected - 1) <= rtol:
        raise ReplayCheckError("%s %.4f, expected %.4f (rtol %.2g)" % (what, observed, expected, rtol))
