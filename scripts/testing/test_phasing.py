"""Unit tests for phased delivery of developer projects (phasing.py). Fast, no orca."""
import numpy as np
import pandas as pd
import pytest

import phasing

CFG = phasing.phasing_cfg({"phasing": {"enabled": True, "units_per_year": {81: 75, 82: 75, 83: 200}}})


def test_cfg_disabled_without_block():
    assert phasing.phasing_cfg({})["enabled"] is False


def test_first_phase_caps_large_projects_only():
    first = phasing.next_delivery([929, 30, 367, 250], 0, [82, 81, 81, 83], CFG)
    assert first.tolist() == [75, 30, 75, 200]


def test_delivery_schedule_completes_project():
    planned, delivered, years = 929, 75, 1
    while delivered < planned:
        delivered += int(phasing.next_delivery([planned], [delivered], [82], CFG)[0])
        years += 1
    assert delivered == planned
    assert years == 13  # 75 per year: 929 units take 13 years


def test_complete_buildings_get_nothing():
    assert phasing.next_delivery([10, 0], [10, 0], [81, 31], CFG).tolist() == [0, 0]


def test_unknown_type_uses_default_rate():
    assert phasing.next_delivery([500], 0, [84], CFG).tolist() == [200]


def _select(selected, others=()):
    """limit_selection on proposals 0..n-1 (selected, in order) plus unselected ones."""
    units = pd.Series(list(selected) + list(others), dtype=float)
    return phasing.limit_selection(np.arange(len(selected)), units, TARGET, 0.25).tolist()


TARGET = 100


def test_limit_selection_keeps_fitting_prefix():
    # target 100, tolerance 0.25: 40 + 40 + 50 = 130 > 125 -> keep the first two (80)
    assert _select([40, 40, 50]) == [0, 1]


def test_limit_selection_no_change_when_within_tolerance():
    assert _select([60, 60]) == [0, 1]


def test_limit_selection_first_too_big_uses_smallest():
    # the first proposal alone (929) busts 125; the city has a 12-unit proposal (index 2)
    assert _select([929, 500], others=[12]) == [2]


def test_limit_selection_first_too_big_and_nothing_smaller():
    assert _select([929], others=[929, 0]) == [0]


def _old_limit(build_idx, net_units, target, tol):
    """The rule as run5017 ran it (models.py + phasing.limit_selection, 2026-10-18)."""
    total = np.cumsum(net_units.loc[build_idx].values.astype(float))
    limit = target * (1.0 + tol)
    positive = net_units[net_units > 0]
    if len(total) == 0 or total[-1] <= limit:
        keep = len(total)
    else:
        keep = int(np.searchsorted(total, limit, side="right"))
        if keep == 0:
            keep = -1 if (positive.min() if len(positive) else np.inf) < total[0] else 1
    if keep < 0:
        return np.array([positive.idxmin()])
    return np.asarray(build_idx)[:keep]


def test_limit_selection_same_as_run5017_rule():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        n = int(rng.integers(1, 30))
        units = pd.Series(rng.choice([0, 1, 2, 4, 12, 30, 75, 200, 929], n).astype(float),
                          index=rng.permutation(10_000)[:n])
        build_idx = units.index.values[rng.permutation(n)[: int(rng.integers(0, n + 1))]]
        target, tol = float(rng.integers(1, 400)), float(rng.choice([0.0, 0.25, 0.5]))
        new = phasing.limit_selection(build_idx, units, target, tol)
        assert new.tolist() == _old_limit(build_idx, units, target, tol).tolist()
