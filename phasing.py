"""Phased delivery of large residential developer projects.

A new developer building records its full size in `planned_units`, but only the
first phase goes into `residential_units`; the rest is delivered over the next
years at a yearly rate per building type (configs/res_developer.yaml: phasing).
Plain pandas functions, no orca.
"""
import numpy as np
import pandas as pd


def phasing_cfg(res_cfg):
    """Phasing settings from res_developer.yaml (disabled when the block is missing)."""
    p = (res_cfg or {}).get("phasing", {}) or {}
    rates = {int(k): int(v) for k, v in (p.get("units_per_year", {}) or {}).items()}
    return {"enabled": bool(p.get("enabled", False)), "rates": rates,
            "default_rate": int(p.get("default_units_per_year", max(rates.values()) if rates else 200))}


def yearly_rate(building_type_id, cfg):
    """Units a project of this building type delivers per year."""
    return pd.Series(building_type_id).map(cfg["rates"]).fillna(cfg["default_rate"]).astype(int).values


def next_delivery(planned, delivered, building_type_id, cfg):
    """Units delivered this year: min(remaining, yearly rate).
    A new project (delivered = 0) gets its first phase."""
    remaining = np.clip(np.asarray(planned, int) - np.asarray(delivered, int), 0, None)
    return np.minimum(remaining, yearly_rate(building_type_id, cfg))


def limit_selection(build_idx, net_units, target, tolerance):
    """Selection cap used with phasing: the proposals to build, in selection order.

    build_idx: proposals picked by proposal_select; net_units: units of all the
    city's proposals (index = proposal). Keep proposals while the running total
    stays within target x (1 + tolerance); the rest of the demand carries to the
    next year. If the first proposal alone is too big, build the city's smallest
    proposal instead when it is smaller, else the first one (phasing spreads it).
    """
    build_idx = np.asarray(build_idx)
    total = np.cumsum(net_units.loc[build_idx].values.astype(float))
    limit = target * (1.0 + tolerance)
    if len(total) == 0 or total[-1] <= limit:
        return build_idx
    keep = int(np.searchsorted(total, limit, side="right"))
    if keep > 0:
        return build_idx[:keep]
    positive = net_units[net_units > 0]
    if len(positive) and positive.min() < total[0]:
        return np.array([positive.idxmin()])
    return build_idx[:1]
