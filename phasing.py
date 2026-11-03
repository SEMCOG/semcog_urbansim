"""Phased delivery of large residential developer projects, and phased land.

A new developer building records its full size in `planned_units`, but only the
first phase goes into `residential_units`; the rest is delivered over the next
years at a yearly rate per building type (configs/res_developer.yaml: phasing).

Land phasing (res_developer.yaml: land_phasing): a large parcel with a planned
residential use is offered to the developer one phase of land at a time. A project
uses only the land its units need at the density of recent subdivisions, and keeps
the earlier phases' buildings; the next phase opens after the project is delivered.
Plain pandas functions, no orca.
"""
import numpy as np
import pandas as pd

SQFT_PER_ACRE = 43560


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


def land_phasing_cfg(res_cfg):
    """Land phasing settings from res_developer.yaml (disabled when the block is missing)."""
    p = (res_cfg or {}).get("land_phasing", {}) or {}
    return {"enabled": bool(p.get("enabled", False)),
            "min_sqft": float(p.get("min_acres", 15)) * SQFT_PER_ACRE,
            "phase_sqft": float(p.get("phase_acres", 15)) * SQFT_PER_ACRE,
            "gap_years": int(p.get("gap_years", 2)),
            "future_uses": list(p.get("future_uses", ["RESIDENTIAL"])),
            "units_per_acre": {int(k): float(v) for k, v in (p.get("units_per_acre", {}) or {}).items()},
            "min_units": {int(k): int(v) for k, v in (p.get("min_units", {}) or {}).items()}}


def land_phased(parcel_sqft, future_use, cfg):
    """Parcels developed in land phases: large, with a planned use in cfg['future_uses']."""
    if not cfg["enabled"]:
        return pd.Series(False, index=parcel_sqft.index)
    return (parcel_sqft >= cfg["min_sqft"]) & future_use.reindex(parcel_sqft.index).isin(cfg["future_uses"])


def phase_size(available_sqft, phased, cfg):
    """Land offered to the developer: one phase at most on phased parcels."""
    return available_sqft.where(~phased, available_sqft.clip(upper=cfg["phase_sqft"]))


def land_used_sqft(planned_units, building_type_id, cfg):
    """Land a phased project uses: its full size (planned_units) at the type's density.
    Types without a density return NaN (the caller keeps the footprint rule)."""
    upa = pd.Series(building_type_id).map(cfg["units_per_acre"]).values
    return np.asarray(planned_units, float) / upa * SQFT_PER_ACRE


def phase_proposal_ok(net_units, building_type_id, phased, available_sqft, cfg):
    """Proposals allowed by the land phase rules. On a phased parcel a proposal may not
    be larger than the phase at its type's density (types without a density: no limit)
    and needs at least the type's min_units. Proposals on other parcels: all allowed."""
    net = np.asarray(net_units, float)
    btype = pd.Series(np.asarray(building_type_id))
    upa = btype.map(cfg["units_per_acre"]).values
    too_big = net > np.asarray(available_sqft, float) / SQFT_PER_ACRE * upa  # NaN: False
    too_small = net < btype.map(cfg["min_units"]).fillna(0).values
    return ~(np.asarray(phased, bool) & (too_big | too_small))
