"""Stock definitions for stock-based demolition targets.

Shared by demolition_calibration.py (rates from history and the base year) and
scored_demolition_events (targets each simulated year), so both measure the
stock the same way. Plain pandas functions, no orca.

Categories follow configs/demolition_rates.csv: residential units of building
types 81/82/83 and non-residential sqft.
"""
import numpy as np
import pandas as pd

RES_CATEGORIES = {"type81units": 81, "type82units": 82, "type83units": 83}
NONRES_CATEGORY = "typenonsqft"
CATEGORIES = list(RES_CATEGORIES) + [NONRES_CATEGORY]


def stock_rate_cfg(cfg):
    """Defaults for the stock_rate keys of configs/demolition_model.yaml."""
    s = cfg.get("stock_rate", {}) or {}
    d = cfg.get("detroit", {}) or {}
    ex = d.get("value_rule_exempt", {}) or {}
    return {
        "res_max_year_built":    s.get("res_stock_max_year_built", 1949),
        "nonres_max_year_built": s.get("nonres_stock_max_year_built", 1969),
        "detroit_la_ids":        list(d.get("large_area_ids", [5])),
        "rate_per_vacant_unit":  d.get("rate_per_vacant_unit", 0.042),
        "floor_min":             d.get("vacancy_floor_min", 0.05),
        "floor_share":           d.get("vacancy_floor_share_of_base", 0.5),
        "exempt_types":          list(ex.get("building_types", list(RES_CATEGORIES.values()))),
        "exempt_max_units":      ex.get("max_units", 4),
        "min_age":               cfg.get("min_age_eligible", 10),
    }


def vacant_pool_mask(b, la, p):
    """Detroit buildings that can be demolished without moving anyone.

    b needs building_type_id, residential_units, non_residential_sqft,
    households (count), building_age, sp_filter, event_id. la is the large
    area of each building. p is stock_rate_cfg(...).
    """
    return (
        la.isin(p["detroit_la_ids"])
        & b["building_type_id"].isin(p["exempt_types"])
        & b["residential_units"].between(1, p["exempt_max_units"])
        & (b["households"] == 0)
        & (b["non_residential_sqft"].fillna(0) == 0)
        & (b["building_age"] >= p["min_age"])
        & (b["sp_filter"] >= 0)
        & (b["event_id"].fillna(0) <= 0)
    )


def stock_by_city(b, la, p):
    """Stock per city and category (index city_id, columns CATEGORIES).

    Residential outside Detroit: units of type k built before the cut year.
    Detroit residential: units of type k in the vacant pool.
    Non-residential: sqft of buildings built before the non-res cut year.
    b also needs city_id and year_built.
    """
    detroit = la.isin(p["detroit_la_ids"])
    pool = vacant_pool_mask(b, la, p)
    old_res = b["year_built"] <= p["res_max_year_built"]
    old_nonres = b["year_built"] <= p["nonres_max_year_built"]
    out = {}
    for cat, btype in RES_CATEGORIES.items():
        in_stock = (b["building_type_id"] == btype) & np.where(detroit, pool, old_res)
        out[cat] = b["residential_units"].where(in_stock, 0).groupby(b["city_id"]).sum()
    out[NONRES_CATEGORY] = b["non_residential_sqft"].fillna(0).where(old_nonres, 0).groupby(b["city_id"]).sum()
    return pd.DataFrame(out).fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation: targets, vacancy floor, carry-over (scored_demolition_events)
# ─────────────────────────────────────────────────────────────────────────────

def rates_by_city(stock_rates):
    """configs/demolition_stock_rates.csv (long) -> rate, large area, floor per city."""
    rates = stock_rates.pivot(index="city_id", columns="category", values="rate").reindex(columns=CATEGORIES)
    city = stock_rates.groupby("city_id")[["large_area_id", "floor"]].first()
    return rates.fillna(0.0), city["large_area_id"], city["floor"]


def stock_targets(stock, rates, rate_mult=1.0):
    """Yearly target per city and category: rate x current stock x multiplier."""
    cities = rates.index.union(stock.index)
    return rates.reindex(cities).fillna(0.0) * stock.reindex(cities).fillna(0.0) * rate_mult


def floor_cap(vacant, units, floor):
    """Units that can be removed while vacancy stays at or above the floor.

    Removing x vacant units gives vacancy (V - x) / (U - x); it equals the
    floor at x = (V - f U) / (1 - f). Rounded down, not below 0.
    """
    x = (np.asarray(vacant, float) - np.asarray(floor, float) * np.asarray(units, float)) / (1 - np.asarray(floor, float))
    return np.floor(np.clip(x, 0, None))


def allocate_cap(targets, cap, available):
    """Split one cap over categories (Series by category).

    No change when the targets fit under the cap. Otherwise the cap goes to the
    categories in proportion to their targets, each up to min(target, units
    available); what one category cannot use moves to the others.
    """
    targets = targets.astype(float)
    if targets.sum() <= cap:
        return targets
    room = np.minimum(targets, available.reindex(targets.index).fillna(0).astype(float))
    out = pd.Series(0.0, index=targets.index)
    remaining = float(cap)
    for _ in range(len(targets)):
        open_ = (room - out) > 1e-9
        if remaining <= 1e-9 or not open_.any() or targets[open_].sum() <= 0:
            break
        add = np.minimum(targets[open_] / targets[open_].sum() * remaining, (room - out)[open_])
        out[open_] += add
        remaining -= add.sum()
    return out


def round_target(target, u):
    """Whole units to demolish from a fractional target: the integer part, plus
    one more with probability equal to the fraction (u: uniform draw in [0, 1)).
    Same expected volume as the target, so small city targets add up over years."""
    whole = np.floor(target)
    return whole + (1.0 if u < target - whole else 0.0)
