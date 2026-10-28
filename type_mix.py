"""Target mix of residential building types for the residential developer.

The developer's units in a large area are split by a target share per form
(single-family, condo, apartment) from a table by large area and year
(configs/res_developer.yaml: type_mix). The mix applies to all new units of the
year, scheduled events included: the developer fills each form's gap after the
events. Plain pandas functions, no orca.
"""
import numpy as np
import pandas as pd

# res_type_mix.csv columns per pro forma form
FORM_COL = {"single-family": "share_sf", "condo": "share_condo", "apartment": "share_apt"}
FORM_BTYPE = {"single-family": 81, "condo": 82, "apartment": 83}


def type_mix_cfg(res_cfg):
    """type_mix settings from res_developer.yaml; None (rule off) when the block is missing."""
    t = (res_cfg or {}).get("type_mix") or {}
    if not t.get("file"):
        return None
    return {"file": t["file"], "scenario": t.get("scenario")}


def load_shares(path, year, scenario=None):
    """Form shares of new units by large area for one year (rows sum to 1)."""
    df = pd.read_csv(path)
    if scenario is not None and "scenario" in df.columns:
        df = df[df.scenario == scenario]
    df = df[df.year == year].set_index("large_area_id")
    shares = pd.DataFrame({f: df[c] for f, c in FORM_COL.items()})
    return shares.div(shares.sum(axis=1), axis=0)


def split_units(total, weights):
    """Split an integer total by weights (largest remainder); returns {key: int}."""
    total = int(total)
    w = pd.Series(weights, dtype=float).clip(lower=0)
    if total <= 0 or w.sum() <= 0:
        return {k: 0 for k in w.index}
    raw = w / w.sum() * total
    out = np.floor(raw).astype(int)
    rest = total - out.sum()
    out[(raw - out).sort_values(ascending=False).index[:rest]] += 1
    return out.to_dict()


def developer_quota(dev_total, event_units, shares):
    """Developer units per form in a large area.

    Target per form = (developer + event units) x share; the developer builds the
    gap above the events (never negative), scaled to its own total.
    """
    total = dev_total + sum(event_units.get(f, 0) for f in shares)
    gap = {f: max(0.0, total * s - event_units.get(f, 0)) for f, s in shares.items()}
    return split_units(dev_total, gap if sum(gap.values()) > 0 else shares)


def allocate_to_cities(quota, city_targets, min_units=0):
    """Split one form's large-area quota over cities by their targets; {city: int}.

    With min_units, no city gets a positive quota below min_units (one building of the
    form): the city with the smallest such allocation (then the lowest id) is dropped
    and the quota is split again over the other cities, until every allocation is 0 or
    at least min_units. If no city can take min_units, the quota stays unmet (all 0).
    """
    targets = {m: t for m, t in city_targets.items() if t > 0}
    out = split_units(quota, targets)
    while min_units > 0:
        small = sorted((u, m) for m, u in out.items() if 0 < u < min_units)
        if not small:
            break
        del targets[small[0][1]]
        out = split_units(quota, targets)
    return {m: out.get(m, 0) for m in city_targets}
