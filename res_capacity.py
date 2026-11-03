"""Feasible residential capacity of a city: the units the residential developer could
build from profitable proposals. It caps the city's yearly developer target.
Plain pandas functions, no orca.
"""
import numpy as np


def net_units(feasibility, ave_unit_size, current_units, min_unit_size):
    """Net new units per proposal, as in Developer.pick:
    round(residential_sqft / max(ave_unit_size, min_unit_size)) - current_units."""
    unit_size = np.maximum(ave_unit_size.reindex(feasibility.index).values, min_unit_size)
    return ((feasibility["residential_sqft"].values / unit_size).round()
            - current_units.reindex(feasibility.index).values)


def feasible_capacity(feasibility, parcel_size, ave_unit_size, current_units,
                      min_unit_size, max_parcel_size, allowed=None, profitable_only=True):
    """Sum over parcels of the largest positive net units among the parcel's buildable
    profitable proposals (max_profit > 0, max_profit_far > 0, parcel_size < max_parcel_size).
    One value per parcel: the table holds several FAR variants and forms per parcel.
    allowed: optional boolean mask of other proposal rules (e.g. the land phase rules).
    profitable_only=False also counts the loss-making proposals as the least-loss pass
    selects them: per parcel, the one with the least loss per sqft. A parcel then counts
    the larger of its profitable and its least-loss units."""
    if len(feasibility) == 0:
        return 0
    net = net_units(feasibility, ave_unit_size, current_units, min_unit_size)
    ok = ((feasibility["max_profit_far"].values > 0)
          & (parcel_size.reindex(feasibility.index).values < max_parcel_size) & (net > 0))
    if allowed is not None:
        ok &= np.asarray(allowed, bool)
    profitable = feasibility["max_profit"].values > 0
    f = feasibility.assign(_net=net)
    best = f[ok & profitable].groupby(level=0)["_net"].max()
    if not profitable_only:
        loss = f[ok & ~profitable]
        loss = loss.iloc[np.argsort(-(loss["max_profit"] / loss["building_sqft"]).values, kind="stable")]
        loss = loss[~loss.index.duplicated(keep="first")]["_net"]
        best = best.combine(loss, max, fill_value=0)
    return int(best.sum())
