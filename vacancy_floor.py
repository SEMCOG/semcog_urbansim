"""Vacancy floor and demolished space, shared by the demolition and developer steps.
Plain pandas functions, no orca.

Demolition of mostly empty buildings lowers a city's vacancy, and the developer does
not rebuild that space: the target space is reduced by the space demolished since the
base year. A floor keeps weak markets from sliding to 0% vacancy: demolition stops at
it, and the developer target never goes below it.

    housing:          floor = max(min, share x base-year vacancy)
    nonres job space: floor = share x base-year vacancy
"""
import numpy as np
import pandas as pd

# demolition steps whose removals are not rebuilt (developer redevelopment replaces
# a building on the same parcel and is not counted)
DEMOLITION_STEPS = ("scheduled_demolition_events", "scored_demolition_events")


def floor_rate(base_vacancy, share, min_rate=0.0):
    """Vacancy floor from the base-year vacancy."""
    return np.maximum(min_rate, share * np.asarray(base_vacancy, float))


def target_space(agents, target_vacancy, demolished, floor):
    """Target space with demolished space not rebuilt, never below the floor vacancy.

    A floor above the target vacancy is not used (it would build beyond the target).
    """
    agents = np.asarray(agents, float)
    tv = np.asarray(target_vacancy, float)
    f = np.minimum(np.asarray(floor, float), tv)
    return np.maximum(agents / (1 - tv) - np.asarray(demolished, float), agents / (1 - f))


def demolished_units(dropped):
    """Housing units removed by demolition per city_id."""
    d = dropped[dropped["step"].isin(DEMOLITION_STEPS)]
    return d.groupby("city_id")["residential_units"].sum()


def demolished_job_spaces(dropped, sqft_per_job, btype_group):
    """Job spaces removed by demolition per (city_id, form group).

    Job spaces = non_residential_sqft / sqft per job of the building type
    (dropped_buildings has no job_spaces for scheduled events).
    btype_group: building_type_id -> form group.
    """
    d = dropped[dropped["step"].isin(DEMOLITION_STEPS) & (dropped["non_residential_sqft"] > 0)]
    spaces = (d["non_residential_sqft"] / d["building_type_id"].map(sqft_per_job)).replace(
        [np.inf, -np.inf], np.nan).fillna(0)
    group = d["building_type_id"].map(btype_group)
    return spaces.groupby([d["city_id"], group]).sum()
