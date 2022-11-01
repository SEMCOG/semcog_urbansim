import warnings

import numpy as np
import orca
import pandas as pd
from urbansim.utils import misc
from os import path

import assumptions

warnings.filterwarnings("ignore", category=pd.io.pytables.PerformanceWarning)

table_dir = "~/semcog_urbansim/data"

for name in [
    "remi_pop_total",
    "persons",
    "parcels",
    "zones",
    "semmcds",
    "counties",
    "employment_sectors",
    "building_sqft_per_job",
    "annual_relocation_rates_for_households",
    "annual_relocation_rates_for_jobs",
    "annual_employment_control_totals",
    "travel_data",
    "zoning",
    "large_areas",
    "building_types",
    "land_use_types",
    # "workers_labor_participation_rates",
    # "workers_employment_rates_by_large_area_age",
    "workers_employment_rates_by_large_area",
    "transit_stops",
    "crime_rates",
    "schools",
    "poi",
    "group_quarters",
    "group_quarters_control_totals",
    "annual_household_control_totals",
    "events_addition",
    "events_deletion",
    "refiner_events",
    "income_growth_rates",
    "target_vacancies",
    "target_vacancies_mcd",
    "demolition_rates",    
    "landmark_worksites",
    "multi_parcel_buildings"
]:
    store = orca.get_injectable("store")
    orca.add_table(name, store[name])

# #35 change csv column name from b_city_id to city_id
# orca.add_table('extreme_hu_controls', pd.read_csv(
#     path.join(table_dir, "extreme_hu_controls.csv"), index_col='b_city_id'))
# orca.add_table(
#     "extreme_hu_controls",
#     pd.read_csv(path.join(table_dir, "extreme_hu_controls.csv"), index_col="city_id"),
# )


@orca.table("mcd_total")
def mcd_total():
    return pd.read_csv(path.join(table_dir, "mcd_2050_draft_noreview.csv")).set_index(
        "semmcd"
    )


@orca.table("debug_res_developer")
def debug_res_developer():
    return pd.DataFrame(columns=["year", "mcd", "target_units", "units_added"])


@orca.table("bg_hh_increase")
def bg_hh_increase():
    bg_hh_inc = pd.read_csv(path.join(table_dir, "ACS_HH_14_19_BG.csv"))
    bg_hh_inc["GEOID"] = bg_hh_inc["GEOID"].astype(int)
    # initialized iteration variable
    bg_hh_inc["occupied"] = bg_hh_inc["OccupiedHU19"]
    bg_hh_inc["previous_occupied"] = bg_hh_inc["OccupiedHU14"]
    bg_hh_inc["occupied_year_minus_1"] = -1
    bg_hh_inc["occupied_year_minus_2"] = -1
    bg_hh_inc["occupied_year_minus_3"] = -1
    return bg_hh_inc[
        [
            "GEOID",
            "OccupiedHU19",
            "OccupiedHU14",
            "occupied",
            "previous_occupied",
            "occupied_year_minus_1",
            "occupied_year_minus_2",
            "occupied_year_minus_3",
        ]
    ].set_index("GEOID")


@orca.table(cache=True)
def buildings(store):
    df = store["buildings"]
    # Todo: combine two sqft prices into one and set non use sqft price to 0
    df.loc[df.market_value < 0, "market_value"] = 0
    df["sqft_price_nonres"] = df.market_value * 1.0 / 0.7 / df.non_residential_sqft
    df.loc[df.sqft_price_nonres > 1000, "sqft_price_nonres"] = 0
    df.loc[df.sqft_price_nonres < 0, "sqft_price_nonres"] = 0
    df["sqft_price_res"] = (
        df.market_value
        * 1.0
        / 0.7
        / (df.sqft_per_unit.astype(int) * df.residential_units)
    )
    df.loc[df.sqft_price_res > 1000, "sqft_price_res"] = 0
    df.loc[df.sqft_price_res < 0, "sqft_price_res"] = 0
    df.fillna(0, inplace=True)

    # df['hu_filter'] = 0
    df["mcd_model_quota"] = 0
    # cites = [551, 1155, 1100, 3130, 6020, 6040]
    # sample = df[df.residential_units > 0]
    # sample = sample[~(sample.index.isin(store['households'].building_id))]
    # # #35
    # for c in sample.b_city_id.unique():
    #     frac = 0.9 if c in cites else 0.5
    #     # #35
    #     df.loc[sample[sample.b_city_id == c].sample(frac=frac, replace=False).index.values, 'hu_filter'] = 1

    # TODO, this is placeholder. will update with special emp buildings lookup later

    df[
        "sp_filter"
    ] = 0  # special filter: for event location/buildings, landmark buildings, etc
    landmark_worksites = store["landmark_worksites"]
    df.loc[
        landmark_worksites[landmark_worksites.building_id.isin(
            df.index)].building_id, "sp_filter"
    ] = -1  # set landmark building_id as negative for blocking

    df["event_id"] = 0  # also add event_id for event reference

    return df


@orca.table(cache=True)
def households(store, buildings):
    df = store["households"]
    b = buildings.to_frame(["large_area_id"])
    b = b[b.large_area_id.isin({161.0, 3.0, 5.0, 125.0, 99.0, 115.0, 147.0, 93.0})]
    df.loc[df.building_id == -1, "building_id"] = np.random.choice(
        b.index.values, (df.building_id == -1).sum()
    )
    idx_invalid_building_id = np.in1d(df.building_id, b.index.values) == False
    df.loc[idx_invalid_building_id, "building_id"] = np.random.choice(
        b.index.values, idx_invalid_building_id.sum()
    )
    df["large_area_id"] = misc.reindex(b.large_area_id, df.building_id)
    # dtype optimization
    df["workers"] = df["workers"].fillna(0).astype(np.int8)
    df["children"] = df["children"].fillna(0).astype(np.int8)
    df["persons"] = df["persons"].astype(np.int8)
    df["cars"] = df["cars"].astype(np.int8)
    df["race_id"] = df["race_id"].astype(np.int8)
    df["income"] = df["income"].astype(np.int32)
    df["age_of_head"] = df["age_of_head"].astype(np.int8)
    df["large_area_id"] = df["large_area_id"].astype(np.uint8)
    return df.fillna(0)


@orca.table(cache=True)
def persons(store):
    df = store["persons"]
    df["relate"] = df["relate"].astype(np.int8)
    df["age"] = df["age"].astype(np.int8)
    df["worker"] = df["worker"].astype(np.int8)
    df["sex"] = df["sex"].astype(np.int8)
    df["race_id"] = df["race_id"].astype(np.int8)
    df["member_id"] = df["member_id"].astype(np.int8)
    df["household_id"] = df["household_id"].astype(np.int64)
    return df


@orca.table(cache=True)
def jobs(store, buildings):
    df = store["jobs"]
    b = buildings.to_frame(["large_area_id"])
    b = b[b.large_area_id.isin({161.0, 3.0, 5.0, 125.0, 99.0, 115.0, 147.0, 93.0})]
    df.loc[df.building_id == -1, "building_id"] = np.random.choice(
        b.index.values, (df.building_id == -1).sum()
    )
    idx_invalid_building_id = np.in1d(df.building_id, b.index.values) == False
    df.loc[idx_invalid_building_id, "building_id"] = np.random.choice(
        b.index.values, idx_invalid_building_id.sum()
    )
    df["large_area_id"] = misc.reindex(b.large_area_id, df.building_id)
    return df.fillna(0)


@orca.table(cache=True)
def parcels(store, zoning):
    parcels_df = store["parcels"]
    #  based on zoning.is_developable, adjust parcels pct_undev
    pct_undev = zoning.pct_undev.copy()
    # Parcel is NOT developable, leave as is unless events are present (173,616 parcels)
    pct_undev[zoning.is_developable == 0] = 100
    # Parcel is developable, but refer to the field “percent_undev” for how much of the parcel is actually developable (1,791,169 parcels)
    # Parcel is developable, but contains underground storage tanks
    pct_undev[zoning.is_developable == 2] += 10
    parcels_df["pct_undev"] = pct_undev.clip(0, 100).astype('int16')
    return parcels_df


@orca.table(cache=True)
def base_job_space(buildings):
    return buildings.jobs_non_home_based.to_frame("base_job_space")


# these are dummy returns that last until accessibility runs
for node_tbl in ["nodes", "nodes_walk", "nodes_drv"]:
    empty_df = pd.DataFrame()
    orca.add_table(node_tbl, empty_df)


# this specifies the relationships between tables
orca.broadcast("nodes_walk", "buildings", cast_index=True, onto_on="nodeid_walk")
orca.broadcast("nodes_walk", "parcels", cast_index=True, onto_on="nodeid_walk")
orca.broadcast("nodes_drv", "buildings", cast_index=True, onto_on="nodeid_drv")
orca.broadcast("nodes_drv", "parcels", cast_index=True, onto_on="nodeid_drv")
orca.broadcast("parcels", "buildings", cast_index=True, onto_on="parcel_id")
orca.broadcast("buildings", "households", cast_index=True, onto_on="building_id")
orca.broadcast("buildings", "jobs", cast_index=True, onto_on="building_id")
orca.broadcast("households", "persons", cast_index=True, onto_on="household_id")
orca.broadcast(
    "building_types", "buildings", cast_index=True, onto_on="building_type_id"
)
orca.broadcast("zones", "parcels", cast_index=True, onto_on="zone_id")
orca.broadcast("schools", "parcels", cast_on="parcel_id", onto_index=True)
