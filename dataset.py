import warnings

import numpy as np
import orca
import pandas as pd
from urbansim.utils import misc
from os import path

import assumptions
import utils
import input_paths

warnings.filterwarnings("ignore", category=pd.io.pytables.PerformanceWarning)

table_dir = "data"

for name in [
    "remi_hh_pop",
    "remi_income_ratios",       # per-LA cumulative income ratios (base yr 2022)
    "remi_local_price_ratios",  # per-LA cumulative price ratios (base yr 2022)
    "persons",
    "parcels",
    "zones",
    "semmcds",
    "counties",
    "employment_sectors",
    "building_sqft_per_job",
    "annual_relocation_rates_for_households", # need 2055 update
    "annual_relocation_rates_for_jobs", # need 2055 update
    "annual_employment_control_totals",
    "travel_data",
    "travel_data_2030",
    "zoning",
    "large_areas",
    "building_types",
    "land_use_types",
    # "workers_labor_participation_rates",
    "employed_workers_rate", # need 2055 update
    "transit_stops",
    "crime_rates",
    "schools",
    "points_of_interest_by_category",
    "travel_survey_bg_vars",  # block-group behavioral vars (joined to parcels by geoid)
    "group_quarters",
    "group_quarters_households",
    "group_quarters_control_totals",
    "annual_household_control_totals",
    "events_addition",
    "events_deletion",
    "refiner_events", # need 2055 update
    "target_vacancies", # need 2055 update
    "target_vacancies_mcd", # need 2055 update
    "demolition_rates",
    "landmark_worksites", # need 2055 update
    "mcd_total",
    "parcel_maz_crossing_shares",  # parcel->MAZ area shares for parcels spanning MAZ
    "dropped_buildings",
    "bg_hh_increase",
]:
    store = orca.get_injectable("store")
    if name not in store:
        print('skip loading %s while adding table to orca' % name)
        continue
    orca.add_table(name, store[name])

# #35 change csv column name from b_city_id to city_id
# orca.add_table('extreme_hu_controls', pd.read_csv(
#     path.join(table_dir, "extreme_hu_controls.csv"), index_col='b_city_id'))
# orca.add_table(
#     "extreme_hu_controls",
#     pd.read_csv(path.join(table_dir, "extreme_hu_controls.csv"), index_col="city_id"),
# )


@orca.table("debug_res_developer")
def debug_res_developer():
    return pd.DataFrame(columns=["year", "mcd", "target_units", "units_added"])


@orca.table("bg_hh_increase", cache=True)
def bg_hh_increase():
    # Base block-group household trend = 2020 -> 2025 household change, computed
    # from the 2020 base (BG_HH_2020_HDF) and the 2025 base
    def bg_hh_counts(hdf):
        h = pd.read_hdf(hdf, "households")
        b = pd.read_hdf(hdf, "buildings")
        p = pd.read_hdf(hdf, "parcels")
        pgeo = (26 * 10**10
                + p["county_id"].astype("int64") * 10**7
                + p["census_bg_id"].astype("int64"))
        hgeo = h["building_id"].map(b["parcel_id"].map(pgeo))
        return hgeo.dropna().astype("int64").value_counts()

    occ_2025 = bg_hh_counts(input_paths.BASE_HDF)        # 2055 base year (2025)
    occ_2020 = bg_hh_counts(input_paths.BG_HH_2020_HDF)  # 2050 base year (2020)
    bg = pd.DataFrame(
        {"occupied": occ_2025, "previous_occupied": occ_2020}
    ).fillna(0).astype(int)
    bg.index.name = "GEOID"
    bg["occupied_year_minus_1"] = -1
    bg["occupied_year_minus_2"] = -1
    bg["occupied_year_minus_3"] = -1
    return bg


@orca.table(cache=True)
def buildings(store):
    df = store["buildings"]
    # Skip recalculation when resuming from checkpoint - use checkpoint data as-is
    if orca.is_injectable('use_checkpoint') and orca.get_injectable('use_checkpoint'):
        return df
    df = df.fillna(0)
    # Todo: combine two sqft prices into one and set non use sqft price to 0
    df.loc[df.market_value < 0, "market_value"] = 0
    df["sqft_price_nonres"] = df.market_value * 1.0 / 0.7 / df.non_residential_sqft
    df.loc[df.sqft_price_nonres > 1500, "sqft_price_nonres"] = 0
    df.loc[df.sqft_price_nonres < 0, "sqft_price_nonres"] = 0
    df["sqft_price_res"] = (
        df.market_value
        * 1.0
        / 0.7
        / (df.sqft_per_unit.astype(int) * df.residential_units)
    )
    # fill out-of-bound sqft_price_res with la avg
    oob = (df.sqft_price_res > 1500) | (df.sqft_price_res < 0) | df.sqft_price_res.isna()
    df.loc[oob, "sqft_price_res"] = np.nan
    la_id = store["parcels"]["large_area_id"].reindex(df["parcel_id"].values).values
    df["_la"] = la_id
    la_avg = df[~oob].groupby("_la")["sqft_price_res"].mean()
    region_avg = df.loc[~oob, "sqft_price_res"].mean()
    df["sqft_price_res"] = df["sqft_price_res"].fillna(
        df["_la"].map(la_avg).fillna(region_avg)
    )
    df.drop(columns=["_la"], inplace=True)
    df.fillna(0, inplace=True)

    df["mcd_model_quota"] = 0

    # drop city_id if exists
    df = df.drop(columns=["city_id"], errors="ignore")

    # hu_filter assignment
    df["hu_filter"] = 0
    hu_cities = [551, 1155, 1100, 3130, 6020, 6040]
    b_city_id = misc.reindex(store["parcels"]["city_id"], df["parcel_id"]).fillna(0)
    sample = df[(df.residential_units > 0) & ~df.index.isin(store["households"].building_id)]
    sample_city = b_city_id.reindex(sample.index)
    for c in hu_cities:
        city_sample = sample.index[sample_city == c]
        if len(city_sample):
            chosen = pd.Series(city_sample).sample(frac=0.8, replace=False).values
            df.loc[chosen, "hu_filter"] = 1

    df["sp_filter"] = 0  # special filter: for event location/buildings, landmark buildings, etc
    # skip if not presented
    if "landmark_worksites" in store:
        landmark_worksites = store["landmark_worksites"]
        df.loc[
            landmark_worksites[landmark_worksites.building_id.isin(df.index)].building_id,
            "sp_filter",
        ] = -1  # set landmark building_id as negative for blocking

    df["event_id"] = 0  # also add event_id for event reference

    return df


@orca.table(cache=True)
def households(store, buildings):
    df = store["households"]
    # Skip recalculation when resuming from checkpoint
    if orca.is_injectable('use_checkpoint') and orca.get_injectable('use_checkpoint'):
        return df
    b = buildings.to_frame(["large_area_id", "residential_units"])
    b = b[b.large_area_id.isin({161.0, 3.0, 5.0, 125.0, 99.0, 115.0, 147.0, 93.0})]
    _bid_dtype = df["building_id"].dtype
    n_unplaced = (df.building_id == -1).sum()
    if n_unplaced:
        df.loc[df.building_id == -1, "building_id"] = np.random.choice(
            b.index.values, n_unplaced
        ).astype(_bid_dtype)

    bid_to_la = {
        1: 3, 2: 125, 3:99, 4: 161, 5: 115, 6: 147, 7: 93, 8: 5
    }
    idx_invalid_building_id = np.isin(df.building_id, b.index.values) == False
    hh_to_assign = df.loc[idx_invalid_building_id, "building_id"]
    for bid, laid in bid_to_la.items():
        local_hh = hh_to_assign[hh_to_assign//1000000 == bid]
        df.loc[local_hh.index, 'building_id'] = np.random.choice(
            b[(b.large_area_id==laid)&(b.residential_units>0)].index.values, local_hh.size
        ).astype(_bid_dtype)

    df["large_area_id"] = misc.reindex(b.large_area_id, df.building_id,)

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
    # Skip recalculation when resuming from checkpoint
    if orca.is_injectable('use_checkpoint') and orca.get_injectable('use_checkpoint'):
        return df
    b = buildings.to_frame(["large_area_id"])
    b = b[b.large_area_id.isin({161.0, 3.0, 5.0, 125.0, 99.0, 115.0, 147.0, 93.0})]
    _bid_dtype = df["building_id"].dtype
    n_unplaced = (df.building_id == -1).sum()
    if n_unplaced:
        df.loc[df.building_id == -1, "building_id"] = np.random.choice(
            b.index.values, n_unplaced
        ).astype(_bid_dtype)
    idx_invalid_building_id = np.isin(df.building_id, b.index.values) == False
    n_invalid = idx_invalid_building_id.sum()
    if n_invalid:
        df.loc[idx_invalid_building_id, "building_id"] = np.random.choice(
            b.index.values, n_invalid
        ).astype(_bid_dtype)
    df["large_area_id"] = misc.reindex(b.large_area_id, df.building_id)
    return df.fillna(0)


@orca.table(cache=True)
def parcels(store, zoning):
    parcels_df = store["parcels"]
    # Skip recalculation when resuming from checkpoint
    if orca.is_injectable('use_checkpoint') and orca.get_injectable('use_checkpoint'):
        return parcels_df
    #  based on zoning.is_developable, adjust parcels pct_undev
    pct_undev = zoning.pct_undev.copy()
    # Parcel is NOT developable, leave as is unless events are present (173,616 parcels)
    pct_undev[zoning.is_developable == 0] = 100
    # Parcel is developable, but refer to the field “percent_undev” for how much of the parcel is actually developable (1,791,169 parcels)
    # Parcel is developable, but contains underground storage tanks
    pct_undev[zoning.is_developable == 2] += 10
    parcels_df["pct_undev"] = pct_undev.clip(0, 100).astype("int16")
    parcels_df["pct_undev"] = parcels_df["pct_undev"].fillna(0)
    return parcels_df

@orca.table(cache=True)
def census_tracts(store):
    parcels_df = store["parcels"]
    # remove pseudo parcels with bg_id <0
    parcels_df = parcels_df[parcels_df['census_bg_id']>0]
    # need int32 to avoid overflow
    bg_id = parcels_df['census_bg_id'].astype('int32')
    cty_id = parcels_df['county_id'].astype('int32')
    tract_id = bg_id // 1000 + cty_id * 10000
    # craete tracts df
    tracts_df = pd.DataFrame({'tract_id': tract_id, 'county_id': cty_id})
    unique_tracts = tracts_df.groupby('tract_id').first()
    return unique_tracts

@orca.table(cache=True)
def base_job_space(buildings):
    return buildings.jobs_non_home_based.to_frame("base_job_space")

@orca.table(cache=True)
def building_to_maz_override(store):
    # baseyear building_id -> {maz_id, zone_id} override (fixes buildings whose
    # parcel spans multiple MAZ/TAZ). Source: main.h5 `building_to_maz_override`.
    return store["building_to_maz_override"][["maz_id", "zone_id"]]


@orca.table(cache=True)
def accessibility_walk_indicator_by_parcel():
    return pd.read_hdf(input_paths.ACCESS_INDICATORS_H5, "accessibility_walk_indicator_by_parcel")

@orca.table(cache=True)
def accessibility_bike_indicator_by_parcel():
    return pd.read_hdf(input_paths.ACCESS_INDICATORS_H5, "accessibility_bike_indicator_by_parcel")

@orca.table(cache=True)
def accessibility_drive_indicator_by_parcel():
    return pd.read_hdf(input_paths.ACCESS_INDICATORS_H5, "accessibility_drive_indicator_by_parcel")

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
orca.add_injectable("remi_base_year", 2025)
