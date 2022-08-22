"""Testing the availability of the variables to be used for the 2050 forecast Estimation"""
import numpy as np
import pandas as pd
import orca
from os import path
from urbansim.utils import misc
from pprint import pprint


# import models, utils

table_dir = "~/semcog_urbansim/data"


def main():
    hdf = pd.HDFStore(
        "/home/da/share/urbansim/RDF2050/model_inputs/base_hdf/forecast_data_input_072722.h5",
        mode="r",
    )
    orca.add_injectable("store", hdf)
    load_tables_to_store()
    import variables

    households = orca.get_table("households")
    valid_hh = [check_var(households, x) for x in households.columns]
    valid_hh_vars = [x for i, x in enumerate(households.columns) if valid_hh[i]]
    invalid_hh_vars = [x for i, x in enumerate(households.columns) if not valid_hh[i]]

    buildings = orca.get_table("buildings")
    valid_b = [check_var(buildings, x) for x in buildings.columns]
    valid_b_vars = [x for i, x in enumerate(buildings.columns) if valid_b[i]]
    invalid_b_vars = [x for i, x in enumerate(buildings.columns) if not valid_b[i]]

    jobs = orca.get_table("jobs")
    valid_j = [check_var(jobs, x) for x in jobs.columns]
    valid_j_vars = [x for i, x in enumerate(jobs.columns) if valid_j[i]]
    invalid_j_vars = [x for i, x in enumerate(jobs.columns) if not valid_j[i]]

    # 2045 vars list
    with open("variables_2045.txt", "r") as f:
        old_hh_vars, old_b_vars, old_j_vars = [
            l.strip().split(",") for l in f.readlines()
        ]

    print(
        "households total number of columns: %s (%s valid, %s invalid)"
        % (len(households.columns), len(valid_hh_vars), len(invalid_hh_vars))
    )
    print("Invalid households vars:")
    pprint(invalid_hh_vars, indent=4)
    print(
        "buildings total number of columns: %s (%s valid, %s invalid)"
        % (len(buildings.columns), len(valid_b_vars), len(invalid_b_vars))
    )
    print("Invalid buildings vars:")
    pprint(invalid_b_vars, indent=4)
    print(
        "jobs total number of columns: %s (%s valid, %s invalid)"
        % (len(jobs.columns), len(valid_j_vars), len(invalid_j_vars))
    )
    print("Invalid jobs vars:")
    pprint(invalid_j_vars, indent=4)

    # comparing to last forecast
    print("vars in the last forecast but not in the current (hhs, buildings, jobs):")
    pprint([x for x in old_hh_vars if x not in households.columns])
    pprint([x for x in old_b_vars if x not in buildings.columns])
    pprint([x for x in old_j_vars if x not in jobs.columns])
    hdf.close()
    return


def load_tables_to_store():
    store = orca.get_injectable("store")
    for name in store.keys():
        if name[1:] in ["buildings", "households", "persons", "jobs", "base_job_space"]:
            continue
        orca.add_table(name[1:], store[name])
    # these are dummy returns that last until accessibility runs
    for node_tbl in ["nodes", "nodes_walk", "nodes_drv"]:
        empty_df = pd.DataFrame()
        orca.add_table(node_tbl, empty_df)
    return


def check_var(df_wrapper, var_name):
    try:
        re = df_wrapper.to_frame(var_name)
        return re.shape[1] == 1
    except:
        return False


@orca.table("mcd_total")
def mcd_total():
    return pd.read_csv(path.join(table_dir, "mcd_2050_draft_noreview.csv")).set_index(
        "semmcd"
    )


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
        * 1.25
        / 0.7
        / (df.sqft_per_unit.astype(int) * df.residential_units)
    )
    df.loc[df.sqft_price_res > 1000, "sqft_price_res"] = 0
    df.loc[df.sqft_price_res < 0, "sqft_price_res"] = 0
    df.fillna(0, inplace=True)

    # df['hu_filter'] = 0
    df["mcd_model_quota"] = 0

    df[
        "sp_filter"
    ] = 0  # special filter: for event location/buildings, landmark buildings, etc
    # df.loc[
    #     store["landmark_worksites"].building_id, "sp_filter"
    # ] = -1  # set landmark building_id as negative for blocking

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
def base_job_space(buildings):
    return buildings.jobs_non_home_based.to_frame("base_job_space")


if __name__ == "__main__":
    main()
