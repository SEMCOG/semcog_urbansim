import os
import yaml
import shutil
import random
import pandas as pd
import numpy as np
import orca
import verify_data_structure
import utils


@orca.injectable("year")
def year():
    default_year = 2020
    try:
        iter_var = orca.get_injectable("iter_var")
        if iter_var is not None:
            return iter_var
        else:
            return default_year
    except:
        return default_year


orca.add_injectable("transcad_available", False)

# maps building type ids to general building types; reduces dimensionality

# keys: binging type
# vause: network filter landues general_type
orca.add_injectable(
    "building_type_map",
    {
        11: "Institutional",
        13: "Institutional",
        14: "Institutional",
        21: "Retail",
        23: "Office",
        31: "Industrial",
        32: "Industrial",
        33: "Industrial",
        41: "TCU",
        42: "TCU",
        51: "Medical",
        52: "Medical",
        53: "Medical",
        61: "Entertainment",
        63: "Entertainment",
        65: "Hospitality",
        71: "Others",
        81: "Residential",
        82: "Residential",
        83: "Residential",
        84: "Residential",
        91: "Entertainment",
        92: "Institutional",
        93: "Institutional",
        94: "othercommercial",
        95: "TCU",
    },
)

###

##current building types
##

###


# this maps building "forms" from the developer model
# to building types so that when the developer builds a
# "form" this can be converted for storing as a type
# in the building table - in the long run, the developer
# forms and the building types should be the same and the
# developer model should account for the differences

# keys: from proforma forms
# valus: biling typs aplyed to parcelses
orca.add_injectable(
    "form_to_btype",
    {
        "residential": [81, 82, 83],
        "industrial": [31, 32, 33],
        "retail": [21, 65],
        "office": [23],
        "medical": [51, 52, 53],
        "entertainment": [61, 63, 91],
        # "mixedresidential": [21, 81, 82, 83],
        # "mixedoffice": [23, 81, 82, 83],
    },
)

seed = 271828

# seed = 79
print("using seed", seed)
random.seed(seed)
np.random.seed(seed)
utils.run_log(f"Seed: {seed}")

working_store = 'data/checkpoint_store.h5'

def load_latest_input_hdf():
    data_path = r"/mnt/hgfs/urbansim/RDF2050/model_inputs/base_hdf"
    if os.path.exists(data_path) == False:
        data_path = "/home/da/share/U_RDF2050/model_inputs/base_hdf"
    hdf_list = [
        (data_path + "/" + f)
        for f in os.listdir(data_path)
        if ("forecast_data_input" in f) & (f[-3:] == ".h5")
    ]
    hdf_last = max(hdf_list, key=os.path.getctime)
    hdf_last = "/mnt/hgfs/urbansim/RDF2050/model_inputs/base_hdf/forecast_data_input_031523.h5"
    utils.run_log(f"Data: {hdf_last}")

    return hdf_last

def load_last_checkpoint(runnum):
    # example runnum run1001.h5
    run_path = "runs"
    hdf_path = os.path.join(run_path, runnum)
    utils.run_log(f"Loading checkpoint data: {hdf_path}")
    saved_run = pd.HDFStore(hdf_path, "r")
    return saved_run

def update_store_from_checkpoint(store, checkpoint):
    tbs_to_update = [
        "buildings",
        "jobs",
        "parcels",
        "households",
        "persons",
        "group_quarters",
        "dropped_buildings",
        "bg_hh_increase",
    ]
    # get the last year finished
    year = max([int(k.split('/')[1]) for k in checkpoint.keys() if k.split('/')[1]!='base'])
    orca.add_injectable('checkpoint_year', year)
    for k in checkpoint.keys():
        if "/%s/" % year not in k:
            continue
        tb = k.split('/')[-1]
        if tb not in tbs_to_update:
            continue
        print("Loading table %s from checkpoint year %s..." % (tb, year))
        if tb in store:
            cols = store[tb].columns
            dtypes = store[tb].dtypes
            store[tb] = checkpoint[k][cols].astype(dtypes)
        else:
            store[tb] = checkpoint[k]
    return store

### Transportation accessibility variables
NEAR_MAX_VARS = {
    "walk": {
        "indicator_table": "accessibility_walk_indicator_by_parcel",
        "fillna_val": 35.0,
        "column_names": [
            "hospitals_walk_near_max30",
            "urgent_cares_walk_near_max30",
            "health_centers_walk_near_max30",
            "all_healthcare_walk_near_max30",
            "grocery_stores_walk_near_max30",
            "libraries_walk_near_max30",
            "parks_local_walk_near_max30",
            "parks_bike_walk_near_max30",
            "parks_school_walk_near_max30",
            "parks_local_school_walk_near_max30",
            "schools_k8_walk_near_max30",
            "schools_912_walk_near_max30",
            "pharmacies_walk_near_max30",
            "childcare_walk_near_max30",
            "fire_stations_walk_near_max30",
            "fixed_route_bus_walk_near_max30",
            "american_job_centers_walk_near_max30",
            "community_colleges_walk_near_max30",
            "passenger_train_stations_walk_near_max30",
        ]
    },
    "bike": {
        "indicator_table": "accessibility_bike_indicator_by_parcel",
        "fillna_val": 65.0,
        "column_names": [
            "hospitals_bike_near_max60",
            "urgent_cares_bike_near_max60",
            "health_centers_bike_near_max60",
            "all_healthcare_bike_near_max60",
            "grocery_stores_bike_near_max60",
            "libraries_bike_near_max60",
            "parks_local_bike_near_max60",
            "parks_bike_bike_near_max60",
            "parks_school_bike_near_max60",
            "parks_local_school_bike_near_max60",
            "schools_k8_bike_near_max60",
            "schools_912_bike_near_max60",
            "pharmacies_bike_near_max60",
            "childcare_bike_near_max60",
            "fire_stations_bike_near_max60",
            "fixed_route_bus_bike_near_max60",
            "american_job_centers_bike_near_max60",
            "community_colleges_bike_near_max60",
            "passenger_airports_bike_near_max60",
            "passenger_train_stations_bike_near_max60",
        ]
    },
    "drive": {
        "indicator_table": "accessibility_drive_indicator_by_parcel",
        "fillna_val": 65.0,
        "column_names": [
            "hospitals_drive_near_max60",
            "urgent_cares_drive_near_max60",
            "health_centers_drive_near_max60",
            "all_healthcare_drive_near_max60",
            "grocery_stores_drive_near_max60",
            "libraries_drive_near_max60",
            "parks_vehicle_drive_near_max60",
            "schools_k8_drive_near_max60",
            "schools_912_drive_near_max60",
            "pharmacies_drive_near_max60",
            "childcare_drive_near_max60",
            "fire_stations_drive_near_max60",
            "american_job_centers_drive_near_max60",
            "community_colleges_drive_near_max60",
            "passenger_airports_drive_near_max60",
            "passenger_train_stations_drive_near_max60",
        ]
    }
}

# Variables for 'cumulative' and 'gravity' accessibility (fillna_value is 0)
CUMULATIVE_VARS = {
    "walk": {
        "indicator_table": "accessibility_walk_indicator_by_parcel",
        "fillna_val": 0,
        "column_names": [
            "jobs_walk_cumulative_5min", "jobs_walk_cumulative_10min",
            "jobs_walk_cumulative_15min", "jobs_walk_cumulative_30min",
            "fixed_route_bus_weekday_walk_10min", "fixed_route_bus_weekend_walk_10min"
        ]
    },
    "bike": {
        "indicator_table": "accessibility_bike_indicator_by_parcel",
        "fillna_val": 0,
        "column_names": [
            "jobs_bike_cumulative_5min", "jobs_bike_cumulative_10min",
            "jobs_bike_cumulative_15min", "jobs_bike_cumulative_30min",
            "fixed_route_bus_weekday_bike_10min", "fixed_route_bus_weekend_bike_10min"
        ]
    },
    "drive": {
        "indicator_table": "accessibility_drive_indicator_by_parcel",
        "fillna_val": 0,
        "column_names": [
            "jobs_drive_cumulative_10min", "jobs_drive_cumulative_15min",
            "jobs_drive_cumulative_20min", "jobs_drive_cumulative_25min",
            "jobs_drive_cumulative_30min", "jobs_drive_cumulative_45min",
            "jobs_drive_gravity_90min"
        ]
    }
}
orca.add_injectable("NEAR_MAX_VARS", NEAR_MAX_VARS)
orca.add_injectable("CUMULATIVE_VARS", CUMULATIVE_VARS)

def verify():
    # load latest input hdf
    # hdf_last = load_latest_input_hdf()
    hdf_last = "/mnt/hgfs/urbansim/RDF2050/model_inputs/base_hdf/forecast_data_input_031523.h5"
    orca.add_injectable("input_hdf_path", hdf_last)
    hdf_store = pd.HDFStore(hdf_last, "r")
    # hdf = pd.HDFStore(data_path + "/" +"forecast_data_input_091422.h5", "r")
    print("HDF data: ", hdf_last)

    if orca.is_injectable('use_checkpoint') and orca.get_injectable('use_checkpoint'):
        # copy input hdf
        shutil.copy(hdf_last, working_store)
        hdf_store.close()
        hdf_store = pd.HDFStore(working_store, 'a')
        # load from the last check point
        saved_runnum = orca.get_injectable('runnum_to_resume')
        saved_run = load_last_checkpoint(saved_runnum)
        hdf_store = update_store_from_checkpoint(hdf_store, saved_run)

    # verifying data structure and save data structure config
    new = verify_data_structure.yaml_from_store(hdf_store)
    with open("configs/data_structure.yaml", "w") as out:
        out.write(new)

    return hdf_store


# 2045 input hdf
orca.add_injectable('hdf_input_2045', '/mnt/hgfs/urbansim/RDF2045/data/base_year/all_semcog_data_02-02-18-final-forecast.h5')
orca.add_injectable('forecast_input_2040', '/mnt/hgfs/urbansim/RDF2050/model_improvements/2024_spring/2010_data')

orca.add_injectable("store", verify())