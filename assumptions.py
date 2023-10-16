import os
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

working_store = '/home/da/semcog_urbansim/data/checkpoint_store.h5'

def load_latest_input_hdf():
    data_path = r"/home/da/share/urbansim/RDF2050/model_inputs/base_hdf"
    if os.path.exists(data_path) == False:
        data_path = "/home/da/share/U_RDF2050/model_inputs/base_hdf"
    hdf_list = [
        (data_path + "/" + f)
        for f in os.listdir(data_path)
        if ("forecast_data_input" in f) & (f[-3:] == ".h5")
    ]
    hdf_last = max(hdf_list, key=os.path.getctime)
    hdf_last = "/home/da/share/urbansim/RDF2050/model_inputs/base_hdf/forecast_data_input_031523.h5"
    utils.run_log(f"Data: {hdf_last}")

    return hdf_last

def load_last_checkpoint(runnum):
    # example runnum run1001.h5
    run_path = "/home/da/semcog_urbansim/runs"
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

def verify():
    # load latest input hdf
    # hdf_last = load_latest_input_hdf()
    hdf_last = '/home/da/share/urbansim/RDF2050/model_inputs/base_hdf/forecast_data_input_031523.h5'
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
    with open("/home/da/semcog_urbansim/configs/data_structure.yaml", "w") as out:
        out.write(new)

    return hdf_store


orca.add_injectable("store", verify())

