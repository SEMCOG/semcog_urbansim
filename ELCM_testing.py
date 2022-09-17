#!/usr/bin/env python
# coding: utf-8
from urbansim.models.regression import RegressionModel

import os
import numpy as np
import pandas as pd
import orca
from urbansim.utils import misc
import sys
import time
from tqdm import tqdm

from dcm_ard_libs import minimize, neglog_DCM

# from guppy import hpy; h=hpy()
# import pymrmr

# suppress sklearn warnings
def warn(*args, **kwargs):
    pass

os.chdir("/home/da/semcog_urbansim")
sys.path.append("/home/da/forecast_data_input")
from data_checks.estimation_variables_2050 import *


# import utils
# data_path = r"/home/da/share/U_RDF2050/model_inputs/base_hdf"
data_path = r'/home/da/share/urbansim/RDF2050/model_inputs/base_hdf'
hdf_list = [
    (data_path + "/" + f)
    for f in os.listdir(data_path)
    if ("forecast_data_input" in f) & (f[-3:] == ".h5")
]
hdf_last = max(hdf_list, key=os.path.getctime)
# hdf = pd.HDFStore(hdf_last, "r")
hdf = pd.HDFStore(data_path + "/" +"forecast_data_input_091422.h5", "r")
print("HDF data: ", hdf_last)

var_validation_list = [
    (data_path + "/" + f)
    for f in os.listdir(data_path)
    if ("variable_validation" in f) & (f[-5:] == ".yaml")
]
var_validation_last = max(var_validation_list, key=os.path.getctime)
with open(var_validation_last, "r") as f:
    vars_config = yaml.load(f, Loader=yaml.FullLoader)
valid_b_vars = vars_config["buildings"]["valid variables"]
valid_hh_vars = vars_config["households"]["valid variables"]


vars_to_skip = [
    "large_area_id",
    "county_id",
    "parcel_id",
    "st_parcel_id",
    "geoid",
    "st_geoid",
    "b_ln_parcels_parcel_far",
    "parcels_parcel_far",
    "parcels_st_parcel_far",
    "parcels_census_bg_id",
    "nodeid_drv",
    "st_nodeid_drv",
    "nodeid_walk",
    "st_nodeid_walk",
    "semmcd",
    "city_id",
    "census_bg_id",
    "x",
    "y",
    "zone_id",
    "improvement_value",
    "market_value",
    "landvalue",
    "parcels_landvalue",
    "b_ln_sqft_price_nonres",
    "zones_tazce10_n",
    "tazce10",
    "parcels_zones_tazce10_n",
    "parcels_st_zones_tazce10_n",
    "st_parcels_zones_tazce10_n",
    "st_parcels_st_zones_tazce10_n",
    "st_zones_tazce10_n",
    "parcels_land_use_type_id",
    "st_parcels_land_use_type_id",
    "st_parcels_max_dua",
    "parcels_max_dua",
    "parcels_max_height",
    "st_parcels_max_height",
    "parcels_school_id",
    "st_parcels_school_id",
    "parcels_sev_value",
    "st_parcels_sev_value",
    "parcels_centroid_x",
    "st_parcels_centroid_x",
    "parcels_centroid_y",
    "st_parcels_centroid_y",
    "parcels_school_id",
    "st_parcels_school_id",
    "parcels_land_cost",
    "st_parcels_land_cost",
    "b_ln_market_value",
    "st_b_ln_market_value",
    "general_type"
]

hh_vars_to_skip = [
	
]

for var in valid_b_vars:
    if "parcels_" + var in valid_b_vars:
        vars_to_skip.append("parcels_" + var)


def apply_filter_query(df, filters=None):
    if filters:
        if isinstance(filters, str):
            query = filters
        else:
            query = " and ".join(filters)
        return df.query(query)
    else:
        return df


def load_hlcm_df():
    # load both hh
    hh = households.to_frame([var for var in valid_hh_vars if var not in hh_vars_to_skip])
    b = buildings.to_frame([var for var in valid_b_vars[:150] if var not in vars_to_skip] + ["large_area_id"])
    return hh, b

# config
choice_column = "building_id"
hh_sample_size = 100
estimation_sample_size = 10
# load variables
RELOAD = False
if RELOAD:
    orca.add_injectable("store", hdf)
    load_tables_to_store()
    from notebooks.models_test import *
    import variables
    buildings = orca.get_table("buildings")
    households = orca.get_table("households")
    orca.run(["build_networks"])
    orca.run(["neighborhood_vars"])
    hh, b = load_hlcm_df()
    # sampling hh
    hh = hh[hh.large_area_id == 125]
    hh = hh[hh.building_id > 1]
    # hh = hh.sample(estimation_sample_size)
    hh = hh.sample(hh_sample_size)
    picked_bid = hh["building_id"]
    hh = hh.reset_index()
    hh = hh.fillna(0)
    # sampling b
    b = b[b.large_area_id == 125]
    b = b[b.residential_units > 0]
    b_sample = b[~b.index.isin(hh.building_id)].sample((estimation_sample_size-1)*hh_sample_size)
    b_sample = pd.concat([b.loc[hh.building_id], b_sample])
    b_sample = b_sample.reset_index()
    b_sample = b_sample.fillna(0)
    # remove unnecessary col in HH
    hh = hh[[col for col in hh.columns if 'id' not in col ]]
    # remove unnecessary col in buildings
    b_sample = b_sample[[col for col in b_sample.columns if col not in [ 'large_area_id']]]

    X_df = pd.concat([pd.concat([hh]*estimation_sample_size).reset_index(), b_sample], axis=1)
    # Y: 1 for the building picked
    Y = X_df.building_id.isin(picked_bid).astype(int).values
    X_df = X_df[[col for col in X_df.columns if col not in ['building_id', 'index']]]
    orig_var_names = X_df.columns
    # print("orig_var_names",orig_var_names)
    # df to ndarray
    X = X_df.values
    # standardize X
    used_val = np.arange(X.shape[1])[np.std(X, axis=0, dtype=np.float64) > 0]
    X = X[:, np.std(X, axis=0, dtype=np.float64) > 0]
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0, dtype=np.float64)
    # shuffle X
    shuffled_index = np.arange(Y.size)
    np.random.shuffle(shuffled_index)
    X = X[shuffled_index, :].astype(float)
    Y = Y[shuffled_index].reshape(-1, 1)
    # TODO: Y_onehot
    Y_onehot = Y
    # availablechoice is 1
    available_choice = np.ones((X.shape[0], 1))

    # theta: m x 1
    theta = np.zeros((X.shape[1], 1))
else:
    cache_hdf = pd.HDFStore('lcm_testing.hdf', 'r')
    theta, X, Y, Y_onehot, available_choice = cache_hdf['theta'], cache_hdf['X'], cache_hdf['Y'], cache_hdf['Y_onehot'], cache_hdf['available_choice']
    theta, X, Y, Y_onehot, available_choice = theta.values, X.values, Y.values, Y_onehot.values, available_choice.values

theta_optim_full = minimize(theta, neglog_DCM, -20000, X, Y, Y_onehot, available_choice)
# print([orig_var_names[used_val][i] for i in theta_optim_full.argsort()[0][:20]])
print(Y.argsort()[::-1][:20])
print(X.dot(theta_optim_full).reshape(-1).argsort()[::-1][:20])
print([x for x in Y.reshape(-1).argsort()[::-1][:hh_sample_size] if x in X.dot(theta_optim_full).reshape(-1).argsort()[::-1][:hh_sample_size]])

print('done')
