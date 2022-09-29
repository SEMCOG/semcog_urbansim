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
import time

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
hdf = pd.HDFStore(hdf_last, "r")
# hdf = pd.HDFStore(data_path + "/" +"forecast_data_input_091422.h5", "r")
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


def load_hlcm_df(hh_var, b_var):
    # load both hh
    hh = households.to_frame(hh_var)
    b = buildings.to_frame(b_var)
    return hh, b

def columns_in_vars(vars):
    hh_columns, b_columns = [], []
    for varname in vars:
        if ':' in varname:
            vs = varname.split(':')
            hh_columns.append(vs[0])
            b_columns.append(vs[1])
        else:
            if varname in valid_hh_vars:
                hh_columns.append(varname)
            elif varname in valid_b_vars:
                b_columns.append(varname)
            else:
                print(varname, " not found in both hh and buildings table")
    return hh_columns, b_columns
            

def get_interaction_vars( df, varname):
    """Get interaction variables from variable name

    Args:
        varname (string): name of the interaction variable
    """
    if ":" in varname:
        var1, var2 = varname.split(":")
        var1, var2 = var1.strip(), var2.strip()
        return (df[var1] * df[var2]).values.reshape(-1, 1)
    else:
        return df[varname].values.reshape(-1, 1)

@orca.column('households', cache=True, cache_scope='iteration')
def residential_units(households, buildings):
    return misc.reindex(buildings.residential_units, households.building_id)

@orca.column('households', cache=True, cache_scope='iteration')
def year_built(households, buildings):
    return misc.reindex(buildings.year_built, households.building_id)

@orca.column('households', cache=True, cache_scope='iteration')
def mcd_model_quota(households, buildings):
    return misc.reindex(buildings.mcd_model_quota, households.building_id)

used_vars = pd.read_excel("/home/da/share/urbansim/RDF2050/model_estimation/configs_hlcm_2050.xlsx", sheet_name=1)
v1 = used_vars[~used_vars["new variables 1"].isna()]["new variables 1"].unique()
v2 = used_vars[~used_vars["new variables 2"].isna()]["new variables 2"].unique()
vars_to_use = np.array(list(set(v1.tolist()).union(v2.tolist())))
# vars_to_use = used_vars[0].unique()

# config
choice_column = "building_id"
hh_sample_size = 10000
estimation_sample_size = 50
# load variables
RELOAD = True
if RELOAD:
    orca.add_injectable("store", hdf)
    load_tables_to_store()
    from notebooks.models_test import *
    import variables
    buildings = orca.get_table("buildings")
    households = orca.get_table("households")
    # set year to 2050 
    orca.add_injectable('year', 2050)
    orca.run(["build_networks"])
    orca.run(["neighborhood_vars"])
    orca.run(["mcd_hu_sampling"])
# TODO: get vars from vars list from last forecast
    hh_columns, b_columns = columns_in_vars(vars_to_use)

    hh_filter_columns = ["building_id", "large_area_id", "mcd_model_quota", "year_built", "residential_units"]
    b_filter_columns = ["large_area_id", "mcd_model_quota"]

    hh_var = hh_columns + hh_filter_columns
    b_var = b_columns + b_filter_columns
    hh, b = load_hlcm_df(hh_var, b_var)
    # sampling hh
    # from the new move-ins, last 5-10 years
    # weighted by mcd_quota
    hh = hh[hh.large_area_id == 125]
    hh = hh[hh.building_id > 1]
    hh = hh[hh.residential_units > 0]
    hh = hh[hh.year_built > 2005]
    hh["mcd_model_quota"] += 1 # add 1 to all hh's mcd_model_quota for weights
    hh = hh.sample(hh_sample_size, weights="mcd_model_quota")
    # hh = hh.sample(hh_sample_size)
    picked_bid = hh["building_id"]
    hh = hh.reset_index()
    hh = hh.fillna(0)
    # sampling b
    # building age < 10
    # weighted by mcd_quota
    # b = b[b.large_area_id == 125]
    # b = b[b.residential_units > 0]
    # b = b[b.year_built > 2005]
    # b["mcd_model_quota"] += 1 # add 1 to all buildings's mcd_model_quota for weights
    # sample buildings from the chosen HH's buildings list
    uhh_id = hh.building_id.unique()
    sampled_b_id = []
    for _ in range(estimation_sample_size-1):
        for j in hh.building_id:
           sampled_b_id.append(np.random.choice(uhh_id[uhh_id!=j])) 
    # b_sample = b[~b.index.isin(hh.building_id)].sample( # replace sample or not
    #     (estimation_sample_size-1)*hh_sample_size, replace=False, weights="mcd_model_quota")
    b_sample = b.loc[sampled_b_id]
    b_sample = pd.concat([b.loc[hh.building_id], b_sample])
    b_sample = b_sample.reset_index()
    b_sample = b_sample.fillna(0)
    # remove unnecessary col in HH
    hh = hh[[col for col in hh.columns if col not in hh_filter_columns+["household_id"]]]
    # remove unnecessary col in buildings
    b_sample = b_sample[[col for col in b_sample.columns if col not in b_filter_columns]]

    X_df = pd.concat(
        [pd.concat([hh]*estimation_sample_size).reset_index(drop=True), b_sample], axis=1)
    # Y: 1 for the building picked
    # Y = X_df.building_id.isin(picked_bid).astype(int).values
    # Y: set first hh_sample_size item 1
    Y = np.zeros((hh_sample_size*estimation_sample_size,1), dtype=int)
    Y[:hh_sample_size,0] = 1
    # remove extra cols
    X_df = X_df[[col for col in X_df.columns if col not in ['building_id']]]
    orig_var_names = X_df.columns
    # create interaction variables
    newX_cols_name = vars_to_use
    X_wiv = np.array([])
    for varname in newX_cols_name:
        if X_wiv.size > 0:
            X_wiv = np.concatenate((X_wiv, get_interaction_vars(X_df, varname)), axis=1)
        else:
            X_wiv = get_interaction_vars(X_df, varname)
            
    # print("orig_var_names",orig_var_names)
    # df to ndarray
    X = X_wiv
    # standardize X
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
    # cache input tables
    cache_hdf = pd.HDFStore('lcm_testing.hdf')
    cache_hdf["theta"] = pd.DataFrame(theta)
    cache_hdf["X"] = pd.DataFrame(X)
    cache_hdf["Y"] = pd.DataFrame(Y)
    cache_hdf["Y_onehot"] = pd.DataFrame(Y_onehot) 
    cache_hdf["available_choice"] = pd.DataFrame(available_choice)
    cache_hdf.close()
else:
    cache_hdf = pd.HDFStore('lcm_testing.hdf', 'r')
    theta, X, Y, Y_onehot, available_choice = cache_hdf['theta'], cache_hdf['X'], cache_hdf['Y'], cache_hdf['Y_onehot'], cache_hdf['available_choice']
    theta, X, Y, Y_onehot, available_choice = theta.values, X.values, Y.values, Y_onehot.values, available_choice.values
    newX_cols_name = vars_to_use

# only keep variables with variation
used_val = np.arange(X.shape[1])[np.std(X, axis=0, dtype=np.float64) > 0]

# dtypes conversion
X = {0:X, 1:X}
theta = {0:theta, 1:theta}
Y = 1 - Y # 0 means picked, 1 means not picked
Y_onehot = np.concatenate((Y_onehot, 1-Y_onehot), axis=1)
available_choice = np.concatenate((available_choice, available_choice), axis=1)

t0 = time.time()
theta_optim_full = minimize(theta, neglog_DCM, -20000, X, Y, Y_onehot, available_choice)
t1 = time.time()
# print([orig_var_names[used_val][i] for i in theta_optim_full.argsort()[0][:20]])
# print(Y.argsort()[::-1][:20])
# print(X.dot(theta_optim_full[0]).reshape(-1).argsort()[::-1][:20])
# print([x for x in Y.reshape(-1).argsort()[::-1][:hh_sample_size] if x in X.dot(theta_optim_full[0]).reshape(-1).argsort()[::-1][:hh_sample_size]])

# exporting theta
out_theta = pd.DataFrame(theta_optim_full[0], columns=['theta'])
out_theta.index = newX_cols_name[used_val]
out_theta = out_theta.loc[out_theta.theta.abs().sort_values(ascending=False).index]
out_theta.to_csv('out_theta_%s.txt' % (estimation_sample_size))

print('done')
