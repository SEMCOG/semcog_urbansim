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
import yaml

from dcm_ard_libs import minimize, neglog_DCM
from test_large_MNL_HLCM import run_elcm_large_MNL, run_large_MNL

# from guppy import hpy; h=hpy()
# import pymrmr

# suppress sklearn warnings
def warn(*args, **kwargs):
    pass

os.chdir("/home/da/semcog_urbansim")

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
valid_job_vars = vars_config["jobs"]["valid variables"]


def apply_filter_query(df, filters=None):
    if filters:
        if isinstance(filters, str):
            query = filters
        else:
            query = " and ".join(filters)
        return df.query(query)
    else:
        return df


def load_hlcm_df(jobs, buildings, job_var, b_var):
    jobs = jobs.to_frame(job_var)
    b = buildings.to_frame(b_var)
    return jobs, b

def columns_in_vars(jobs, buildings, vars):
    job_columns, b_columns = [], []
    for varname in vars:
        if varname in jobs.columns:
            job_columns.append(varname.strip())
        elif varname in buildings.columns:
            b_columns.append(varname.strip())
        else:
            print(varname, " not found in both jobs and buildings table")
    return job_columns, b_columns
            

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


used_vars = pd.read_excel("/home/da/share/urbansim/RDF2050/model_estimation/configs_elcm_large_area_sector.xlsx", sheet_name=1)
v1 = used_vars[~used_vars["variables 1"].isna()]["variables 1"].unique()
v2 = used_vars[~used_vars["Variables 2"].isna()]["Variables 2"].unique()
vars_to_use = np.array(list(set(v1.tolist()).union(v2.tolist())))
# vars_to_use = used_vars[0].unique()

# config
choice_column = "building_id"
job_filter_columns = ["building_id", "slid", "home_based_status"]
b_filter_columns = ["large_area_id", "non_residential_sqft"]
# load variables
RELOAD = False
if RELOAD:
    # from notebooks.models_test import *
    import models
    buildings = orca.get_table("buildings")
    jobs = orca.get_table("jobs")
    orca.add_injectable('year', 2020)
    orca.run(["build_networks_2050"])
    orca.run(["neighborhood_vars"])
    # TODO: get vars from vars list from last forecast
    job_columns, b_columns = columns_in_vars(jobs, buildings, vars_to_use)

    job_var = job_columns + job_filter_columns
    b_var = b_columns + b_filter_columns
    job_region, b_region = load_hlcm_df(jobs, buildings, job_var, b_var)
    job_region.to_csv('jobs.csv')
    b_region.to_csv('b_elcm.csv')
else:
    job_region = pd.read_csv('jobs.csv', index_col=0)
    b_region = pd.read_csv('b_elcm.csv', index_col=0)

def estimation(SLID):
    job_sample_size = 5000
    estimation_sample_size = 50
    # sampling jobs
    # from the new move-ins, last 5-10 years
    # weighted by mcd_quota
    job = job_region[job_region.slid == SLID]
    job = job[job.building_id > 1]
    job = job[job.home_based_status == 0]
    # if total number of job is less than job_sample_size 
    job_sample_size = min(job_sample_size, job.shape[0])
    job = job.sample(job_sample_size)
    job = job.reset_index()
    job = job.fillna(0)
    # sampling b
    # sample buildings from the chosen job's buildings list
    ujob_bid = job.building_id.unique()
    sampled_b_id = []
    for _ in range(estimation_sample_size-1):
        for j in job.building_id:
            sampled_b_id.append(np.random.choice(ujob_bid[ujob_bid!=j]))

    b_sample = b_region.loc[sampled_b_id]
    b_sample = pd.concat([b_region.loc[job.building_id], b_sample])
    b_sample = b_sample.reset_index()
    b_sample = b_sample.fillna(0)
    # remove unnecessary col in jobs
    job = job[[col for col in job.columns if col not in job_filter_columns+["job_id"]]]
    # remove unnecessary col in buildings
    b_sample = b_sample[[col for col in b_sample.columns if col not in b_filter_columns]]

    X_df = pd.concat(
        [pd.concat([job]*estimation_sample_size).reset_index(drop=True), b_sample], axis=1)
    # Y: 1 for the building picked
    # Y = X_df.building_id.isin(picked_bid).astype(int).values
    # Y: set first job_sample_size item 1
    Y = np.zeros((job_sample_size*estimation_sample_size,1), dtype=int)
    Y[:job_sample_size,0] = 1
    # remove extra cols
    X_df = X_df[[col for col in X_df.columns if col not in ['building_id']]]
    # create interaction variables
    newX_cols_name = vars_to_use
    X_wiv = np.array([])
    for varname in newX_cols_name:
        if X_wiv.size > 0:
            X_wiv = np.concatenate((X_wiv, get_interaction_vars(X_df, varname)), axis=1)
        else:
            X_wiv = get_interaction_vars(X_df, varname)

    # df to ndarray
    X = X_wiv

    # col index with 0 variation
    used_val = np.arange(X.shape[1])[np.std(X, axis=0, dtype=np.float64) > 0]
    unused_val = np.array([x for x in range(X.shape[1]) if x not in used_val])

    # only keep variables with variation
    X = X[:, np.std(X, axis=0, dtype=np.float64) > 0]
    # standardize X
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

    # dtypes conversion
    X = {0:X, 1:X}
    theta = {0:theta, 1:theta}
    Y = 1 - Y # 0 means picked, 1 means not picked
    Y_onehot = np.concatenate((Y_onehot, 1-Y_onehot), axis=1)
    available_choice = np.concatenate((available_choice, available_choice), axis=1)

    t0 = time.time()
    theta_optim_full = minimize(theta, neglog_DCM, -10000, X, Y, Y_onehot, available_choice)
    t1 = time.time()
    print("minimizer finished in ", t1-t0)

    # exporting theta
    out_theta = pd.DataFrame(theta_optim_full[0], columns=['theta'])
    out_theta.index = newX_cols_name[used_val]
    out_theta = out_theta.loc[out_theta.theta.abs().sort_values(ascending=False).index]
    out_theta.to_csv('out_theta_job_%s_%s.txt' % (SLID, estimation_sample_size))

    print("Warning: variables with 0 variation")
    print(newX_cols_name[unused_val])
    print('ARD-DCM done')

if __name__ == "__main__":
    # slid_list = jobs['slid'].unique()
    # for slid in slid_list:
    #     estimation(slid)
    #     run_elcm_large_MNL(job_region, b_region, slid, 40)
    estimation(500125)
    run_elcm_large_MNL(job_region, b_region, 500125, 40)