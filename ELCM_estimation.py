#!/usr/bin/env python
"""
ELCM_estimation.py

This script performs estimation using the Automatic Relevance Determination (ARD-DCM)
for sector_id x large area within an urban simulation context. It handles data loading, variable
selection, estimation, and running Multinomial Logit (MNL) models. The script is designed for
flexible execution across multiple large areas.
"""
from urbansim.models.regression import RegressionModel

import os
import numpy as np
import pandas as pd
import time

from lcm_utils import *
from dcm_ard_libs import minimize, neglog_DCM
from fit_large_MNL_LCM import run_elcm_large_MNL
from urbansim_templates import modelmanager as mm
mm.initialize('configs/elcm_2050')

# from guppy import hpy; h=hpy()
# import pymrmr

# suppress sklearn warnings
def warn(*args, **kwargs):
    pass

os.chdir("/home/da/semcog_urbansim")

# config
data_path = r'/home/da/share/urbansim/RDF2050/model_inputs/base_hdf'
var_pool_table_path = "/home/da/share/urbansim/RDF2050/model_estimation/configs_elcm_large_area_sector.xlsx"
job_filter_columns = ["building_id", "slid", "home_based_status"]
b_filter_columns = ["large_area_id", "non_residential_sqft", "vacant_job_spaces"]

# Estimation Config
JOBS_SAMPLE_SIZE = 10000
ESTIMATION_SAMPLE_SIZE = 50
TOP_VARIABLES_TO_USE = 20

def estimation(SLID, job_region, b_region, vars_to_use):
    """
    Perform estimation using the ARD-DCM model for employment location choice.

    This function performs estimation using the ARD-DCM (Automatic Relevance Determination) for employment
    location choice for a given SLID, using the provided job and building data along with
    a list of variables to use.

    Parameters:
    SLID (int): Jobs SLID.
    job_region (pd.DataFrame): Job data DataFrame.
    b_region (pd.DataFrame): Building data DataFrame.
    vars_to_use (np.ndarray): Array of variable names

    Returns:
    None
    """
    # sampling jobs
    # from the new move-ins, last 5-10 years
    # weighted by mcd_quota
    job = job_region[job_region.slid == SLID]
    job = job[job.building_id > 1]
    job = job[job.home_based_status == 0]

    # if total number of job is less than job_sample_size 
    job_sample_size = min(JOBS_SAMPLE_SIZE, job.shape[0])

    job = job.sample(job_sample_size)
    job = job.reset_index()
    job = job.fillna(0)

    # sample buildings from the chosen job's buildings list
    bid_sample_pool = b_region[b_region.large_area_id == SLID % 1000].index
    sampled_b_id = []
    for _ in range(ESTIMATION_SAMPLE_SIZE-1):
        for j in job.building_id:
            sampled_b_id.append(np.random.choice(bid_sample_pool[bid_sample_pool!=j]))

    b_sample = b_region.loc[sampled_b_id]
    b_sample = pd.concat([b_region.loc[job.building_id], b_sample])
    b_sample = b_sample.reset_index()
    b_sample = b_sample.fillna(0)

    # remove unnecessary col in jobs
    job = job[[col for col in job.columns if col not in job_filter_columns+["job_id"]]]

    # remove unnecessary col in buildings
    b_sample = b_sample[[col for col in b_sample.columns if col not in b_filter_columns]]

    X_df = pd.concat(
        [pd.concat([job]*ESTIMATION_SAMPLE_SIZE).reset_index(drop=True), b_sample], axis=1)
    # Y: 1 for the building picked
    # Y: set first job_sample_size item 1
    Y = np.zeros((job_sample_size*ESTIMATION_SAMPLE_SIZE,1), dtype=int)
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
    theta_optim_full = minimize(theta, neglog_DCM, -3000, X, Y, Y_onehot, available_choice)
    t1 = time.time()
    print("minimizer finished in ", t1-t0)

    # exporting theta
    out_theta = pd.DataFrame(theta_optim_full[0], columns=['theta'])
    out_theta.index = newX_cols_name[used_val]
    out_theta = out_theta.loc[out_theta.theta.abs().sort_values(ascending=False).index]
    out_theta.to_csv('./configs/elcm_2050/thetas/out_theta_job_%s_%s.txt' % (SLID, ESTIMATION_SAMPLE_SIZE))

    print("Warning: variables with 0 variation")
    print(newX_cols_name[unused_val.tolist()])
    print('ARD-DCM done')

if __name__ == "__main__":
    valid_job_vars, valid_b_vars = get_elcm_valid_vars(data_path)
    job_region, b_region, vars_to_use = load_elcm_dataset(valid_job_vars, valid_b_vars, 
                        var_pool_table_path, job_filter_columns, b_filter_columns, use_cache=True)

    # get slid unique list
    slid_list = job_region['slid'].unique().tolist()

    # loop through slid
    for slid in slid_list[:5]:
        # if selected sector_id, skip it and use job scaling model instead
        sector_id = slid // 100000
        if sector_id in [1, 7, 12, 13, 15, 18]:
            continue

        # skip slid which have very small sample size
        if slid in [1100115, 1100147]:
            continue

        estimation(slid, job_region, b_region, vars_to_use)
        run_elcm_large_MNL(job_region, b_region, slid, TOP_VARIABLES_TO_USE)
    # estimation(500125)
    # run_elcm_large_MNL(job_region, b_region, 500125, 30)
    # slid which have failed LargeMNL run due to LinAlgError:
    # [500115, 500093, 1100093, 1500115]