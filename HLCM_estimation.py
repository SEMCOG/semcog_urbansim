#!/usr/bin/env python
"""
HLCM_estimation.py

This script performs estimation using the Automatic Relevance Determination (ARD-DCM)
for various large areas within an urban simulation context. It handles data loading, variable
selection, estimation, and running Multinomial Logit (MNL) models. The script is designed for
flexible execution across multiple large areas.
"""

import os
import numpy as np
import pandas as pd
import time
import time

from lcm_utils import *
from dcm_ard_libs import minimize, neglog_DCM
from fit_large_MNL_LCM import run_large_MNL
from urbansim_templates import modelmanager as mm
mm.initialize('configs/hlcm_2050')

# from guppy import hpy; h=hpy()
# import pymrmr

# suppress sklearn warnings
def warn(*args, **kwargs):
    pass

os.chdir("/home/da/semcog_urbansim")

# config
data_path = r'/home/da/share/urbansim/RDF2050/model_inputs/base_hdf'
var_pool_table_path = "/home/da/share/urbansim/RDF2050/model_estimation/configs_hlcm_2050_update3.xlsx"
choice_column = "building_id"
hh_filter_columns = ["building_id", "large_area_id", "mcd_model_quota", "year_built", "residential_units"]
b_filter_columns = ["large_area_id", "mcd_model_quota", "residential_units"]

# Estimation Config
HH_SAMPLE_SIZE = 10000
ESTIMATION_SAMPLE_SIZE = 50

def estimation(LARGE_AREA_ID, hh_region, b_region, vars_to_use):
    """
    Perform estimation using the ARD-DCM model.

    This function performs estimation using the ARD-DCM (Automatic Relevance Determination) for a given
    large area, using the provided household and building data along with a list of variables to use.

    Parameters:
    LARGE_AREA_ID (int): ID of the large area for estimation.
    hh_region (pd.DataFrame): Household data DataFrame.
    b_region (pd.DataFrame): Building data DataFrame.
    vars_to_use (np.ndarray): Array of variable names

    Returns:
    None
    """
    # sampling hh
    # from the new move-ins, last 5-10 years
    # weighted by mcd_quota
    hh = hh_region[hh_region.large_area_id == LARGE_AREA_ID]
    hh = hh[hh.building_id > 1]
    hh = hh[hh.residential_units > 0]
    hh = hh[hh.year_built > 2005]

    # exclude hh in pseudo buildings
    hh = hh[hh.building_id < 90000000]

    # add 1 to all hh's mcd_model_quota for weights
    hh["mcd_model_quota"] += 1 

    # if total number of hh is less than HH_SAMPLE_SIZE 
    hh_sample = min(HH_SAMPLE_SIZE, hh.shape[0])

    hh = hh.sample(hh_sample, weights="mcd_model_quota")
    hh = hh.reset_index()
    hh = hh.fillna(0)

    # sample buildings from the chosen HH's buildings list
    uhh_id = hh.building_id.unique()
    sampled_b_id = []
    for _ in range(ESTIMATION_SAMPLE_SIZE-1):
        for j in hh.building_id:
            sampled_b_id.append(np.random.choice(uhh_id[uhh_id!=j]))
    b_sample = b_region.loc[sampled_b_id]
    b_sample = pd.concat([b_region.loc[hh.building_id], b_sample])
    b_sample = b_sample.reset_index()
    b_sample = b_sample.fillna(0)

    # remove unnecessary col in HH
    hh = hh[[col for col in hh.columns if col not in hh_filter_columns+["household_id"] or col in ['year_built']]]

    # remove unnecessary col in buildings
    b_sample = b_sample[[col for col in b_sample.columns if col not in b_filter_columns]]

    X_df = pd.concat(
        [pd.concat([hh]*ESTIMATION_SAMPLE_SIZE).reset_index(drop=True), b_sample], axis=1)
    # Y: 1 for the building picked
    # Y = X_df.building_id.isin(picked_bid).astype(int).values
    # Y: set first hh_sample item 1
    Y = np.zeros((hh_sample*ESTIMATION_SAMPLE_SIZE,1), dtype=int)
    Y[:hh_sample,0] = 1
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

    # Optimize theta using ARD minimization
    t0 = time.time()
    theta_optim_full = minimize(theta, neglog_DCM, -10000, X, Y, Y_onehot, available_choice)
    t1 = time.time()
    print("minimizer finished in ", t1-t0)

    # exporting theta
    out_theta = pd.DataFrame(theta_optim_full[0], columns=['theta'])
    out_theta.index = newX_cols_name[used_val]
    out_theta = out_theta.loc[out_theta.theta.abs().sort_values(ascending=False).index]
    out_theta.to_csv('configs/hlcm_2050/thetas/out_theta_%s_%s.txt' % (LARGE_AREA_ID, ESTIMATION_SAMPLE_SIZE))

    # Print variables with zero variation
    print("Warning: variables with 0 variation")
    print(newX_cols_name[unused_val])
    print('ARD-DCM done')

if __name__ == "__main__":
    # Configuration for large area estimation
    la_estimation_configs = {
        3: {
            'skip_estimation': True,
            'number_of_var_to_use': 40
        },
        5: {
            'skip_estimation': True,
            'number_of_var_to_use': 40
        },
        93: {
            'skip_estimation': True,
            'number_of_var_to_use': 50
        },
        99: {
            'skip_estimation': True,
            'number_of_var_to_use': 40
        },
        115: {
            'skip_estimation': True,
            'number_of_var_to_use': 50
        },
        125: {
            'skip_estimation': True,
            'number_of_var_to_use': 40
        },
        147: {
            'skip_estimation': True,
            'number_of_var_to_use': 40
        },
        161: {
            'skip_estimation': True,
            'number_of_var_to_use': 50
        },
    }
    # Get valid household and building variables
    valid_hh_vars, valid_b_vars = get_hlcm_valid_vars(data_path)
    # Load household data, building data, and variables to use
    hh_region, b_region, vars_to_use = load_hlcm_dataset(valid_hh_vars, valid_b_vars, var_pool_table_path, 
                        hh_filter_columns, b_filter_columns, use_cache=True)
    # Loop through each large area and perform estimation and MNL model run
    for la_id, la_config in la_estimation_configs.items():
        if not la_config['skip_estimation']:
            # Perform estimation for the current large area
            estimation(la_id, hh_region, b_region, vars_to_use)
        # Run large MNL model for the current large area
        run_large_MNL(hh_region, b_region, la_id, la_config['number_of_var_to_use'])