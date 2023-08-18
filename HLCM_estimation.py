#!/usr/bin/env python
"""
HCLM_estimation.py

This script performs estimation using the AutoRegressive Distributed Choice Model (ARD-DCM)
for various large areas within an urban simulation context. It handles data loading, variable
selection, estimation, and running Multinomial Logit (MNL) models. The script is designed for
flexible execution across multiple large areas.
"""

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

def get_valid_vars(data_path: str) -> tuple[list[str], list[str]]:
    """
    Extract valid household and building variable names from a YAML configuration file.

    Parameters:
    data_path (str): Path to the directory containing the variable validation YAML files.

    Returns:
    tuple: A tuple containing two lists of valid variable names: valid household variable names and valid building variable names.
    """
    var_validation_list = [
        os.path.join(data_path, f)
        for f in os.listdir(data_path)
        if ("variable_validation" in f) and (f[-5:] == ".yaml")
    ]
    var_validation_last = max(var_validation_list, key=os.path.getctime)
    
    with open(var_validation_last, "r") as f:
        vars_config = yaml.load(f, Loader=yaml.FullLoader)
    
    valid_b_vars = vars_config["buildings"]["valid variables"]
    valid_hh_vars = vars_config["households"]["valid variables"]
    return valid_hh_vars, valid_b_vars

def load_hlcm_df(households: orca.DataFrameWrapper, buildings: orca.DataFrameWrapper, hh_var: list[str], b_var: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and return household and building DataFrames.

    Parameters:
    households (orca.DataFrameWrapper): Orca DataFrameWrapper for households.
    buildings (orca.DataFrameWrapper): Orca DataFrameWrapper for buildings.
    hh_var (list[str]): Names of the household variables to load.
    b_var (list[str]): Names of the building variables to load.

    Returns:
    tuple: A tuple containing two DataFrames: the household DataFrame and the building DataFrame.

    Example:
    >>> household_df, building_df = load_hlcm_df(households, buildings, ['persons'], ['parcel_id'])
    """
    hh = households.to_frame(hh_var)
    b = buildings.to_frame(b_var)
    return hh, b

def columns_in_vars(vars: list[str], valid_hh_vars: list[str], valid_b_vars: list[str]) -> tuple[list[str], list[str]]:
    """
    Categorize variables into household and building columns.

    This function takes a list of variable names and categorizes them into
    household columns and building columns based on the presence of a colon
    separator or matching valid variable names.

    Parameters:
    vars (list[str]): List of variable names to categorize.
    valid_hh_vars (list[str]): List of valid household variable names
    valid_b_vars (list[str]): List of valid building variable names

    Returns:
    tuple: A tuple containing two lists of strings: household column names and building column names.

    Example:
    >>> hh_vars, b_vars = columns_in_vars(['persons', 'parcel_id', 'income:improvement_value'])
    >>> print(hh_vars)
    ['persons', 'income']
    >>> print(b_vars)
    ['parcel_id']
    """
    hh_columns, b_columns = [], []
    for varname in vars:
        if ':' in varname:
            hh_col, b_col = map(str.strip, varname.split(':'))
            if hh_col in valid_hh_vars:
                hh_columns.append(hh_col)
            if b_col in valid_b_vars:
                b_columns.append(b_col)
        elif varname in valid_hh_vars:
            hh_columns.append(varname)
        elif varname in valid_b_vars:
            b_columns.append(varname)
        else:
            print(varname, " not found in both hh and buildings table")
    return hh_columns, b_columns

def get_interaction_vars(df: pd.DataFrame, varname: str) -> np.ndarray:
    """
    Get interaction variables from variable name.

    This function calculates interaction variables based on the provided variable name
    within the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the variables.
        varname (str): The name of the interaction variable.

    Returns:
        np.ndarray: A NumPy array containing the calculated interaction variables.

    Example:
    >>> data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    >>> df = pd.DataFrame(data)
    >>> interaction_array = get_interaction_vars(df, 'A:B')
    """
    if ":" in varname:
        var1, var2 = map(str.strip, varname.split(":"))
        return (df[var1] * df[var2]).values.reshape(-1, 1)
    else:
        return df[varname].values.reshape(-1, 1)

def load_dataset(valid_hh_vars, valid_b_vars):
    """
    Load and preprocess dataset variables

    This function loads and preprocesses the dataset variables needed for estimation. It extracts the set of valid variables
    from the variable pool table, loads the necessary variables from the 'buildings' and 'households' Orca tables,
    and caches the resulting household and building DataFrames for later use.

    Parameters:
    valid_hh_vars (list): A list of valid household variable names.
    valid_b_vars (list): A list of valid building variable names.

    Returns:
    tuple: A tuple containing the following elements:
        - hh_region (pd.DataFrame): A DataFrame containing loaded household data.
        - b_region (pd.DataFrame): A DataFrame containing loaded building data.
        - vars_to_use (np.ndarray): An array of variable names used for modeling.
    """
    # Load the variable pool table and extract valid variable names
    used_vars = pd.read_excel(var_pool_table_path, sheet_name=2)
    v1 = used_vars[~used_vars["new variables 1"].isna()]["new variables 1"].unique()
    v2 = used_vars[~used_vars["new variables 2"].isna()]["new variables 2"].unique()
    vars_to_use = np.array(list(set(v1.tolist()).union(v2.tolist())))

    # Choose whether to reload data or use cached data
    RELOAD = False
    if RELOAD:
        # from notebooks.models_test import *
        import models
        buildings = orca.get_table("buildings")
        households = orca.get_table("households")

        # set year to 2020 and run build network and neigh vars
        orca.add_injectable('year', 2020)
        orca.run(["build_networks_2050"])
        orca.run(["neighborhood_vars"])

        # set year to 2050 and run mcd_hu_sampling
        orca.add_injectable('year', 2050)
        orca.run(["mcd_hu_sampling"])

        # Get valid variables for modeling and load corresponding data
        hh_columns, b_columns = columns_in_vars(vars_to_use, valid_hh_vars, valid_b_vars)
        hh_var = hh_columns + hh_filter_columns
        b_var = b_columns + b_filter_columns
        hh_region, b_region = load_hlcm_df(households, buildings, hh_var, b_var)

        # Cache the loaded DataFrames as CSV files
        hh_region.to_csv('hh.csv')
        b_region.to_csv('b_hlcm.csv')
    else:
        hh_region, b_region = load_cache_hh_b('hh.csv', 'b_hlcm.csv')
    return hh_region, b_region, vars_to_use

def load_cache_hh_b(hh_csv_path: str, b_csv_path: str):
    """
    Load household and building data from CSV files and register them as tables.

    Parameters:
    hh_csv_path (str): Path to the household CSV file.
    b_csv_path (str): Path to the building CSV file.
    """
    try:
        hh_region = pd.read_csv(hh_csv_path, index_col=0)
        b_region = pd.read_csv(b_csv_path, index_col=0)
    except FileNotFoundError:
        print("CSV file not found. Please provide correct file paths.")
        return 
    
    orca.add_table('households', hh_region)
    orca.add_table('buildings', b_region)
    return hh_region, b_region

def estimation(LARGE_AREA_ID, hh_region, b_region, vars_to_use):
    """
    Perform estimation using the ARD-DCM model.

    This function performs estimation using the ARD-DCM (AutoRegressive Distributed Choice Model) for a given
    large area, using the provided household and building data along with a list of variables to use.

    Parameters:
    LARGE_AREA_ID (int): ID of the large area for estimation.
    hh_region (pd.DataFrame): Household data DataFrame.
    b_region (pd.DataFrame): Building data DataFrame.
    vars_to_use (np.ndarray): Array of variable names for modeling.

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
    valid_hh_vars, valid_b_vars = get_valid_vars(data_path)
    # Load household data, building data, and variables to use
    hh_region, b_region, vars_to_use = load_dataset(valid_hh_vars, valid_b_vars)
    # Loop through each large area and perform estimation and MNL model run
    for la_id, la_config in la_estimation_configs.items():
        if not la_config['skip_estimation']:
            # Perform estimation for the current large area
            estimation(la_id, hh_region, b_region, vars_to_use)
        # Run large MNL model for the current large area
        run_large_MNL(hh_region, b_region, la_id, la_config['number_of_var_to_use'])