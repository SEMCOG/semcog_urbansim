#!/usr/bin/env python
"""
REPM Feature Selection

This script automates the feature selection process for the Real Estate Price Model (REPM). It loads building
and variable data, generates REPM configuration files, performs Lasso regression-based feature selection,
and updates the configuration files with selected coefficients and model performance metrics.

"""

import os
import numpy as np
import pandas as pd
import orca
from urbansim.utils import misc
import scipy
import time
import yaml
from tqdm import tqdm

from utils import apply_filter_query

# suppress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings

warnings.warn = warn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso

REPM_PATH = "./configs/repm_2050/"
VARS_TO_SKIP = [
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
]


def generate_repm_config(mat:scipy.sparse.csr_matrix, vars_used:[str]):
    """
    Generate configuration files for Real Estate Price Models (REPM) based on the input matrix and variable list.

    This function generates configuration files for Real Estate Price Models (REPM) using a sparse matrix and a list
    of variables. The configuration files specify different settings for each model based on the variable names and
    model types.

    Parameters:
    mat (scipy.sparse.csr_matrix): A sparse matrix containing the data used for generating REPM configurations.
    vars_used (List[str]): A list of variable names used for generating the configurations.

    Returns:
    None
    """
    filter_cols = [
        "sqft_price_nonres",
        "sqft_price_res",
        "non_residential_sqft",
        "hedonic_id",
        "residential_units",
    ]
    filter_col_ind = [vars_used.index(name) for name in filter_cols]
    df = pd.DataFrame(
        np.vstack([mat.getrow(i).todense() for i in filter_col_ind]).transpose(),
        columns=filter_cols,
    )
    df = df[~df["hedonic_id"].isna()]
    df["hedonic_id"] = df["hedonic_id"].astype("int")
    hid_count = df.hedonic_id.value_counts()
    for hid in hid_count.index:
        config = {
            "name": "RegressionModel",
            "model_type": "regression",
            "fit_filters": [],
            "predict_filters": "hedonic_id == " + str(hid),
            "ytransform": "np.exp",
            "target_variable": "",
            # "model_expression": {"left_side": ""},
        }
        if hid % 100 in [81, 82, 83, 84]:
            prefix = "res_repm"
            price_col = "sqft_price_res"
            size_col = "residential_units"
        else:
            prefix = "nonres_repm"
            price_col = "sqft_price_nonres"
            size_col = "non_residential_sqft"
        config["fit_filters"].append(size_col + " > 0")
        config["fit_filters"].append(price_col + " > 1")
        config["fit_filters"].append(price_col + " < 650")
        config["fit_filters"].append("hedonic_id == " + str(hid))
        config["target_variable"] = "np.log1p(%s)" % price_col
        config_name = prefix + str(hid)
        with open("configs/repm_2050/%s.yaml" % config_name, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return


def get_training_data(mat:scipy.sparse.csr_matrix, yaml_config:str, vars_used:[str]):
    """
    Extract training data for the Real Estate Price Model (REPM) from the input matrix and configuration.

    This function extracts training data for the Real Estate Price Model (REPM) based on the provided sparse matrix,
    configuration YAML file, and a list of variable names. The function applies filters, performs target variable
    transformations, and returns the training data ready for model training.

    Parameters:
    mat (scipy.sparse.csr_matrix): A sparse matrix containing the data used for training.
    yaml_config (str): Path to the YAML configuration file for the REPM.
    vars_used (List[str]): A list of variable names used for training.

    Returns:
    tuple: A tuple containing the following elements:
        - np.ndarray: X - Filtered and transformed feature matrix for training.
        - np.ndarray: Y - Transformed target variable for training.
        - List[str]: List of variable names used in the training.
        - int: Number of samples in the training data.
    """
    filter_cols = [
        "sqft_price_nonres",
        "sqft_price_res",
        "non_residential_sqft",
        "hedonic_id",
        "residential_units",
    ]
    filter_col_ind = [vars_used.index(name) for name in filter_cols]
    df = pd.DataFrame(
        np.vstack([mat.getrow(i).todense() for i in filter_col_ind]).transpose(),
        columns=filter_cols,
    )
    with open(yaml_config, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    fit_filters = conf["fit_filters"]
    target_var = conf["target_variable"]
    target_var_raw = target_var[target_var.find("(") + 1 : target_var.find(")")]
    filtered_df = apply_filter_query(df, fit_filters)
    if filtered_df.size == 0:
        print("Config %s has 0 size dataframe after filtering" % (yaml_config))
        return (
            [0 for _ in range(len(vars_used))],
            [0 for _ in range(len(vars_used))],
            None,
            filtered_df.size,
        )
    else:
        print("Config %s has sample size: %s" % (yaml_config, filtered_df.size))
    filtered_ind = filtered_df.index
    # get filtered rows, if load all variables, sys will stalled
    cols_ind_to_fit = [i for i in range(mat.shape[0]) if i not in filter_col_ind]
    cols_names_to_fit = [vars_used[i] for i in cols_ind_to_fit]
    all_vars_filtered = mat.toarray()[:, filtered_ind].transpose()
    all_vars_filtered = all_vars_filtered[:, cols_ind_to_fit]
    y = filtered_df.loc[:, target_var_raw].values
    # target transformation
    y = np.log1p(y)
    return (
        np.nan_to_num(all_vars_filtered, copy=False),
        y,
        cols_names_to_fit,
        all_vars_filtered.shape[0],
    )


def feature_selection_lasso(mat: scipy.sparse.csr_matrix, yaml_config: str, vars_used: [str]):
    """
    Perform feature selection using Lasso regression.

    This function performs feature selection using Lasso regression based on the provided sparse matrix,
    configuration YAML file, and a list of variable names. It adjusts the regularization strength based on
    the sample size and returns selected feature names, coefficients, model score, and sample size.

    Parameters:
    mat (scipy.sparse.csr_matrix): A sparse matrix containing the data used for training.
    yaml_config (str): Path to the YAML configuration file for the REPM.
    vars_used (List[str]): A list of variable names used for training.

    Returns:
    tuple: A tuple containing the following elements:
        - Optional[List[str]]: Selected feature names after Lasso feature selection.
        - Optional[List[float]]: Coefficients corresponding to the selected features.
        - Optional[float]: Model score after Lasso feature selection.
        - int: Number of samples in the training data.
    """
    # get training data
    all_vars_filtered, y, cols_names_to_fit, sample_size = get_training_data(
        mat, yaml_config, vars_used
    )

    # return if sample size is 0
    if sample_size == 0:
        return None, None, None, sample_size

    # adjusting alpha based on sample_size
    if sample_size <= 50:
        alpha = 2
    elif sample_size <= 100:
        alpha = 0.8
    else:
        alpha = 0.3

    # sklearn model pipeline
    pipeline = Pipeline(
        [
            # ('scaler',StandardScaler()), # disable standardize
            # alpha control the strength of regularization,
            # higher the alpha, stronger the regularization, more vars get 0 coef
            ("model", Lasso(alpha=alpha, fit_intercept=True))
        ]
    )

    # Grid search step, need manually enable
    # search = GridSearchCV(pipeline,
    #                   {'model__alpha':np.arange(0.1,10,0.1)},
    #                   cv = 5, scoring="neg_mean_squared_error",verbose=3
    #                 )

    # fit model
    pipeline.fit(all_vars_filtered, y)

    # get col names with coef > 0
    cols_names = [
        cols_names_to_fit[i]
        for i in np.arange(all_vars_filtered.shape[1])[
            np.abs(pipeline._final_estimator.coef_) > 0
        ]
    ]

    # get coef and intercept
    coef = list(
        pipeline._final_estimator.coef_[np.abs(pipeline._final_estimator.coef_) > 0]
    )
    intercept = pipeline._final_estimator.intercept_
    cols_names.insert(0, "Intercept")
    coef.insert(0, intercept)

    # get score
    score = pipeline.score(all_vars_filtered, y)

    print(score)
    print(cols_names)
    print(coef)
    print(intercept)
    return cols_names, coef, score, sample_size


def load_variables(buildings:orca.DataFrameWrapper, valid_b_vars:[str]):
    """
    Load and preprocess variables from buildings data.

    This function loads and preprocesses variables from the buildings data based on the provided list of valid
    building variable names. It filters out specific variables, standardizes them, and converts them into a sparse matrix.

    Parameters:
    buildings (orca.DataFrameWrapper): The buildings data table.
    valid_b_vars (List[str]): A list of valid building variable names.

    Returns:
    tuple: A tuple containing the following elements:
        - List[str]: List of valid building variable names after preprocessing.
        - scipy.sparse.csr_matrix: A sparse matrix containing the preprocessed variable data.
    """
    # init used variables
    vars_used = []
    # this loop will stall the system if it loads all 1200 variables
    mat_list = [] # matrix

    # log start timestamp
    t0 = time.time()

    # loop through valid building variables
    for i, var in enumerate(tqdm(valid_b_vars[:])):
        # skip variables in the list or if it's standardized variable
        if var in VARS_TO_SKIP or "st_" in var or var not in buildings.columns:
            continue

        # load var from buildings df as Series
        s = buildings.to_frame(var).iloc[:, 0]

        # fill inf with 0
        s.replace(np.inf, 0, inplace=True)
        s.replace(-np.inf, 0, inplace=True)

        # fill nan with 0
        s.fillna(0, inplace=True)

        # add to matrix if 
        if pd.api.types.is_numeric_dtype(s):
            vars_used.append(var)
            sparse_mat = scipy.sparse.csr_matrix(s.values)
            mat_list.append(sparse_mat)

        # clear cache every 50 vars
        if i % 50 == 0:
            orca.clear_columns("buildings")

    # clear all orca cache to reduce memory usage
    orca.clear_all()
    
    # to sparse matrix
    mat = scipy.sparse.vstack(mat_list)

    # record runtime
    t1 = time.time()
    # 2692.167008 MB RAM usage with 500 variables loaded
    print("load_variables finsihed in ", t1 - t0)
    return vars_used, mat


def run_repm():
    """
    Run the Real Estate Price Model (REPM) pipeline.

    This function executes the entire pipeline for the Real Estate Price Model (REPM). It loads necessary
    data, generates configuration files, performs feature selection using Lasso regression, and writes the
    results to the configuration files.

    Note: Make sure to customize constants like VARS_TO_SKIP, REPM_PATH

    Returns:
    None
    """
    # load model and steps
    import models
    orca.run(["build_networks"])
    orca.run(["neighborhood_vars"])

    buildings = orca.get_table("buildings")
    valid_b_vars = buildings.columns

    for var in valid_b_vars:
        if "parcels_" + var in valid_b_vars:
            VARS_TO_SKIP.append("parcels_" + var)

    # load variables
    # nodes_walk_vacancy_750m contains odd values
    vars_used, mat = load_variables(buildings, valid_b_vars)
    # cannot use orca from now on

    # remove old REPM configs
    for f in os.listdir(REPM_PATH):
        os.remove(os.path.join(REPM_PATH, f))

    # generate new REPM configs
    generate_repm_config(mat, vars_used)

    # ## REPM Configs
    repm_configs = []
    for filename in os.listdir(REPM_PATH):
        if not filename.endswith(".yaml"):
            continue
        repm_configs.append(os.path.join(REPM_PATH, filename))
    repm_configs.sort()

    result = {}
    for config in repm_configs:
        col_names, coef, score, sample_size = feature_selection_lasso(
            mat, config, vars_used
        )
        if sample_size == 0:
            result[config] = {"fit_parameters": {}, "fit_rsquared": 0.0, "sample_size": sample_size}
        else:
            result[config] = {
                "fitted": True,
                "fit_parameters": {"Coefficient": {col_names[i]: float(coef[i]) for i in range(len(col_names))}, "Std. Error": {}, "T-Score": {}},
                "fit_rsquared": float(score),
                "fit_rsquared_adj": float(score),
                "sample_size": sample_size,
            }
        result[config]["model_expression"] = {
            "left_side": "np.log1p(sqft_price_%s)" % ("nonres" if "nonres" in config else "res"),
            "right_side": [col for col in col_names if col != "Intercept"]
        }
        if type(result[config]["sample_size"]) == np.int64:
            result[config]["sample_size"] = result[config]["sample_size"].item()
        # if result[config]['coef'] == 'None': continue
        for coe in result[config]["fit_parameters"]["Coefficient"]:
            result[config]["fit_parameters"]["Coefficient"][coe] = float(
                result[config]["fit_parameters"]["Coefficient"][coe])
        with open(config, "a") as f:
            yaml.dump(result[config], f, default_flow_style=False, sort_keys=False)
    print("done")


if __name__ == "__main__":
    # run repm
    run_repm()