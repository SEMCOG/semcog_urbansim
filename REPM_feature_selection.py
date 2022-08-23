#!/usr/bin/env python
# coding: utf-8
# feature selection
# 1. MRMR
# 2. ANOVA F-test - applicable for both numeric and categorical
# 3. use old estimation result as guide
# Performance optimization 
# - sample buildings by type and then fit
# - sample vars 
from urbansim.models.regression import RegressionModel

import os
import numpy as np
import pandas as pd
import orca
from urbansim.utils import misc
import sys
import scipy
import time
from tqdm import tqdm
# from guppy import hpy; h=hpy()
import pymrmr

# suppress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso

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
print("HDF data: ", hdf_last)

var_validation_list = [
    (data_path + "/" + f)
    for f in os.listdir(data_path)
    if ("variable_validation" in f) & (f[-5:] == ".yaml")
]
var_validation_last = max(var_validation_list, key=os.path.getctime)
with open(var_validation_last, 'r') as f:
    vars_config = yaml.load(f, Loader=yaml.FullLoader)
valid_b_vars = vars_config['buildings']['valid variables']


orca.add_injectable("store", hdf)
load_tables_to_store()
from notebooks.models_test import *
import variables

orca.run(["build_networks"])
orca.run(["neighborhood_vars"])

vars_to_skip = [
    'parcel_id', 'st_parcel_id', 'geoid', 'st_geoid', 'b_ln_parcels_parcel_far', 'parcels_parcel_far',
    'parcels_st_parcel_far',
    'nodeid_drv', 'st_nodeid_drv', 'nodeid_walk', 'st_nodeid_walk', 'semmcd', 
    'city_id', 'census_bg_id', 'x', 'y',
    'zone_id', 'improvement_value', 'market_value', 'landvalue', 'b_ln_sqft_price_nonres',
    'zones_tazce10_n', 'tazce10', 'parcels_zones_tazce10_n', 'parcels_st_zones_tazce10_n',
    'st_parcels_zones_tazce10_n', 'st_parcels_st_zones_tazce10_n', 'st_zones_tazce10_n',
    'parcels_land_use_type_id', 'st_parcels_land_use_type_id', 'st_parcels_max_dua', 'parcels_max_dua',
    'parcels_max_height', 'st_parcels_max_height', 'parcels_school_id', 'st_parcels_school_id', 
    'parcels_sev_value', 'st_parcels_sev_value', 'parcels_centroid_x', 'st_parcels_centroid_x',
    'parcels_centroid_y', 'st_parcels_centroid_y', 'parcels_school_id', 'st_parcels_school_id',
    'parcels_land_cost', 'st_parcels_land_cost', 'b_ln_market_value', 'st_b_ln_market_value'
]

def estimate_repm(yaml_config):
    model = RegressionModel.from_yaml(str_or_buffer=misc.config(yaml_config))
    b = orca.get_table("buildings").to_frame(model.columns_used()).fillna(0)
    print(model.fit(b))
    model.to_yaml(str_or_buffer=misc.config(yaml_config))
    return model.fit_parameters

def apply_filter_query(df, filters=None):
    if filters:
        if isinstance(filters, str):
            query = filters
        else:
            query = ' and '.join(filters)
        return df.query(query)
    else:
        return df

def get_training_data(mat, yaml_config, vars_used):
    filter_cols = ['sqft_price_nonres', 'sqft_price_res', 'st_sqft_price_res', 'st_sqft_price_nonres', 'non_residential_sqft',  'hedonic_id', 'residential_units']
    filter_col_ind = [vars_used.index(name) for name in filter_cols]
    df = pd.DataFrame(np.vstack([mat.getrow(i).todense() for i in filter_col_ind]).transpose(), columns=filter_cols)
    with open( yaml_config, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    fit_filters = conf['fit_filters']
    target_var = conf['model_expression']['left_side']
    target_var_raw = target_var[target_var.find("(")+1:target_var.find(")")]
    filtered_df = apply_filter_query(df, fit_filters)
    if filtered_df.size == 0: 
        print("Config %s has 0 size dataframe after filtering" % (yaml_config))
        return [0 for _ in range(len(vars_used))], [0 for _ in range(len(vars_used))], None, filtered_df.size
    else:
        print("Config %s has sample size: %s" % (yaml_config, filtered_df.size))
    filtered_ind = filtered_df.index
    # get filtered rows, if load all variables, sys will stalled 
    # all_vars_filtered = pd.DataFrame(np.hstack([mat.getcol(i).todense() for i in filtered_ind]).transpose(), columns=vars_used)
    cols_ind_to_fit = [i for i in range(mat.shape[0]) if i not in filter_col_ind]
    cols_names_to_fit = [vars_used[i] for i in cols_ind_to_fit]
    all_vars_filtered = mat.toarray()[:, filtered_ind].transpose()
    all_vars_filtered = all_vars_filtered[:, cols_ind_to_fit]
    y = filtered_df.loc[:, target_var_raw].values
    return np.nan_to_num(all_vars_filtered, copy=False), y, cols_names_to_fit, all_vars_filtered.shape[0]

def feature_selection_f_classif(mat, yaml_config, vars_used):
    all_vars_filtered, y, cols_names_to_fit, sample_size = get_training_data(mat, yaml_config, vars_used)
    f, p = f_classif(all_vars_filtered, y)
    return f, p

def feature_selection_lasso(mat, yaml_config, vars_used):
    all_vars_filtered, y, cols_names_to_fit, sample_size = get_training_data(mat, yaml_config, vars_used)
    if sample_size == 0:
        return None, None, None, sample_size
    pipeline = Pipeline([
                        # ('scaler',StandardScaler()), # disable standardize
                        # alpha control the strength of regularization, 
                        # higher the alpha, stronger the regularization, more vars get 0 coef
                        ('model',Lasso(alpha=0.4))
    ])
    # search = GridSearchCV(pipeline,
    #                   {'model__alpha':np.arange(0.1,10,0.1)},
    #                   cv = 5, scoring="neg_mean_squared_error",verbose=3
    #                 )
    pipeline.fit(all_vars_filtered, y)
    cols_names = [cols_names_to_fit[i] for i in np.arange(all_vars_filtered.shape[1])[np.abs(pipeline._final_estimator.coef_) > 0]]
    coef = pipeline._final_estimator.coef_[np.abs(pipeline._final_estimator.coef_) > 0]
    score = pipeline.score(all_vars_filtered, y)
    print(score)
    print(cols_names)
    print(coef)
    # f, p = f_classif(np.nan_to_num(all_vars_filtered, copy=False), y)
    # return f, p
    # print(search.best_params_)
    # {'model__alpha': 0.30000000000000004}
    return cols_names, coef, score, sample_size

# ## REPM Configs
repm_configs = []
path = "./configs/repm/"
for filename in os.listdir(path):
    if not filename.endswith(".yaml"):
        continue
    repm_configs.append(os.path.join(path, filename))
repm_configs.sort()
# list(enumerate(repm_configs))


def load_variables():
    vars_used = []
    t0 = time.time()
    # TODO: Fix RAM limit
    # this loop will stall the system if it loads all 1200 variables
    mat_list = []
    for i, var in enumerate(tqdm(valid_b_vars[:])):
        # skip variables in the list or if it's standardized variable
        if var in vars_to_skip or 'st_' in var:
            continue
        s = buildings.to_frame(var).iloc[:, 0]
        if pd.api.types.is_numeric_dtype(s):
            vars_used.append(var)
            sparse_mat = scipy.sparse.csr_matrix(s.values)
            mat_list.append(sparse_mat)
        # clear cache every 50 vars
        if i % 50 ==0: orca.clear_columns('buildings')
    # clear all orca cache
    orca.clear_all()
    mat = scipy.sparse.vstack(mat_list)
    t1 = time.time()
    # 2692.167008 MB RAM usage with 500 variables loaded
    print("finsihed in ", t1 - t0)
    return vars_used, mat

buildings = orca.get_table('buildings')
# load variables
n = len(valid_b_vars)
vars_used, mat = load_variables()
# cannot use orca from now on

# save matrix
# scipy.sparse.save_npz('sparse_matrix.npz', mat)
# load matrix
# mat = scipy.sparse.load_npz('sparse_matrix.npz')
# save vars_used
# with open('vars_used.txt', 'w') as f:
#     f.write(','.join(vars_used))
# load vars_used
# with open('vars_used.txt', 'r') as f:
#     vars_used = f.readline().split(',')
# f, p = feature_selection_lasso(mat, repm_configs[-1], vars_used)

result = {}
for config in repm_configs:
	col_names, coef, score, sample_size = feature_selection_lasso(mat, config, vars_used)
	if sample_size == 0:
		result[config] = {
            'coef': 'None',
            'score': 'None',
            'sample_size': sample_size
        }
		continue
	result[config] = {
       'coef': {col_names[i]: float(coef[i]) for i in range(len(col_names))},
       'score': score,
       'sample_size': sample_size
    }
    # f, p = feature_selection_f_classif(mat, config, vars_used)
    # f = np.nan_to_num(f)
    # p = np.nan_to_num(p)
    # print('\t'.join([str(vars_used[x]) for x in np.argsort(p)[::-1][:10]]))
    # print('\t'.join([str(p[x]) for x in np.argsort(p)[::-1][:10]]))
for k in result:
	if type(result[k]['sample_size']) == np.int64: 
		result[k]['sample_size'] = result[k]['sample_size'].item()
	if result[k]['coef'] == 'None': continue
	result[k]['score'] = float(result[k]['score'])
	for coe in result[k]['coef']:
		result[k]['coef'][coe] = float(result[k]['coef'][coe])
with open('repm_result.yaml', 'w') as f:
    yaml.dump(result, f, default_flow_style=False)
# print(f_classif(df.values, y))
print('done')
