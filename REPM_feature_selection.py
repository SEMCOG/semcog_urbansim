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
from guppy import hpy; h=hpy()

import pymrmr
from sklearn.feature_selection import SelectKBest, f_classif

os.chdir("/home/da/semcog_urbansim")
sys.path.append("/home/da/forecast_data_input")
from data_checks.estimation_variables_2050 import *


# import utils
# data_path = r"/home/da/share/U_RDF2050/model_inputs/base_hdf"
data_path = r"/home/da/share/urbansim/RDF2050/model_inputs/base_hdf"
hdf_list = [
    (data_path + "/" + f)
    for f in os.listdir(data_path)
    if ("forecast_data_input" in f) & (f[-3:] == ".h5")
]
hdf_last = max(hdf_list, key=os.path.getctime)
hdf = pd.HDFStore(hdf_last, "r")
print("HDF data: ", hdf_last)


orca.add_injectable("store", hdf)
load_tables_to_store()
from notebooks.models_test import *


orca.run(["build_networks"])
orca.run(["neighborhood_vars"])

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

def feature_selection(mat, yaml_config, vars_used):
    filter_cols = ['sqft_price_nonres', 'sqft_price_res', 'non_residential_sqft',  'hedonic_id', 'residential_units']
    filter_col_ind = [vars_used.index(name) for name in filter_cols]
    df = pd.DataFrame(np.vstack([mat.getrow(i).todense() for i in filter_col_ind]).transpose(), columns=filter_cols)
    with open( yaml_config, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    fit_filters = conf['fit_filters']
    target_var = conf['model_expression']['left_side']
    target_var_raw = target_var[target_var.find("(")+1:target_var.find(")")]
    filtered_df = apply_filter_query(df, fit_filters)
    if filtered_df.size == 0: 
        print("0 size dataframe after filtering")
        return None, None
    filtered_ind = filtered_df.index
    # get filtered rows, if load all variables, sys will stalled 
    # all_vars_filtered = pd.DataFrame(np.hstack([mat.getcol(i).todense() for i in filtered_ind]).transpose(), columns=vars_used)
    all_vars_filtered = mat.toarray()[:,filtered_ind].transpose()
    y = filtered_df.loc[:, target_var_raw].values
    f, p = f_classif(np.nan_to_num(all_vars_filtered, copy=False), y)
    return f, p

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
        if var in ['parcel_id', 'geoid', 'b_ln_parcels_parcel_far', 'parcels_parcel_far']:
            continue
        s = buildings.to_frame(var).iloc[:, 0]
        if pd.api.types.is_numeric_dtype(s):
            vars_used.append(var)
            sparse_mat = scipy.sparse.csr_matrix(s.values)
            mat_list.append(sparse_mat)
        # clear cache every 50 vars
        if i % 50 ==0: orca.clear_columns('buildings')
    mat = scipy.sparse.vstack(mat_list)
    t1 = time.time()
    # 2692.167008 MB RAM usage with 500 variables loaded
    print("finsihed in ", t1 - t0)
    return vars_used, mat

with open('/home/da/share/urbansim/RDF2050/model_inputs/base_hdf/081022_variable_validation.yaml', 'r') as f:
    vars_config = yaml.load(f, Loader=yaml.FullLoader)
buildings = orca.get_table('buildings')
valid_b_vars = vars_config['buildings']['valid variables']

# load variables
n = len(valid_b_vars)
vars_used, mat = load_variables()
# clear all orca cache
orca.clear_all()
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

for config in repm_configs:
    f, p = feature_selection(mat, config, vars_used)
    if not p:
        continue
    print('\t'.join([vars_used[x] for x in np.argsort(p)[::-1][:10]]))
    print('\t'.join([p[x] for x in np.argsort(p)[::-1][:10]]))
# print(f_classif(df.values, y))
print('done')