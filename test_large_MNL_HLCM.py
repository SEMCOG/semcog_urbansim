#!/usr/bin/env python
# coding: utf-8
from urbansim.models.regression import RegressionModel

import os
import numpy as np
import pandas as pd
import orca
from urbansim.utils import misc
from urbansim.models import util
import sys
import time
from tqdm import tqdm
import time

from dcm_ard_libs import minimize, neglog_DCM

from urbansim_templates.models import LargeMultinomialLogitStep

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

def columns_in_vars(vars):
    hh_columns, b_columns = [], []
    for varname in vars:
        if ':' in varname:
            vs = varname.split(':')
            hh_columns.append(vs[0])
            b_columns.append(vs[1])
        else:
            hh_columns.append(varname)
            b_columns.append(varname)
    return hh_columns, b_columns

def load_hlcm_df(hh_var, b_var):
    # load both hh
    hh = households.to_frame(hh_var)
    b = buildings.to_frame(b_var)
    return hh, b
            
@orca.column('households', cache=True, cache_scope='iteration')
def residential_units(households, buildings):
    return misc.reindex(buildings.residential_units, households.building_id)

@orca.column('households', cache=True, cache_scope='iteration')
def year_built(households, buildings):
    return misc.reindex(buildings.year_built, households.building_id)

@orca.column('households', cache=True, cache_scope='iteration')
def mcd_model_quota(households, buildings):
    return misc.reindex(buildings.mcd_model_quota, households.building_id)

thetas = pd.read_csv("out_theta_50.txt", index_col=0)
hh_filter_columns = ["building_id", "large_area_id", "mcd_model_quota", "year_built", "residential_units"]
b_filter_columns = ["large_area_id", "mcd_model_quota"]

if False:
    orca.add_injectable('year', 2020)
    # config
    choice_column = "building_id"
    hh_sample_size = 10000
    estimation_sample_size = 50
    # load variables
    orca.add_injectable("store", hdf)
    load_tables_to_store()
    from notebooks.models_test import *
    import variables
    buildings = orca.get_table("buildings")
    households = orca.get_table("households")
    orca.run(["build_networks"])
    orca.run(["neighborhood_vars"])
    orca.run(["mcd_hu_sampling"])
    # TODO: get vars from vars list from last forecast
    used_vars = thetas.index
    hh_columns, b_columns = columns_in_vars(used_vars)


    hh_var = hh_columns + hh_filter_columns
    b_var = b_columns + b_filter_columns
    hh, b = load_hlcm_df(hh_var, b_var)
    hh.to_csv('hh.csv')
    b.to_csv('b.csv')
else:
    hh = pd.read_csv('hh.csv', index_col=0)
    b = pd.read_csv('b.csv', index_col=0)

hh = hh[hh.large_area_id == 125]
hh = hh[hh.building_id > 1]
hh = hh[hh.residential_units > 0]
hh = hh[hh.year_built > 2005]
hh = hh.fillna(0) # found 12 missing values in ln_income
hh = hh[[col for col in hh.columns if col not in hh_filter_columns+["household_id"]]+['building_id']]

b = b[b.large_area_id == 125]
b = b[b.residential_units > 0]
# b = b[b.year_built > 2000]
b = b[[col for col in b.columns if col not in b_filter_columns]]

# (df-df.mean())/df.std()

hh_cols_to_std = [col for col in hh.columns if col not in ['building_id']]
hh[hh_cols_to_std] = (hh[hh_cols_to_std]-hh[hh_cols_to_std].mean())/hh[hh_cols_to_std].std()
b_cols_to_std = [col for col in b.columns]
b[b_cols_to_std] = (b[b_cols_to_std]-b[b_cols_to_std].mean())/b[b_cols_to_std].std()
orca.add_table('hh', hh)
orca.add_table('b', b)

m = LargeMultinomialLogitStep()
m.choosers = ['hh']
m.chooser_sample_size = 10000
# m.chooser_filters = chooser_filter

# Define the geographic alternatives agent is selecting amongst
m.alternatives = ['b']
m.choice_column = 'building_id'
m.alt_sample_size = 25
# m.alt_filters = alts_filter

# use top 40 variables
selected_variables = thetas.theta.abs().sort_values(ascending=False).index[:25]

m.model_expression = util.str_model_expression(selected_variables, add_constant=False)

# m.out_choosers = 'hh161' # if different from estimation
# m.out_alternatives = 'bb161' # if different from estimation
m.constrained_choices = True
m.alt_capacity = 'residential_units'
# m.out_chooser_filters = ['building_id == -1']


# %%
# model estimation and save the results to default folder "configs/"

m.fit()
m.name = 'hlcm_city_test_125'
with open("configs/hlcm_city_test_125.yaml", 'w') as f:
    yaml.dump(m.to_dict(), f, default_flow_style=False)
#mm.register(m)

print('done')
