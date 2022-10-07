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
import yaml

from dcm_ard_libs import minimize, neglog_DCM

from urbansim_templates.models import LargeMultinomialLogitStep

# from guppy import hpy; h=hpy()
# import pymrmr

# suppress sklearn warnings
def warn(*args, **kwargs):
    pass

os.chdir("/home/da/semcog_urbansim")

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
            

hh_sample_size = 10000
estimation_sample_size = 50
LARGE_AREA_ID = 147
number_of_vars_to_use = 40
choice_column = "building_id"
hh_filter_columns = ["building_id", "large_area_id", "mcd_model_quota", "year_built", "residential_units"]
b_filter_columns = ["large_area_id", "mcd_model_quota", "residential_units"]

thetas = pd.read_csv("out_theta_%s_%s.txt" % (LARGE_AREA_ID, estimation_sample_size), index_col=0)
# reload variables?
RELOAD = False
if RELOAD:
    # config
    # load variables
    import models
    orca.add_injectable('year', 2020)
    buildings = orca.get_table("buildings")
    households = orca.get_table("households")
    orca.run(["build_networks_2050"])
    orca.run(["neighborhood_vars"])
    # orca.run(["mcd_hu_sampling"])
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

hh = hh[hh.large_area_id == LARGE_AREA_ID]
hh = hh[hh.building_id > 1]
hh = hh[hh.residential_units > 0]
hh = hh[hh.year_built > 2005]
hh = hh.fillna(0) # found 12 missing values in ln_income
hh = hh[[col for col in hh.columns if col not in hh_filter_columns+["household_id"]]+['building_id']]

b = b[b.large_area_id == LARGE_AREA_ID]
b = b[b.residential_units > 0]
# b = b[b.year_built > 2000]
b = b[[col for col in b.columns if col not in b_filter_columns]]

# (df-df.mean())/df.std()

# remove extra columns
hh_cols_to_std = [col for col in hh.columns if col not in ['building_id']]
# standardize hh
hh[hh_cols_to_std] = (hh[hh_cols_to_std]-hh[hh_cols_to_std].mean())/hh[hh_cols_to_std].std()
b_cols_to_std = [col for col in b.columns]

b_cols_with_0_std = b.columns[b.std()==0]
# standardize buildings
b[b_cols_to_std] = (b[b_cols_to_std]-b[b_cols_to_std].mean())/b[b_cols_to_std].std()
# adding hh and b to orca
orca.add_table('hh', hh)
orca.add_table('b', b)

m = LargeMultinomialLogitStep()
m.choosers = ['hh']
m.chooser_sample_size = min(hh_sample_size, hh.shape[0])
# m.chooser_filters = chooser_filter

# Define the geographic alternatives agent is selecting amongst
m.alternatives = ['b']
m.choice_column = choice_column
m.alt_sample_size = estimation_sample_size
# m.alt_filters = alts_filter

# use top 40 variables
# filter variables
# some variables has 0 std, need to remove them for the MNL to run
v = thetas.theta.abs().sort_values(ascending=False).index
v_wo_0_std = [col for col in v if all(
    [vv.strip() not in b_cols_with_0_std for vv in col.split(':')])]
selected_variables = v_wo_0_std[:number_of_vars_to_use]
# add 10 least important variables
# selected_variables = np.concatenate((selected_variables, thetas.theta.abs().sort_values(ascending=False).index[-10:]))

m.model_expression = util.str_model_expression(selected_variables, add_constant=False)

# m.out_choosers = 'hh161' # if different from estimation
# m.out_alternatives = 'bb161' # if different from estimation
m.constrained_choices = True
m.alt_capacity = 'residential_units'
# m.out_chooser_filters = ['building_id == -1']


# %%
# model estimation and save the results to default folder "configs/"

m.fit()
m.name = 'hlcm_city_test_%s' % (LARGE_AREA_ID)
with open("configs/hlcm_city_test_%s_%svars.yaml" % (LARGE_AREA_ID, len(selected_variables)), 'w') as f:
    yaml.dump(m.to_dict(), f, default_flow_style=False)
#mm.register(m)

print('done')
