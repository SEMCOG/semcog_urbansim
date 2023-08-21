#!/usr/bin/env python
# coding: utf-8
from urbansim.models.regression import RegressionModel

import os
import numpy as np
import pandas as pd
import orca
from urbansim.models import util
import yaml

from urbansim_templates.models import LargeMultinomialLogitStep
from urbansim_templates import modelmanager as mm

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
    import models
    orca.add_injectable('year', 2020)
    buildings = orca.get_table("buildings")
    households = orca.get_table("households")
    orca.run(["build_networks_2050"])
    orca.run(["neighborhood_vars"])
    # load both hh
    hh = households.to_frame(hh_var)
    b = buildings.to_frame(b_var)
    return hh, b
            

hh_sample_size = 10000
estimation_sample_size = 50
job_sample_size = 1000
job_estimation_sample_size = 80
# LARGE_AREA_ID = 147
# number_of_vars_to_use = 40
choice_column = "building_id"
hh_filter_columns = ["building_id", "large_area_id", "mcd_model_quota", "year_built", "residential_units"]
job_filter_columns = ["building_id", "slid", "home_based_status"]
b_filter_columns = ["large_area_id", "mcd_model_quota", "residential_units", "non_residential_sqft", "vacant_job_spaces"]
building_hh_capacity_col = 'residential_units'
building_job_capacity_col = 'vacant_job_spaces'

# reload variables?
def load_hh_and_b(LARGE_AREA_ID=5, RELOAD=False):
    if RELOAD:
        thetas = pd.read_csv("out_theta_%s_%s.txt" % (LARGE_AREA_ID, estimation_sample_size), index_col=0)
        # config
        # load variables
        # orca.run(["mcd_hu_sampling"])
        # TODO: get vars from vars list from last forecast
        used_vars = thetas.index
        hh_columns, b_columns = columns_in_vars(used_vars)

        hh_var = hh_columns + hh_filter_columns
        b_var = b_columns + b_filter_columns
        hh_region, b_region = load_hlcm_df(hh_var, b_var)
        hh_region.to_csv('hh.csv')
        b_region.to_csv('b.csv')
    else:
        hh_region = pd.read_csv('hh.csv', index_col=0)
        b_region = pd.read_csv('b.csv', index_col=0)
    return hh_region, b_region

def run_large_MNL(hh_region, b_region, LARGE_AREA_ID, number_of_vars_to_use=40):
    thetas = pd.read_csv("./configs/hlcm_2050/thetas/out_theta_%s_%s.txt" % (LARGE_AREA_ID, estimation_sample_size), index_col=0)

    hh = hh_region[hh_region.large_area_id == LARGE_AREA_ID]
    hh = hh[hh.building_id > 1]
    hh = hh[hh.residential_units > 0]
    hh = hh[hh.year_built > 2005]
    # exclude hh in pseudo buildings
    hh = hh[hh.building_id < 90000000]
    hh = hh.fillna(0) # found 12 missing values in ln_income
    hh = hh[[col for col in hh.columns if col not in hh_filter_columns+["household_id"]]+['building_id']]

    b = b_region[b_region.large_area_id == LARGE_AREA_ID]
    b = b[b.residential_units > 0]
    b = b[[col for col in b.columns if col not in b_filter_columns or col == building_hh_capacity_col]]

    # remove extra columns
    hh_cols_to_std = [col for col in hh.columns if col not in ['building_id']]

    # standardize hh
    hh[hh_cols_to_std] = (hh[hh_cols_to_std]-hh[hh_cols_to_std].mean())/hh[hh_cols_to_std].std()

    # building columsn to standardize
    b_cols_to_std = [col for col in b.columns if col not in [building_hh_capacity_col]]
    # standardize buildings
    b[b_cols_to_std] = (b[b_cols_to_std]-b[b_cols_to_std].mean())/b[b_cols_to_std].std()

    # fillin na with 0
    b = b.fillna(0)

    # filter out columns with 0 std
    b_cols_with_0_std = b.columns[b.std()==0]

    # adding hh and b to orca
    orca.add_table('hh_hlcm', hh)
    orca.add_table('b_hlcm', b)

    m = LargeMultinomialLogitStep()
    m.choosers = ['hh_hlcm']
    m.chooser_sample_size = min(hh_sample_size, hh.shape[0])
    # m.chooser_filters = chooser_filter

    # Define the geographic alternatives agent is selecting amongst
    m.alternatives = ['b_hlcm']
    m.choice_column = choice_column
    m.alt_sample_size = estimation_sample_size
    # m.alt_filters = alts_filter

    # use top k variables
    # filter variables
    # some variables has 0 std, need to remove them for the MNL to run
    v = thetas.theta.abs().sort_values(ascending=False).index
    v_wo_0_std = [col for col in v if all(
        [vv.strip() not in b_cols_with_0_std for vv in col.split(':')])]
    selected_variables = v_wo_0_std[:number_of_vars_to_use]

    m.model_expression = util.str_model_expression(selected_variables, add_constant=False)

    m.constrained_choices = True
    m.alt_capacity = building_hh_capacity_col

    m.fit()
    m.name = 'hlcm_%s' % (LARGE_AREA_ID)
    mm.register(m)

    print('done')

def run_elcm_large_MNL(job_region, b_region, SLID, number_of_vars_to_use=40):
    print("Running elcm largeMNL on slid %s..." % SLID)
    thetas = pd.read_csv("./configs/elcm_2050/thetas/out_theta_job_%s_%s.txt" % (SLID, job_estimation_sample_size), index_col=0)

    job = job_region[job_region.slid == SLID]
    job = job[job.building_id > 1]
    job = job[job.home_based_status == 0]
    job = job.fillna(0) # found 12 missing values in ln_income
    job = job[[col for col in job.columns if col not in job_filter_columns+["job_id"]]+['building_id']]

    b = b_region[b_region.large_area_id == SLID % 1000]
    b = b[b.non_residential_sqft > 0]
    b = b[[col for col in b.columns if col not in b_filter_columns or col == building_job_capacity_col]]
    # remove extra columns
    job_cols_to_std = [col for col in job.columns if col not in ['building_id', building_job_capacity_col]]
    # standardize job
    job[job_cols_to_std] = (job[job_cols_to_std]-job[job_cols_to_std].mean())/job[job_cols_to_std].std()

    b_cols_to_std = [col for col in b.columns if col not in [building_job_capacity_col]]
    # standardize buildings
    b[b_cols_to_std] = (b[b_cols_to_std]-b[b_cols_to_std].mean())/b[b_cols_to_std].std()

    # fillin na with 0
    b = b.fillna(0)

    # filter out columns with 0 std
    b_cols_with_0_std = b.columns[b.std()==0]

    # adding job and b to orca
    orca.add_table('job_elcm', job)
    orca.add_table('b_elcm', b)

    m = LargeMultinomialLogitStep()
    m.choosers = ['job_elcm']
    m.chooser_sample_size = min(job_sample_size, job.shape[0])
    # m.chooser_filters = "(slid==%s) & (building_id>1) & (home_based_status==0)" % (SLID)
    # m.chooser_sample_size = min(job_sample_size, jobs.query(m.chooser_filters).shape[0])

    # Define the geographic alternatives agent is selecting amongst
    m.alternatives = ['b_elcm']
    m.choice_column = choice_column
    m.alt_sample_size = job_estimation_sample_size
    # m.alt_filters = "(large_area_id==%s) & (non_residential_sqft>0)" % (SLID % 100)

    # use top k variables
    # filter variables
    # some variables has 0 std, need to remove them for the MNL to run
    v = thetas.theta.abs().sort_values(ascending=False).index
    v_wo_0_std = [col for col in v if all(
        [vv.strip() not in b_cols_with_0_std for vv in col.split(':')])]
    # remove vacant_job_spaces from selected variables
    v_wo_0_std = [col for col in v_wo_0_std if col not in ['vacant_job_spaces']]
    selected_variables = v_wo_0_std[:number_of_vars_to_use]
    # remove variables which with correlation close to 1
    i = number_of_vars_to_use
    while ((b[selected_variables].corr()>0.99).sum()>1).sum() > 0 and i<len(v_wo_0_std):
        corr = b[selected_variables].corr()
        corr_ge_1_sum = (corr>0.99).sum()
        # drop the first variable
        var_to_drop = corr_ge_1_sum[corr_ge_1_sum>1].index[0]
        print("Removed %s from expression due to close to 1 correlation" % var_to_drop)
        selected_variables.remove(var_to_drop)
        print("Added %s as replacement" % v_wo_0_std[i])
        selected_variables.append(v_wo_0_std[i])
        i += 1
        
    m.model_expression = util.str_model_expression(selected_variables, add_constant=False)

    m.constrained_choices = True
    m.alt_capacity = building_job_capacity_col

    m.fit()
    fp = m.model.results['fit_parameters']
    rho = [m.model.results['log_likelihood']['rho_bar_squared']]
    while fp[(fp['P-Values']>0.1) | (fp['P-Values'].isna())].shape[0]>0 and fp.shape[0] > 2:
        # when there are variables with p > .1
        # removing them and refit
        new_vars_inds = [i for i in range(
            fp.shape[0]) if i not in fp[(fp['P-Values']>0.1) | (fp['P-Values'].isna())].index]
        new_vars = [m.model.results['x_names'][i] for i in new_vars_inds]
        m.model_expression = util.str_model_expression(new_vars, add_constant=False)
        m.fit()
        fp = m.model.results['fit_parameters']
        rho.append(m.model.results['log_likelihood']['rho_bar_squared'])

    m.name = 'elcm_%s' % (SLID)
    mm.register(m)
    print('done')

if __name__ == '__main__':
    hh_region, b_region = load_hh_and_b(93, False)
    run_large_MNL(hh_region, b_region, 93, 60)

