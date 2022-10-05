# This is the script for MNLDiscreteChoiceModel estimation and simulation demo - Guangyu
# run it under semcog_urbansim folder

# In[ ]:
from urbansim.models.dcm import MNLDiscreteChoiceModel

import os
import sys
import yaml
import numpy as np
import pandas as pd
import orca
from urbansim.utils import misc

os.chdir("/home/da/semcog_urbansim")
sys.path.append("/home/da/forecast_data_input")
from data_checks.estimation_variables_2050 import *


# ++++++ estimation input preparation +++++++++++
#%% load input data
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

orca.add_injectable("store", hdf)
load_tables_to_store()
import variables

from notebooks.models_test import *
@orca.column('households', cache=True, cache_scope='iteration')
def residential_units(households, buildings):
    return misc.reindex(buildings.residential_units, households.building_id)
@orca.column('households', cache=True, cache_scope='iteration')
def year_built(households, buildings):
    return misc.reindex(buildings.year_built, households.building_id)

#%%
# compute network and variables
orca.run(["build_networks"])
orca.run(["neighborhood_vars"])

#%%
# create yaml
theta_df = pd.read_csv('~/semcog_urbansim/out_theta_50.txt', index_col=0)
chooser_filter = "large_area_id == 125 & building_id > 1 & residential_units > 0 & year_built > 2005"
alts_filter = "large_area_id == 125 & residential_units > 0"

varnames = theta_df.index[:40]
model_expression = ' + '.join(varnames)
with open("./configs/hlcm_2050_testing.yaml", "w") as f:
    yaml.dump({
        'name': 'MNLDiscreteChoiceModel',
        'model_type': 'discretechoice',
        'choosers_fit_filters': chooser_filter,
        'choosers_predict_filters': chooser_filter,
        'alts_fit_filters': alts_filter,
        'alts_predict_filters': alts_filter,
        'choice_column': 'building_id',
        'sample_size': 100,
        'estimation_sample_size': 10000,
        'prediction_sample_size': 100,
        'model_expression': model_expression,
    }, f, default_flow_style=False)

# create comparison yaml wtih 21 variables
varnames = theta_df.index[:41]
model_expression = ' + '.join(varnames)
with open("./configs/hlcm_2050_testing_2.yaml", "w") as f:
    yaml.dump({
        'name': 'MNLDiscreteChoiceModel',
        'model_type': 'discretechoice',
        'choosers_fit_filters': chooser_filter,
        'choosers_predict_filters': chooser_filter,
        'alts_fit_filters': alts_filter,
        'alts_predict_filters': alts_filter,
        'choice_column': 'building_id',
        'sample_size': 100,
        'estimation_sample_size': 10000,
        'prediction_sample_size': 100,
        'model_expression': model_expression,
    }, f, default_flow_style=False)
# ++++++++++ estimate MNLDiscreteChoiceModel ++++++++++++++++++++
######## estimation demo ###########
# %%
m1 = MNLDiscreteChoiceModel.from_yaml(str_or_buffer=misc.config("hlcm_2050_testing.yaml"))
# m1.probability_mode
# compile data for estimation
hh = orca.get_table("households").to_frame(m1.columns_used()).fillna(0)
b = orca.get_table("buildings").to_frame(m1.columns_used()).fillna(0)
m1.fit(hh, b, hh[m1.choice_column])
m1.to_yaml("test_indiv1.yaml")  # save configs

# %%
m2 = MNLDiscreteChoiceModel.from_yaml(str_or_buffer=misc.config("hlcm_2050_testing_2.yaml"))
# m2.probability_mode
# compile data for estimation
hh = orca.get_table("households").to_frame(m2.columns_used()).fillna(0)
b = orca.get_table("buildings").to_frame(m2.columns_used()).fillna(0)
m2.fit(hh, b, hh[m2.choice_column])
m2.to_yaml("test_indiv2.yaml")  # save configs

# %%
# ++++++++++++++++ simulation demo +++++++++++++
# select HH and buildings
hhs = hh.loc[hh.large_area_id == 161]
hhs = hhs.sample(n=5)
hhs.building_id = -1

bs = b.loc[b.large_area_id == 161]
bs = bs.sample(n=5)

# %%
# test simulation
chs = m2.predict(hhs, bs, debug=True)

# %%
chs


# %%
m2prob = m2.probabilities(hhs, bs)
m2prob

# ===========

# ## HLCM Estimation Function  old one
#%%
# def estimate_hlcm(yaml_config):
#     model = MNLDiscreteChoiceModel.from_yaml(str_or_buffer=misc.config(yaml_config))

#     hh = orca.get_table("households").to_frame(model.columns_used()).fillna(0)
#     b = orca.get_table("buildings").to_frame(model.columns_used()).fillna(0)

#     print(model.fit(hh, b, hh[model.choice_column]))

#     return model, model.fit_parameters

# ## HLCM Configs

# In[8]:
# hlcm_configs = [
#     "hlcm/large_area_income_quartile/hlcm400099.yaml",
#     "hlcm/large_area_income_quartile/hlcm200115.yaml",
#     "hlcm/large_area_income_quartile/hlcm100125.yaml",
# ......
# ]
# m1, m1para = estimate_hlcm(hlcm_configs[0])
# m1.to_yaml("./test.yaml")
# ## Estimate a particular HLCM submodel
