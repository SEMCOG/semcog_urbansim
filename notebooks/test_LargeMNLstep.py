#%%
# this is the script for LargeMultinomialLogitStep estimation and simulation
# run this under semcog_urbansim folder

# %%
import os
from re import M
import numpy as np
import pandas as pd
import geopandas as gpd

import orca
from urbansim.utils import misc, networks, yamlio
from urbansim.models import util
from choicemodels.tools import MergedChoiceTable
from urbansim_templates import modelmanager as mm
from urbansim_templates.models import LargeMultinomialLogitStep
os.chdir("/home/da/semcog_urbansim")
from notebooks.estimation_variables_2050 import *

# +++++++++++ estimation input preparation +++++++++++
#%% load input data
data_path = r"/home/da/share/U_RDF2050/model_inputs/base_hdf"
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

#%%
# compute network and variables
orca.run(["build_networks"])
orca.run(["neighborhood_vars"])



#%%
#+++++++++++ largeMNL step simulation +++++++++++

#%%
#function to get extra columns from config
def filter_columns(model):
        conf_cols = [model.chooser_filters, model.alt_filters, 
                        model.out_chooser_filters, model.out_alt_filters, 
                        model.choice_column,
                        model.out_column,
                        model.alt_capacity
                ]
        sl = list(map(util.columns_in_filters, conf_cols))
        return list(set([c for subl in sl for c in subl]))

# %%
f = 'configs/hlcm_city_test.yaml'
d = yamlio.yaml_to_dict(str_or_buffer=f) #import yaml config
step = mm.build_step(d['saved_object']) # build model
mm.register(step, save_to_disk=False) #register to Orca

# %%
#prepare a simulation test
# extract only used columns in model yaml/spec
filter_cols = filter_columns(step)
express_cols = util.columns_in_formula(step.model_expression) 

hh = orca.get_table('households').to_frame(orca.get_table('households').local_columns + express_cols + filter_cols)
bb = orca.get_table('buildings').to_frame(orca.get_table('buildings').local_columns + express_cols + filter_cols)

# %%
# set up simulation example data
hh161 = hh.loc[hh.large_area_id == 161].sample(6)
bb161 = bb.loc[bb.b_city_id == 4005].sample(5)
hh161.building_id = -1
# assign different capacity
bb161.vacant_residential_units = 2
bb161.loc[bb161.index[:3],'vacant_residential_units'] = 1

orca.add_table('hh161', hh161)
orca.add_table('bb161', bb161)

# %%
print(bb161.vacant_residential_units)

#%%
#revise input data and filter 
step.out_choosers = 'hh161'
step.out_alternatives = 'bb161' 
step.constrained_choices = True
step.alt_capacity = 'vacant_residential_units'
step.out_chooser_filters = ['building_id == -1']

#%%
#run simulation
orca.run(['hlcm_city_test'])

#%%
#output, check building_id
hh161

######## largeMNLstep estimation ###########


# %%
mm.initialize('configs/hlcm_large')
# %% [markdown]
# # Overall HLCM
### list of HLCM options
# choosers
# alternatives
#  -------- below for estimation -------
# model_expression
# choice_column: id column linked to alternatives
# chooser_filters: select choosers
# chooser_sample_size: for faster fitting
# alt_filters: select chooserr/cases with certain alternatives
# alt_sample_size: limit alter size, for both estimation and predition
# 
#  -------- for simulation ------
# out_choosers: if absent, will use choosers
# out_alternatives: if absent, will use alternatives
# out_column: id column to write choice
# out_chooser_filters: select chooser for simulation
# out_alt_filters: elect alternative for simulation
# constrained_choices: True or False, capacity constrained?
# alt_capacity: capacity column name
# chooser_size: each chooser can occupies certain capacity, corresponding to alt capacity unit
# max_iter: int number of iterations, if None, run until all choosers have a alternatives
# --------
# name: name of step
# tags
#


# %% ++++++++++++++++++++++++++++++++++++++++++++++++
#  1. test estimation from scratch, add config items
# Define the agent whose behavior is being modeled
m = LargeMultinomialLogitStep()
m.choosers = ['households']
m.chooser_sample_size = 3000
m.chooser_filters = ['large_area_id == 161']

# Define the geographic alternatives agent is selecting amongst
m.alternatives = ['buildings']
m.choice_column = 'building_id'
m.alt_sample_size = 50

selected_variables = [
        'has_children:nodes_walk_percent_hh_with_children',
        'is_race2:nodes_walk_percent_race2',
         'is_young:nodes_walk_retail_jobs',
        # 'ln_income:nodes_walk_ln_popden',
        'nodes_walk_percent_low_income',
        'year_built',
                     ]

m.model_expression = util.str_model_expression(selected_variables, add_constant=False)

m.out_choosers = 'hh161' # if different from estimation
m.out_alternatives = 'bb161' # if different from estimation
m.constrained_choices = True
m.alt_capacity = 'vacant_residential_units'
m.out_chooser_filters = ['building_id == -1']


# %%
# model estimation and save the results to default folder "configs/"

m.fit()
m.name = 'hlcm_city_test2'
#mm.register(m)

#%%
dir(m)

#%%
m.chooser_filters

#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%


# %%
# ++++++++++ 2. model estimatin start from existing model config ++++++++++++++++
f = 'configs/hlcm_city_test.yaml'
d = yamlio.yaml_to_dict(str_or_buffer=f) #import yaml config
m = mm.build_step(d['saved_object']) # build model
mm.register(step, save_to_disk=False) #register to Orca

#%%

m.fit()
m.name = 'hlcm_city_test3'
mm.register(m)

#%%
dir(mm)

#%%
######## update data for estimation or simulation ###########

#%%
# change/update estimation dataset  
sl = list(map(util.columns_in_filters, extra_columns(m)))
filter_cols = list(set([c for subl in sl for c in subl]))
express_cols = util.columns_in_formula(m.model_expression) 

# %%
# filter estimation data
hh = orca.get_table('households').to_frame(orca.get_table('households').local_columns + express_cols + filter_cols)
bb = orca.get_table('buildings').to_frame(orca.get_table('buildings').local_columns + express_cols + filter_cols)
#%%
orca.add_table('hhs', hh)
orca.add_table('bbs', bb)

m.choosers = ['hhs']
m.chooser_sample_size = 5000
m.chooser_filters = ['large_area_id == 161']

# Define the geographic alternatives agent is selecting amongst
m.alternatives = ['bbs']
m.choice_column = 'building_id'
m.alt_sample_size = 100

#%%

m.fit()
m.name = 'hlcm_city_test_update'
mm.register(m)

#%%
orca.get_table('nodes_walk').to_frame("percent_race2")




# %%
#########  possible visualization code ###############
%matplotlib inline

# %%
alts = orca.get_table('cities').to_frame(selected_variables)
mct = MergedChoiceTable(pd.DataFrame([1]), alts)
mct.sample_size=len(alts)
probas = m.model.probabilities(mct).reset_index(level=0)['prob']
city_shp['probas'] = probas

city_shp.plot(column='probas', cmap='Blues', scheme='quantiles', figsize=(10, 12), legend=True)

# %%
city_shp.plot(column='total_hh_change', cmap='Blues', scheme='quantiles', figsize=(10, 12), legend=True)

# %%
city_shp.probas.corr(city_shp.total_hh_change)


