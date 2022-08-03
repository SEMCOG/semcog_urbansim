# This is the script for MNLDiscreteChoiceModel estimation and simulation demo - Guangyu
# run it under semcog_urbansim folder

# In[ ]:
from urbansim.models.dcm import MNLDiscreteChoiceModel

import os
import numpy as np
import pandas as pd
import orca
from urbansim.utils import misc

os.chdir("/home/da/semcog_urbansim")
from notebooks.estimation_variables_2050 import *


import utils

# ++++++ estimation input preparation +++++++++++
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


# ++++++++++ estimate MNLDiscreteChoiceModel ++++++++++++++++++++
######## estimation demo ###########
# %%
m2 = MNLDiscreteChoiceModel.from_yaml(str_or_buffer=misc.config("test_indiv.yaml"))
m2.probability_mode

# %%
# compile data for estimation
hh = orca.get_table("households").to_frame(m2.columns_used()).fillna(0)
b = orca.get_table("buildings").to_frame(m2.columns_used()).fillna(0)

# %%
m2.fit(hh, b, hh[m2.choice_column])

#%%
m2.to_yaml("test_indiv1.yaml")  # save configs


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
