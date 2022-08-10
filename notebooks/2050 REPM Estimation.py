#!/usr/bin/env python
# coding: utf-8

#8/5/2022, set up for incomplete input data
#%%
from urbansim.models.regression import RegressionModel

import os
import numpy as np
import pandas as pd
import orca
from urbansim.utils import misc
import sys

os.chdir("/home/da/semcog_urbansim")
sys.path.append("/home/da/forecast_data_input")
from data_checks.estimation_variables_2050 import *


# import utils
#%%
data_path = r"/home/da/share/U_RDF2050/model_inputs/base_hdf"
# data_path = r"/home/da/share/urbansim/RDF2050/model_inputs/base_hdf"
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
orca.run(["build_networks"])
orca.run(["neighborhood_vars"])


# ## REPM Estimation Function
#%%
def estimate_repm(yaml_config):
    model = RegressionModel.from_yaml(str_or_buffer=misc.config(yaml_config))
    b = orca.get_table("buildings").to_frame(model.columns_used()).fillna(0)
    print(model.fit(b))
    model.to_yaml(str_or_buffer=misc.config(yaml_config))
    return model.fit_parameters


# ## REPM Configs
# In[2]:
repm_configs = []
path = "./configs/repm/"
for filename in os.listdir(path):
    if not filename.endswith(".yaml"):
        continue
    repm_configs.append(os.path.join(path, filename).replace("./configs/", ""))
repm_configs.sort()
list(enumerate(repm_configs))


# ## Estimate a particular REPM submodel
# In[ ]:
estimate_repm(repm_configs[0])


# In[8]:
for i, lcm in list(enumerate(repm_configs))[:]:
    print()
    print(i, lcm)
    estimate_repm(lcm)

# In[9]:
import scipy
import time
with open('/home/da/share/urbansim/RDF2050/model_inputs/base_hdf/081022_variable_validation.yaml', 'r') as f:
    vars_config = yaml.load(f, Loader=yaml.FullLoader)
buildings = orca.get_table('buildings')
valid_b_vars = vars_config['buildings']['valid variables']
vars_used = []
mat = None
t0 = time.time()
for var in valid_b_vars[:]:
    s = buildings.to_frame(var).iloc[:, 0]
    if pd.api.types.is_numeric_dtype(s):
        vars_used.append(var)
        sparse_mat = scipy.sparse.csr_matrix(s.values)
        mat = scipy.sparse.vstack([mat, sparse_mat])
t1 = time.time()
# 2692.167008 MB RAM usage with 500 variables loaded
# save matrix to npz
scipy.sparse.save_npz('sparse_matrix.npz', mat)
print("finsihed in ", t1 - t0)
# %%
