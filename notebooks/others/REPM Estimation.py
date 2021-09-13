#!/usr/bin/env python
# coding: utf-8

# In[1]:


from urbansim.models.regression import RegressionModel

import os
import numpy as np
import pandas as pd

import orca
from urbansim.utils import misc

import dataset
import variables
import models
import utils

orca.run(["build_networks"])
orca.run(["neighborhood_vars"])


# ## REPM Estimation Function

# In[7]:


def estimate_repm(yaml_config):
    model = RegressionModel.from_yaml(str_or_buffer=misc.config(yaml_config))

    b = orca.get_table('buildings').to_frame(model.columns_used()).fillna(0)

    print(model.fit(b))
    
    model.to_yaml(str_or_buffer=misc.config(yaml_config)) #  .replace('.yaml', '_new.yaml')

    return model.fit_parameters


# ## REPM Configs

# In[2]:


repm_configs=[]
path = './configs/repm/'
for filename in os.listdir(path):
    if not filename.endswith('.yaml'): continue
    repm_configs.append(os.path.join(path, filename).replace('./configs/', ''))
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


# In[ ]:




