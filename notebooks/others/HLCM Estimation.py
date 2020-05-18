#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from urbansim.models.dcm import MNLDiscreteChoiceModel

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


# ## HLCM Estimation Function

# In[5]:


def estimate_hlcm(yaml_config):
    model = MNLDiscreteChoiceModel.from_yaml(str_or_buffer=misc.config(yaml_config))

    hh = orca.get_table('households').to_frame(model.columns_used()).fillna(0)
    b = orca.get_table('buildings').to_frame(model.columns_used()).fillna(0)

    print model.fit(hh, b, hh[model.choice_column])

    return model.fit_parameters


# ## HLCM Configs

# In[8]:


hlcm_configs = ['hlcm/large_area_income_quartile/hlcm400099.yaml',
                 'hlcm/large_area_income_quartile/hlcm200115.yaml',
                 'hlcm/large_area_income_quartile/hlcm100125.yaml',
                 'hlcm/large_area_income_quartile/hlcm100161.yaml',
                 'hlcm/large_area_income_quartile/hlcm200093.yaml',
                 'hlcm/large_area_income_quartile/hlcm300099.yaml',
                 'hlcm/large_area_income_quartile/hlcm400147.yaml',
                 'hlcm/large_area_income_quartile/hlcm100093.yaml',
                 'hlcm/large_area_income_quartile/hlcm200125.yaml',
                 'hlcm/large_area_income_quartile/hlcm100147.yaml',
                 'hlcm/large_area_income_quartile/hlcm100005.yaml',
                 'hlcm/large_area_income_quartile/hlcm200003.yaml',
                 'hlcm/large_area_income_quartile/hlcm400003.yaml',
                 'hlcm/large_area_income_quartile/hlcm200161.yaml',
                 'hlcm/large_area_income_quartile/hlcm300005.yaml',
                 'hlcm/large_area_income_quartile/hlcm300147.yaml',
                 'hlcm/large_area_income_quartile/hlcm400115.yaml',
                 'hlcm/large_area_income_quartile/hlcm300115.yaml',
                 'hlcm/large_area_income_quartile/hlcm400125.yaml',
                 'hlcm/large_area_income_quartile/hlcm400161.yaml',
                 'hlcm/large_area_income_quartile/hlcm300125.yaml',
                 'hlcm/large_area_income_quartile/hlcm300161.yaml',
                 'hlcm/large_area_income_quartile/hlcm200005.yaml',
                 'hlcm/large_area_income_quartile/hlcm400093.yaml',
                 'hlcm/large_area_income_quartile/hlcm100003.yaml',
                 'hlcm/large_area_income_quartile/hlcm100099.yaml',
                 'hlcm/large_area_income_quartile/hlcm100115.yaml',
                 'hlcm/large_area_income_quartile/hlcm400005.yaml',
                 'hlcm/large_area_income_quartile/hlcm200099.yaml',
                 'hlcm/large_area_income_quartile/hlcm300003.yaml',
                 'hlcm/large_area_income_quartile/hlcm200147.yaml',
                 'hlcm/large_area_income_quartile/hlcm300093.yaml']


# ## Estimate a particular HLCM submodel

# In[9]:


estimate_hlcm(hlcm_configs[0])

