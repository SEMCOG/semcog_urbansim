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


# ## ELCM Estimation Function

# In[ ]:


def estimate_elcm(yaml_config):
    model = MNLDiscreteChoiceModel.from_yaml(str_or_buffer=misc.config(yaml_config))

    j = orca.get_table('jobs').to_frame(model.columns_used()).fillna(0)
    b = orca.get_table('buildings').to_frame(model.columns_used()).fillna(0)

    print model.fit(j, b, j[model.choice_column])

    return model.fit_parameters


# ## ELCM Configs

# In[ ]:


elcm_configs = ['elcm/large_area_sector/elcm1200093.yaml',
                 'elcm/large_area_sector/elcm1100125.yaml',
                 'elcm/large_area_sector/elcm700003.yaml',
                 'elcm/large_area_sector/elcm1000005.yaml',
                 'elcm/large_area_sector/elcm100125.yaml',
                 'elcm/large_area_sector/elcm1100161.yaml',
                 'elcm/large_area_sector/elcm200003.yaml',
                 'elcm/large_area_sector/elcm800115.yaml',
                 'elcm/large_area_sector/elcm1300005.yaml',
                 'elcm/large_area_sector/elcm700005.yaml',
                 'elcm/large_area_sector/elcm1700147.yaml',
                 'elcm/large_area_sector/elcm600147.yaml',
                 'elcm/large_area_sector/elcm800125.yaml',
                 'elcm/large_area_sector/elcm1600099.yaml',
                 'elcm/large_area_sector/elcm1100099.yaml',
                 'elcm/large_area_sector/elcm200125.yaml',
                 'elcm/large_area_sector/elcm900161.yaml',
                 'elcm/large_area_sector/elcm400099.yaml',
                 'elcm/large_area_sector/elcm300125.yaml',
                 'elcm/large_area_sector/elcm400125.yaml',
                 'elcm/large_area_sector/elcm600125.yaml',
                 'elcm/large_area_sector/elcm1100115.yaml',
                 'elcm/large_area_sector/elcm700115.yaml',
                 'elcm/large_area_sector/elcm200147.yaml',
                 'elcm/large_area_sector/elcm1200005.yaml',
                 'elcm/large_area_sector/elcm900003.yaml',
                 'elcm/large_area_sector/elcm100099.yaml',
                 'elcm/large_area_sector/elcm500147.yaml',
                 'elcm/large_area_sector/elcm100115.yaml',
                 'elcm/large_area_sector/elcm400003.yaml',
                 'elcm/large_area_sector/elcm1500003.yaml',
                 'elcm/large_area_sector/elcm1600125.yaml',
                 'elcm/large_area_sector/elcm700099.yaml',
                 'elcm/large_area_sector/elcm1200147.yaml',
                 'elcm/large_area_sector/elcm900147.yaml',
                 'elcm/large_area_sector/elcm1300003.yaml',
                 'elcm/large_area_sector/elcm1300115.yaml',
                 'elcm/large_area_sector/elcm600093.yaml',
                 'elcm/large_area_sector/elcm400115.yaml',
                 'elcm/large_area_sector/elcm100147.yaml',
                 'elcm/large_area_sector/elcm900125.yaml',
                 'elcm/large_area_sector/elcm1200003.yaml',
                 'elcm/large_area_sector/elcm400093.yaml',
                 'elcm/large_area_sector/elcm300093.yaml',
                 'elcm/large_area_sector/elcm1600147.yaml',
                 'elcm/large_area_sector/elcm500161.yaml',
                 'elcm/large_area_sector/elcm300099.yaml',
                 'elcm/large_area_sector/elcm1300147.yaml',
                 'elcm/large_area_sector/elcm1600005.yaml',
                 'elcm/large_area_sector/elcm500005.yaml',
                 'elcm/large_area_sector/elcm1400147.yaml',
                 'elcm/large_area_sector/elcm900099.yaml',
                 'elcm/large_area_sector/elcm1400161.yaml',
                 'elcm/large_area_sector/elcm400147.yaml',
                 'elcm/large_area_sector/elcm1700099.yaml',
                 'elcm/large_area_sector/elcm600099.yaml',
                 'elcm/large_area_sector/elcm500003.yaml',
                 'elcm/large_area_sector/elcm100093.yaml',
                 'elcm/large_area_sector/elcm1300093.yaml',
                 'elcm/large_area_sector/elcm200093.yaml',
                 'elcm/large_area_sector/elcm300161.yaml',
                 'elcm/large_area_sector/elcm1300125.yaml',
                 'elcm/large_area_sector/elcm800147.yaml',
                 'elcm/large_area_sector/elcm700125.yaml',
                 'elcm/large_area_sector/elcm800161.yaml',
                 'elcm/large_area_sector/elcm1000003.yaml',
                 'elcm/large_area_sector/elcm800099.yaml',
                 'elcm/large_area_sector/elcm1600093.yaml',
                 'elcm/large_area_sector/elcm200161.yaml',
                 'elcm/large_area_sector/elcm200099.yaml',
                 'elcm/large_area_sector/elcm500093.yaml',
                 'elcm/large_area_sector/elcm300003.yaml',
                 'elcm/large_area_sector/elcm1700003.yaml',
                 'elcm/large_area_sector/elcm900093.yaml',
                 'elcm/large_area_sector/elcm1400003.yaml',
                 'elcm/large_area_sector/elcm300005.yaml',
                 'elcm/large_area_sector/elcm1100005.yaml',
                 'elcm/large_area_sector/elcm900115.yaml',
                 'elcm/large_area_sector/elcm1500005.yaml',
                 'elcm/large_area_sector/elcm300147.yaml',
                 'elcm/large_area_sector/elcm1200099.yaml',
                 'elcm/large_area_sector/elcm1500125.yaml',
                 'elcm/large_area_sector/elcm500125.yaml',
                 'elcm/large_area_sector/elcm1200115.yaml',
                 'elcm/large_area_sector/elcm1500115.yaml',
                 'elcm/large_area_sector/elcm1100147.yaml',
                 'elcm/large_area_sector/elcm1200161.yaml',
                 'elcm/large_area_sector/elcm700093.yaml',
                 'elcm/large_area_sector/elcm800005.yaml',
                 'elcm/large_area_sector/elcm600005.yaml',
                 'elcm/large_area_sector/elcm1400115.yaml',
                 'elcm/large_area_sector/elcm1500147.yaml',
                 'elcm/large_area_sector/elcm1200125.yaml',
                 'elcm/large_area_sector/elcm100003.yaml',
                 'elcm/large_area_sector/elcm1600003.yaml',
                 'elcm/large_area_sector/elcm1600161.yaml',
                 'elcm/large_area_sector/elcm700161.yaml',
                 'elcm/large_area_sector/elcm800093.yaml',
                 'elcm/large_area_sector/elcm1000093.yaml',
                 'elcm/large_area_sector/elcm1100093.yaml',
                 'elcm/large_area_sector/elcm1000115.yaml',
                 'elcm/large_area_sector/elcm1400093.yaml',
                 'elcm/large_area_sector/elcm1300161.yaml',
                 'elcm/large_area_sector/elcm1400125.yaml',
                 'elcm/large_area_sector/elcm200115.yaml',
                 'elcm/large_area_sector/elcm1000147.yaml',
                 'elcm/large_area_sector/elcm1700093.yaml',
                 'elcm/large_area_sector/elcm100161.yaml',
                 'elcm/large_area_sector/elcm800003.yaml',
                 'elcm/large_area_sector/elcm1400099.yaml',
                 'elcm/large_area_sector/elcm1500099.yaml',
                 'elcm/large_area_sector/elcm1400005.yaml',
                 'elcm/large_area_sector/elcm1000099.yaml',
                 'elcm/large_area_sector/elcm1500161.yaml',
                 'elcm/large_area_sector/elcm1700115.yaml',
                 'elcm/large_area_sector/elcm1000125.yaml',
                 'elcm/large_area_sector/elcm600115.yaml',
                 'elcm/large_area_sector/elcm400161.yaml',
                 'elcm/large_area_sector/elcm1000161.yaml',
                 'elcm/large_area_sector/elcm1700005.yaml',
                 'elcm/large_area_sector/elcm1500093.yaml',
                 'elcm/large_area_sector/elcm1700161.yaml',
                 'elcm/large_area_sector/elcm1300099.yaml',
                 'elcm/large_area_sector/elcm100005.yaml',
                 'elcm/large_area_sector/elcm1700125.yaml',
                 'elcm/large_area_sector/elcm1600115.yaml',
                 'elcm/large_area_sector/elcm600161.yaml',
                 'elcm/large_area_sector/elcm600003.yaml',
                 'elcm/large_area_sector/elcm500115.yaml',
                 'elcm/large_area_sector/elcm200005.yaml',
                 'elcm/large_area_sector/elcm1100003.yaml',
                 'elcm/large_area_sector/elcm900005.yaml',
                 'elcm/large_area_sector/elcm400005.yaml',
                 'elcm/large_area_sector/elcm700147.yaml',
                 'elcm/large_area_sector/elcm300115.yaml',
                 'elcm/large_area_sector/elcm500099.yaml']


# ## Estimate a particular ELCM submodel

# In[ ]:


estimate_elcm(elcm_configs[0])

