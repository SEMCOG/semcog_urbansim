#!/usr/bin/env python
# coding: utf-8

# In[1]:


import orca
import pandas as pd
from urbansim.models import MNLDiscreteChoiceModel

import models, utils, lcm_utils
from urbansim.utils import misc, networks
import output_indicators


# In[2]:


import numpy as np
from collections import OrderedDict
from choicemodels import MultinomialLogit
from choicemodels.tools import MergedChoiceTable


# In[3]:


buildings = pd.HDFStore('building_store.h5').buildings

buildings_nonres = buildings[buildings.non_residential_sqft > 0]

jobs = orca.get_table('jobs').to_frame(['sector_id', 'building_id', 'home_based_status'])

jobs.building_id = jobs.building_id.astype('int')

buildings_res = buildings[buildings.residential_units > 0]

hh = orca.get_table('households')
hh_vars = hh.local_columns + ['income_quartile', 'is_race4', 'is_race1', 'is_race2', 'ln_income',
                              'hhsize_lt_3', 'is_young', 'has_children']
hh = hh.to_frame(hh_vars)


# In[8]:


model_configs = lcm_utils.get_model_category_configs()

for model_category_name, model_category_attributes in model_configs.items():
    if model_category_name == 'elcm':
        for yaml_config in model_category_attributes['config_filenames']:
            print yaml_config
            model = MNLDiscreteChoiceModel.from_yaml(str_or_buffer=misc.config(yaml_config))
            
            # Patsy-form specification
            patsy_str = ' + '
            patsy_str = patsy_str.join(model.model_expression) + ' - 1'
            print patsy_str
            
            # Pylogit-form specification
            vars_for_dict = OrderedDict([(varname, 'all_same') for varname in model.model_expression])
            
            submodel_id = int(yaml_config.split('elcm')[-1].split('.')[0])
            
            choosers = jobs[(jobs.sector_id == submodel_id) & (jobs.home_based_status == 0)]

            choosers = choosers.loc[np.random.choice(
                choosers.index,
                3000, replace=False)]
            
            chosen = choosers['building_id']
            alternatives = buildings_nonres

            data = MergedChoiceTable(observations = choosers,
                                     alternatives = alternatives,
                                     chosen_alternatives = chosen,
                                     sample_size = 15)
            
            model = MultinomialLogit(data = data.to_frame(),
                         observation_id_col = data.observation_id_col, 
                         choice_col = data.choice_col,
                         model_expression = vars_for_dict,
                         alternative_id_col='building_id') #patsy_str
            print model._estimation_engine
            
            results = model.fit()
            print results
            print ''

