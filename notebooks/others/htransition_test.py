#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from urbansim.utils import misc
import orca


# In[2]:


orca.add_injectable("store", pd.HDFStore(os.path.join(misc.data_dir(),
                                                     "semcog_data.h5"), mode="r"))


# In[4]:


@orca.table('annual_household_control_totals')
def annual_household_control_totals(store):
    df = store['annual_household_control_totals']
    return df

@orca.table('households')
def households(store):
    df = store['households']
    return df
    
@orca.table('persons')
def persons(store):
    df = store['persons']
    return df


# In[10]:


import numpy as np, pandas as pd
from urbansim.models import transition, relocation
@orca.step('households_transition')
def households_transition(households, persons, annual_household_control_totals, iter_var):
    ct = annual_household_control_totals.to_frame()
    for col in ct.columns:
        i = 0
        if col.endswith('_max'):
            if len(ct[col][ct[col]==-1]) > 0:
                ct[col][ct[col]==-1] = np.inf
                i+=1
            if i > 0:
                ct[col] = ct[col] + 1
    tran = transition.TabularTotalsTransition(ct, 'total_number_of_households')
    model = transition.TransitionModel(tran)
    hh = households.to_frame(households.local_columns)
    p = persons.to_frame(persons.local_columns)
    new, added_hh_idx, new_linked =         model.transition(hh, iter_var,
                         linked_tables={'linked': (p, 'household_id')})
    new.loc[added_hh_idx, "building_id"] = -1
    orca.add_table("households", new)
    orca.add_table("persons", new_linked['linked'])


# In[11]:


orca.run([
    "households_transition",  # households transition
#     "households_relocation",  # households relocation model
], iter_vars=range(2012, 2014))


# In[13]:


print orca.get_injectable('iter_var')
hh_sim = orca.get_table('households').to_frame()

store = orca.get_injectable('store')
hh_base = store['households']


# In[ ]:




