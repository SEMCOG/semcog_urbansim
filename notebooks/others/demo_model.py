#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd, numpy as np
from urbansim.models import transition
import orca
from urbansim.utils import misc


# #### Import the simulation framework

# In[ ]:


#import urbansim.sim.simulation as sim


# #### Connect to data store, register it as an injectable

# In[2]:


semcog_hdfstore = pd.HDFStore("data/semcog_data.h5", mode="r")
orca.add_injectable("store", semcog_hdfstore)


# In[ ]:


semcog_hdfstore


# #### Register data tables that will be used in the simulation

# In[3]:


@orca.table('jobs') 
def jobs(store): 
    df = store['jobs']
    return df

@orca.table('buildings') 
def jobs(store): 
    df = store['buildings']
    return df

@orca.table('annual_employment_control_totals')
def annual_employment_control_totals(store):
    df = store['annual_employment_control_totals']
    return df


# In[5]:


buildings = orca.get_table('buildings').to_frame()


# In[10]:


buildings.duplicated().any()


# #### Register your models

# In[ ]:


@orca.step('jobs_transition')
def jobs_transition(jobs, annual_employment_control_totals,iter_var): 
    ct_emp = annual_employment_control_totals.to_frame()
    ct_emp = ct_emp.reset_index().set_index('year')
    tran = transition.TabularTotalsTransition(ct_emp, 'total_number_of_jobs')
    model = transition.TransitionModel(tran)
    j = jobs.to_frame(jobs.local_columns)
    #print "j"
    new, added_jobs_idx, new_linked = model.transition(j, iter_var)
    new.loc[added_jobs_idx, "building_id"] = -1
    orca.add_table("jobs", new)
    
@orca.step('print_year')
def print_year(iter_var):
    print('*** the year is {} ***'.format(iter_var))


# #### A demonstration of running the above models

# In[ ]:


#jobs = orca.get_table('jobs')
#print jobs.to_frame().describe()


# In[ ]:


jobs.to_frame().head()


# In[ ]:


orca.run(["print_year","jobs_transition"], iter_vars=[2010], data_out='runs/test_run.h5', out_interval=1)


# In[ ]:


orca.get_table('jobs').to_frame().describe()


# In[ ]:


print(orca._TABLES)
print(orca._MODELS)
print(orca._INJECTABLES)

##show add_injectable and how it updates the _INJECTABLES list, then show the dictionaries up at top of sim (empty dicts)


# In[ ]:




