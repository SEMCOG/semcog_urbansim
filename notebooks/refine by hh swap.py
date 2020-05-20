#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[2]:


hdf = pd.HDFStore("run592_hh_job_mon_refined_popfix.h5", complib='zlib', complevel=5)


# In[3]:


hdf


# In[5]:


refinement = pd.read_csv("MonForecastHHPOPRefinement.csv").set_index('b_city_id')


# In[6]:


refinement


# In[7]:


refinement.sum()


# In[8]:


refinement.columns.sort_values().unique()


# In[9]:


for year in refinement.columns.sort_values().unique():
    print("year", year)
    b_city_id = hdf[str(year) + '/buildings'][['b_city_id']]
    households = hdf[str(year) + '/households']
    households_col = households.columns
    households = pd.merge(households, b_city_id, left_on='building_id', right_index=True, how='left')
    pre = households[households.b_city_id.isin(refinement.index)].groupby('b_city_id').persons.sum()
    ref = refinement[year]
    gole = pre + ref
    while len(ref) > 0 and ref.abs().sum() > 0:
        ref = ref[ref != 0].copy()
        main_city = ref.abs().argmin()
        target = ref.loc[main_city]
        other_citys = ref[ref * target < 0]

        hh_main = households[households.b_city_id == main_city]
        hh_other = households[households.b_city_id.isin(other_citys.index)]
        
        sample_size = min(10 * abs(target), len(hh_main), len(hh_other))
        
        print(year, main_city, target, len(hh_main), len(hh_other), sample_size)

        hh_main_sample = hh_main.sample(sample_size)
        hh_other_sample = hh_other.sample(sample_size)
        
        corect_dir = (target * (hh_other_sample.persons.values - hh_main_sample.persons.values) > 0)
        hh_main_sample = hh_main_sample[corect_dir]
        hh_other_sample = hh_other_sample[corect_dir]
        
        corect_size = (abs(hh_other_sample.persons.values - hh_main_sample.persons.values) <= abs(target))
        hh_main_sample = hh_main_sample[corect_size]
        hh_other_sample = hh_other_sample[corect_size]
        
        fit = abs(hh_other_sample.persons.values - hh_main_sample.persons.values).cumsum() <= abs(target)
        hh_main_sample = hh_main_sample[fit]
        hh_other_sample = hh_other_sample[fit]

        households.loc[hh_main_sample.index, 'building_id'] = hh_other_sample.building_id.values
        households.loc[hh_main_sample.index, 'b_city_id'] = hh_other_sample.b_city_id.values
        households.loc[hh_other_sample.index, 'building_id'] = hh_main_sample.building_id.values
        households.loc[hh_other_sample.index, 'b_city_id'] = hh_main_sample.b_city_id.values
        
        post = households[households.b_city_id.isin(refinement.index)].groupby('b_city_id').persons.sum()
        
        ref = (gole - post).astype(int)
    
    hdf[str(year) + '/households'] = households[households_col]
    print("done")


# In[10]:


hdf.close()


# In[ ]:




