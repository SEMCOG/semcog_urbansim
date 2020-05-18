#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#changing HH pop at large area level, NOT swapping


# In[ ]:


import pandas as pd


# In[ ]:


st = pd.HDFStore("run592_hh_outway_refined_test.h5", complib='zlib', complevel=5)


# In[ ]:


st


# In[ ]:


refinement = pd.read_csv("MonForecastHHPOPRefinement.csv").set_index('b_city_id')


# In[ ]:


refinement


# In[ ]:


refinement.sum()


# In[ ]:


refinement.loc[5070, '2020']


# In[ ]:


for ind in refinement[str(2020)].index:
    print ind


# In[ ]:


for year in ['2020', '2025', '2030', '2035', '2040', '2045']:
    dfh = pd.merge(st['/' + year + '/households'],
                   st['/' + year + '/buildings'][['b_city_id']], 
                   left_on='building_id',right_index=True, how='left')
    
    dfp = pd.merge(st['/' + year + '/persons'],
                   dfh[['b_city_id']], 
                   left_on='household_id',right_index=True, how='left')
    
    for city_id in refinement[year].index:
        dfpc = dfp.loc[(dfp.b_city_id == city_id) &  (dfp.relate>0)]
        amount = refinement.loc[city_id, year]
        pre = len(dfpc)
        
        print city_id, 'before: ', pre,  'add:', amount
        
        if amount >0:
            p_add = dfpc.sample(amount)
            p_add.index=range(dfp.index.max()+1, dfp.index.max()+1+len(p_add))
            dfp = pd.concat([dfp,p_add])
        elif amount<0:
            p_remove = dfpc.sample(amount).index
            dfp.drop(p_remove, inplace=True)
            
    
   
    dfh['persons'] = dfp.groupby('household_id').size()
    dfh['workers'] = dfp.groupby('household_id').worker.sum()
    dfh.fillna(0, inplace=True)
    print 'after:', len(dfp), dfh.persons.sum()
    
    st['/' + year + '/households'] = dfh
    st['/' + year + '/persons'] =  dfp


# In[ ]:


st.close()


# In[ ]:





# In[ ]:


refinement


# In[ ]:


refinement.drop([5015,5020], inplace=True)


# In[ ]:


refinement


# In[ ]:





# In[ ]:


refinement.index.max()


# In[ ]:


np.append([],refinement.index)


# In[ ]:


refinement.index


# In[ ]:


pd.concat(refinement.index.values, refinement.index.values)


# In[ ]:


refinement.loc[np.append([],refinement.index)]


# In[ ]:


dfp.index.max()


# In[ ]:





# In[ ]:


addp.index=[dfp.index.max()+x for x in range(len(addp))]


# In[ ]:


dfp = pd.concat([dfp,addp])


# In[ ]:


added.index=[1633176426+x for x in range(len(added))]


# In[ ]:





# In[ ]:


hhpersons = dfp.groupby('household_id').size()
workers = dfp.groupby('household_id').worker.sum()


# In[ ]:


dfh = st['/2015/households'].copy()
dfh['persons'] = 0
dfh['persons'] = hhpersons
dfh['workers'] = 0
dfh['workers'] = hhpersons


# In[ ]:





# In[ ]:


1633176426+range(10)


# In[ ]:





# In[ ]:


refinement.columns.sort_values().unique()


# In[ ]:


for year in refinement.columns.sort_values().unique():
    print "year", year
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
        
        print year, main_city, target, len(hh_main), len(hh_other), sample_size

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
    print "done"


# In[ ]:


hdf.close()


# In[ ]:




