#!/usr/bin/env python
# coding: utf-8

# # Residential Price Model

# ## Imports

# In[1]:


import numpy as np
import pandas as pd


# ## Merge

# In[2]:


def merge_buildings_parcels(buildings, parcels):
    return pd.merge(buildings, parcels, left_on='parcel_id', right_index=True)


# ## Calculations

# These functions add new columns to the buildings table by doing
# calculations and pulling data from other tables.
# They all modify the buildings table in-place but also return the
# buildings table for consistency with operations (like merges) that
# return entirely new tables.

# In[3]:


def unit_price_res_column(buildings):
    """
    Calculate residential unit price as improvement_value per residential unit.
    
    """
    buildings['unit_price_res'] = buildings.improvement_value / buildings.residential_units
    buildings['unit_price_res'][buildings['residential_units'] == 0] = 0
    return buildings


# In[4]:


def population_density(buildings, households):
    """
    Calculate population density at the zonal level as people per acre
    then broadcast out to the building level.
    
    """
    sqft_per_acre = 43560
    
    bldg_pop = households.groupby('building_id').persons.sum()
    zone_pop = bldg_pop.groupby(buildings.zone_id).sum()
    zone_acres = buildings.parcel_sqft.groupby(buildings.zone_id).sum() / sqft_per_acre
    pop_density = (zone_pop / zone_acres).fillna(0)
    buildings['popden'] = pd.Series(pop_density[buildings.zone_id].values, 
                                    index=buildings.index)
    return buildings


# In[5]:


def crime_rate(buildings, cities):
    """
    Broadcast crime rate from the cities table to buildings.
    
    """
    buildings['crime08'] = cities.crime08[buildings.city_id].values
    return buildings


# In[6]:


def jobs_within_30_min(buildings, travel_data, jobs):
    """
    Calculate the number of jobs within thirty minutes of each building.
    This is actually done at the zonal level and then broadcast
    to buildings.
    
    """
    # The travel_data table has a multi-level index with from_zone_id
    # and to_zone_id. We care about the travel time between zones so
    # we want to move the to_zone_id into the DataFrame as a regular column
    # and then keep all the zone pairs that are less than 30 minutes apart
    zones_within_30_min = (travel_data.reset_index(level='to_zone_id')
                           .query('am_single_vehicle_to_work_travel_time < 30').to_zone_id)

    # The next step is to tabulate the number of jobs in each zone,
    # broadcast that across the zones within range of each other zone,
    # and finally group by the from_zone_id and total all the jobs within range.
    job_counts = jobs.groupby('zone_id').size()
    job_counts = pd.Series(
        job_counts[zones_within_30_min].fillna(0).values, 
        index=zones_within_30_min.index).groupby(level=0).sum()
    buildings['jobs_within_30_min'] = job_counts[buildings.zone_id].fillna(0).values
    return buildings


# ## Pull Data, Apply Changes

# In[7]:


data_store = pd.HDFStore('./data/semcog_data.h5', mode='r')


# Putting all the calls to the data transformation functions in
# one cell allows me to easily re-run all the transformations when I
# change something or re-open the notebook.

# In[8]:


buildings = merge_buildings_parcels(data_store['buildings'], data_store['parcels'])

buildings = unit_price_res_column(buildings)
buildings = population_density(buildings, data_store['households'])
buildings = crime_rate(buildings, data_store['cities'])
buildings = jobs_within_30_min(buildings, data_store['travel_data'], data_store['jobs'])


# ## Residential Unit Price Regression

# In[9]:


from urbansim.models import RegressionModelGroup


# The [patsy](http://patsy.readthedocs.org/en/latest/) expression defined below
# is the crux of the regression models.
# The terms in `patsy_exp` are are combined to express a model that's a
# combination of all the terms.
# 
# The `I(...)` construction is used to embed regular Python in the model
# expression. Here `I(year_built < 1940)` results in a boolean column that
# flags whether a column was made before 1940.
# The ability to do this saves us from having to pre-populate the buildings
# table with such a column.
# 
# Note that columns can also be transformed by any functions that are available in
# the namespace in which the patsy expression is evaluated.
# In UrbanSim [NumPy](http://www.numpy.org/) will always be available as `np`
# and [Pandas](http://pandas.pydata.org/) will always be available as `pd`.

# In[10]:


def patsy_expression():
    patsy_exp = ['I(year_built < 1940)',
                 'year_built',
                 'stories',
                 'np.log1p(sqft_per_unit)',
                 'np.log1p(popden)',
                 'dist_hwy',
                 'dist_road',
                 'crime08',
                 'np.log1p(jobs_within_30_min)']
    patsy_exp = ' + '.join(patsy_exp)
    return 'np.log(unit_price_res) ~ ' + patsy_exp
model_expression = patsy_expression()


# Filters are combined with ` and ` and passed to the Pandas
# [query](http://pandas.pydata.org/pandas-docs/stable/indexing.html#the-query-method-experimental)
# method.

# In[11]:


estimate_filters = ['residential_units > 0',
                    'sqft_per_unit > 0',
                    'year_built > 1700',
                    'stories > 0',
                    'tax_exempt == 0',
                    '1e5 < unit_price_res < 1e7',
                    '16 <= building_type_id <= 20']


# In[12]:


simulate_filters = ['residential_units > 0',
                    '16 <= building_type_id <= 20']


# The `RegressionModelGroup` is a convenience for interfacing with several
# `RegressionModel`s that operate on different segments of the same table.
# With `RegressionModelGroup` all of the segment regressions can be defined
# separately and then called at the same time on the same table.

# In[13]:


group_keys = [16, 17, 18, 19, 20]
hmg = RegressionModelGroup('building_type_id')
for key in group_keys:
    hmg.add_model_from_params(key, estimate_filters, simulate_filters,
                              model_expression, ytransform=np.exp)


# In[14]:


fits = hmg.fit(buildings)


# In[15]:


fits[16].summary()


# In[16]:


new_unit_price_res = hmg.predict(buildings)


# In[17]:


new_unit_price_res


# # Residential LCM

# Currently this is only for demo and test purposes.

# In[9]:


from urbansim.models import MNLLocationChoiceModel


# In[10]:


# printed as a convenience
buildings.columns.tolist()


# In[11]:


# printed as a convenience
data_store['households'].columns.tolist()


# This patsy expression is similar to the one defined above with one
# additional feature: here we're using patsy's `:` operator to make
# the model depend on the *interaction* of two columns.
# Again, this saves us from having to calculate that beforehand and
# add it to a table or somehow define it separately.

# In[12]:


patsy_expression = ['sqft_per_unit',
                    'np.log1p(unit_price_res)',
                    'income:unit_price_res']
patsy_expression = ' + '.join(patsy_expression)


# In[13]:


estimation_filters = ['building_id > 0']


# In[14]:


households = data_store['households']


# In[15]:


# Not passing in any simulation filters here because I'm only
# experimenting with estimation.
lcm = MNLLocationChoiceModel(
    estimation_filters, None, patsy_expression, 10, 
    choice_column='building_id', name='Test LCM')


# In[16]:


ll = lcm.fit(households, buildings, households.building_id)


# In[17]:


# log-likelihood numbers
ll


# In[18]:


# model coefficients and other info
lcm.fit_results


# In[19]:


lcm.report_fit()


# ## LCM Simulation

# For simulation we need two tables: choosers and alternatives.
# For demo purposes here choosers will be a random subset of households.

# In[20]:


# choose some random households for demonstration
hids = np.random.choice(households.index, size=1000, replace=False)


# In[21]:


def residential_units_table(households, buildings):
    buildings = buildings.query('residential_units > 0')
    vacant_units_per_building = buildings.residential_units.subtract(
        households.groupby('building_id').size(), fill_value=0)
    vacant_units_per_building = vacant_units_per_building[vacant_units_per_building > 0]
    repeated_index = vacant_units_per_building.index.repeat(
        vacant_units_per_building.astype('int'))
    return buildings.loc[repeated_index].reset_index()


# In[22]:


new_assignments = lcm.predict(
    households.loc[hids], residential_units_table(households, buildings))


# In[23]:


new_assignments


# In[ ]:




