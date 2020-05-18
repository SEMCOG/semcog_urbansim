#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import numpy as np
import pandas as pd
from urbansim.models import MNLLocationChoiceModel
from urbansim.models import RegressionModelGroup


# ## Variable Calculations

# In[2]:


def merge_buildings_parcels(buildings, parcels):
    return pd.merge(buildings, parcels, left_on='parcel_id', right_index=True)

def unit_price_res_column(buildings):
    """
    Calculate residential unit price as improvement_value per residential unit.
    
    """
    buildings['unit_price_res'] = buildings.improvement_value / buildings.residential_units
    buildings['unit_price_res'][buildings['residential_units'] == 0] = 0
    return buildings

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

def crime_rate(buildings, cities):
    """
    Broadcast crime rate from the cities table to buildings.
    
    """
    buildings['crime08'] = cities.crime08[buildings.city_id].values
    return buildings

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

##Table of vacant residential units
def residential_units_table(households, buildings):
    buildings = buildings.query('residential_units > 0')
    vacant_units_per_building = buildings.residential_units.subtract(
        households.groupby('building_id').size(), fill_value=0)
    vacant_units_per_building = vacant_units_per_building[vacant_units_per_building > 0]
    repeated_index = vacant_units_per_building.index.repeat(
        vacant_units_per_building.astype('int'))
    return buildings.loc[repeated_index].reset_index()

##Table of job spaces
def job_spaces_table(jobs, buildings):
    buildings = buildings.query('job_spaces > 0')
    vacant_units_per_building = buildings.job_spaces.subtract(
        jobs.groupby('building_id').size(), fill_value=0)
    vacant_units_per_building = vacant_units_per_building[vacant_units_per_building > 0]
    repeated_index = vacant_units_per_building.index.repeat(
        vacant_units_per_building.astype('int'))
    return buildings.loc[repeated_index].reset_index()

data_store = pd.HDFStore('./data/semcog_data.h5', mode='r')

buildings = merge_buildings_parcels(data_store['buildings'], data_store['parcels'])

buildings = unit_price_res_column(buildings)
buildings = population_density(buildings, data_store['households'])
buildings = crime_rate(buildings, data_store['cities'])
buildings = jobs_within_30_min(buildings, data_store['travel_data'], data_store['jobs'])

parcels = data_store['parcels']
households = data_store['households']

vacant_residential_units = residential_units_table(households, buildings)
buildings = pd.merge(buildings,data_store['building_sqft_per_job'],left_on=['zone_id','building_type_id'],right_index=True,how='left')
buildings['job_spaces'] = buildings.non_residential_sqft/buildings.building_sqft_per_job
buildings['job_spaces'] = np.round(buildings['job_spaces'].fillna(0)).astype('int')
jobs = data_store['jobs']

vacant_job_spaces = job_spaces_table(jobs, buildings)


# ## Household Location Choice Model (HLCM)

# ### Estimation

# In[3]:


#Agents for estimation
households_for_estimation = households.loc[np.random.choice(households.index, size=15000, replace=False)]

#Filters on agents
agent_estimation_filters = ['building_id > 0']

agents_simulation_filters = None

##Specification
patsy_expression = ['sqft_per_unit',
                    'income:np.log1p(unit_price_res)',
                    'jobs_within_30_min',
                    'crime08',
                    'popden']
patsy_expression = ' + '.join(patsy_expression)

#Filters on alternatives
estimation_filters = ['building_id > 0']

simulation_filters = ['residential_units>0',
                      'building_type_id in [16,17,18,19,20]']

interaction_filters = ['income*5 > unit_price_res']

##Instantiate HCLM
hlcm = MNLLocationChoiceModel(
    patsy_expression, 10,
    choosers_fit_filters=agent_estimation_filters, choosers_predict_filters=agents_simulation_filters,
    alts_fit_filters=estimation_filters, alts_predict_filters=simulation_filters,interaction_predict_filters=interaction_filters,
    choice_column='building_id', name='HLCM')

##Estimate
hlcm.fit(households_for_estimation, buildings, households_for_estimation.building_id)
hlcm.report_fit()


# ### Simulation

# In[4]:


#Agents for simulationm
hids = np.random.choice(households.index, size=1000, replace=False)

#Simulate
new_assignments = hlcm.predict(households.loc[hids], vacant_residential_units)
print new_assignments


# ## Employment Location Choice Model (ELCM)

# ### Estimation

# In[5]:


#Agents for estimation
jobs_for_estimation = data_store['jobs_for_estimation']

#Filters on agents
agent_estimation_filters = ['home_based_status == 0']

agents_simulation_filters = None

##Specification
patsy_expression = ['np.log1p(non_residential_sqft)',
                    'np.log1p(improvement_value)',
                    'jobs_within_30_min',
                    'popden',
                    'crime08']
patsy_expression = ' + '.join(patsy_expression)

#Filters on alternatives
estimation_filters = ['building_id > 0']

simulation_filters = ['non_residential_sqft>0',
                      'building_type_id not in [16,17,18,19,20]']

##Instantiate ELCM
elcm = MNLLocationChoiceModel(
    patsy_expression, 10,
    choosers_fit_filters=agent_estimation_filters, choosers_predict_filters=agents_simulation_filters,
    alts_fit_filters=estimation_filters, alts_predict_filters=simulation_filters,
    choice_column='building_id', name='ELCM')

##Estimate
elcm.fit(jobs_for_estimation, buildings, jobs_for_estimation.building_id)
elcm.report_fit()


# ### Simulation

# In[6]:


#Agents for simulation
jids = np.random.choice(jobs.index, size=1000, replace=False)

#Simulate
new_assignments = elcm.predict(jobs.loc[jids], vacant_job_spaces)
print new_assignments


# ## Development Project Location Choice Model (DPLCM)

# ### Estimation

# In[7]:


#Agents for estimation
buildings

#Filters on agents
agent_estimation_filters = ['year_built > 2004']

agents_simulation_filters = None

##Specification
patsy_expression = ['np.log1p(dist_hwy)',
                    'np.log1p(land_value)',
                    'floodprone']
patsy_expression = ' + '.join(patsy_expression)

#Filters on alternatives
estimation_filters = ['parcel_id > 0']

simulation_filters = ['parcel_sqft > 10000']

##Instantiate DPLCM
dplcm = MNLLocationChoiceModel(
    patsy_expression, 10,
    choosers_fit_filters=agent_estimation_filters, choosers_predict_filters=agents_simulation_filters,
    alts_fit_filters=estimation_filters, alts_predict_filters=simulation_filters, 
    choice_column='parcel_id', name='DPLCM')

##Estimate
dplcm.fit(buildings, parcels, buildings.parcel_id)
dplcm.report_fit()


# ### Simulation

# In[8]:


#Agents for simulation
bids = np.random.choice(buildings.index, size=1000, replace=False)

#Simulate
new_assignments = dplcm.predict(buildings.loc[bids], parcels.reset_index())
print new_assignments


# ## Real Estate Price Model (REPM)

# ### Estimation

# In[9]:


##Model specification
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

##Estimation filters
estimate_filters = ['residential_units > 0',
                    'sqft_per_unit > 0',
                    'year_built > 1700',
                    'stories > 0',
                    'tax_exempt == 0',
                    '1e5 < unit_price_res < 1e7',
                    '16 <= building_type_id <= 20']

##Simulation filters
simulate_filters = ['residential_units > 0',
                    '16 <= building_type_id <= 20']

##Segmentation
group_keys = [16, 17, 18, 19, 20]
hmg = RegressionModelGroup('building_type_id')
for key in group_keys:
    hmg.add_model_from_params(key, estimate_filters, simulate_filters,
                              model_expression, ytransform=np.exp)
    
##Estimate
fits = hmg.fit(buildings)


# ### Simulation

# In[10]:


##Simulate
new_unit_price_res = hmg.predict(buildings)
print new_unit_price_res


# ## Additional Examples (for illustration only)

# In[ ]:


##Instantiate location choice models
hlcm = MNLLocationChoiceModel(
    estimation_filters, simulation_filters, patsy_expression, 10, 
    choice_column='building_id', name='HLCM')

elcm = MNLLocationChoiceModel(
    estimation_filters, simulation_filters, patsy_expression, 10, 
    choice_column='building_id', name='ELCM')

dplcm = MNLLocationChoiceModel(
    estimation_filters, simulation_filters, patsy_expression, 10, 
    choice_column='parcel_id', name='DPLCM')

##Estimate
lcm.fit(agents_for_estimation, alternatives, chosen_alternatives)

##Simulate
lcm.predict(agents_for_simulation, alternatives)


# In[ ]:


hlcm.name


# In[ ]:


hlcm.fitted


# In[ ]:


hlcm.coefficients


# In[ ]:


hlcm.report_fit()

