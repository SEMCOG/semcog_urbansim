#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time
import os
import random
#from urbansim.models import transition, relocation
from urbansim.developer import sqftproforma, developer
from urbansim.utils import misc, networks
#import dataset, variables, utils, transcad
import dataset, variables, utils
import pandana as pdna
import models

import orca


# In[2]:


dfh=orca.get_table('households').to_frame(orca.get_table('households').local_columns + ['large_area_id'])


# In[4]:


dfh.to_csv('syn_households.csv')


# In[5]:


dfp=orca.get_table('persons').to_frame(orca.get_table('persons').local_columns + ['large_area_id'])


# In[6]:


dfp.to_csv('syn_persons.csv')


# In[2]:


orca.list_tables()


# In[4]:


orca.get_table('target_vacancies').to_frame()


# In[ ]:





# In[ ]:





# In[54]:


sjb = dfj.groupby('building_id').size()


# In[55]:


sjb.name='jobs'


# In[62]:


b4jobs = pd.merge(dfb, sjb.to_frame(), left_index=True, right_index=True, how='left').fillna(0)


# In[63]:


b4jobs.to_csv('b4jobs.csv')


# In[44]:


orca.list_tables()


# In[64]:


orca.get_table('parcels').to_frame(orca.get_table('parcels').local_columns)


# In[39]:


dd=dfh.reset_index()


# In[40]:


dd.loc[dd.large_area_id>100, ['household_id','large_area_id']]


# In[15]:


dfp=orca.get_table('persons').to_frame()[['household_id','age','sex','race_id']]


# In[18]:


pd.merge(dfp,dfh, left_on='household_id', right_index=True, how='left').to_csv('syn_persons_4control.csv')


# In[31]:


for ld, hh in dfh.groupby('large_area_id'):
    print hh.head(20)
    print '--'


# In[ ]:


orca.run(["build_networks"])
orca.run(["neighborhood_vars"])


# In[ ]:


orca.get_table('households').to_frame()


# In[ ]:





# # Variables converted from OPUS(urbansim1) to ORCA(urbansim2)

# In[ ]:



#building type dummy variables (derived from building types table,we have 27 building types)
df=pd.get_dummies(orca.merge_tables(target='buildings', tables=['building_types', 'buildings'], 
                            columns=["building_type_name "])["building_type_name "]).astype(bool)
df.columns = ["type_is_" + i.strip().replace(" ",'_') for i in df.columns]


B_building_age = 2015 - orca.get_table('buildings')['year_built']  #in simulation , use iter_var instead of 2015
B_is_new_construction = B_building_age <= 2
B_is_pre1950 = orca.get_table('buildings')['year_built'] < 1950
B_ln_sqft_per_unit=np.log1p(orca.get_table('buildings')['sqft_per_unit']) #filter out 0s?

@orca.column('buildings')
def lot_sqft_per_unit(parcels, buildings):
    b = buildings.to_frame(["parcel_id", "residential_units"])
    rh_per_p = b.groupby("parcel_id").residential_units.sum()
    sqft_per_rh = parcels.to_frame("parcel_sqft").parcel_sqft / rh_per_p
    out = misc.reindex(sqft_per_rh, b.parcel_id)
    return out.fillna(1).replace(np.inf, 1)


B_ln_lot_size_per_unit = np.log(orca.get_table('buildings')['lot_sqft_per_unit'])
B_ln_price_per_unit = np.log(orca.get_table('buildings')['sqft_price_res'])
B_ln_price_per_sqft = np.log1p((orca.get_table('buildings')['sqft_price_res']/orca.get_table('buildings')['sqft_per_unit']))
B_ln_residential_units = np.log(orca.get_table('buildings')['residential_units'])
B_ln_vacant_residential_units = np.log(orca.get_table('buildings')['vacant_residential_units'])

B_far=orca.merge_tables(target='buildings', tables=['parcels', 'buildings'], columns=["parcel_far"])["parcel_far"]
B_ln_invfar = -np.log(B_far)
B_school_district_achievement = orca.get_table('buildings')['school_district_achievement']


# segment or dumy by large area

Z_jobs = orca.get_table('zones')['employment']
Z_households = orca.get_table('zones')['households']
Z_ln_households = np.log(Z_households)
Z_population =orca.get_table('zones')['population']
Z_ln_population = np.log(Z_population)
Z_ln_empden = np.log(orca.get_table('zones')['empden'])
Z_ln_popden = np.log(orca.get_table('zones')['popden'])

## To be ramade as pandana baset n variables
N_ln_avginc = orca.merge_tables(target='buildings', tables=['nodes_walk', 'buildings'], 
                             columns=["ave_income"])["ave_income"]

N_households = orca.get_table('nodes_walk')['households']
N_ln_households = np.log(N_households)
N_population = orca.get_table('nodes_walk')['population']
N_ln_population = np.log(N_population)
N_jobs = orca.get_table('nodes_walk')['jobs']
node_r1500_acre = orca.get_table('nodes_walk')['node_r1500_sqft'] / 43560.

N_ln_empden = np.log1p(N_jobs / node_r1500_acre)
N_ln_popden = np.log1p(N_population / node_r1500_acre)

N_percent_high_income = orca.get_table('nodes_walk')['highinc_hhs'] / N_households
N_percent_mid_income = orca.get_table('nodes_walk')['midinc_hhs'] / N_households
N_percent_low_income = orca.get_table('nodes_walk')['lowinc_hhs'] / N_households

N_percent_race1=N_percent_mid_income = orca.get_table('nodes_walk')['race_1_hhs'] / N_households
N_percent_race2=N_percent_mid_income = orca.get_table('nodes_walk')['race_2_hhs'] / N_households
N_percent_race3=N_percent_mid_income = orca.get_table('nodes_walk')['race_3_hhs'] / N_households
N_percent_race4=N_percent_mid_income = orca.get_table('nodes_walk')['race_4_hhs'] / N_households

N_percent_hh_with_children = orca.get_table('nodes_walk')['hhs_with_children'] / N_households

N_ln_average_zonal_income = np.log(orca.get_table('nodes_walk')['ave_income'])

# dumy variabal per mcd

C_crime_ucr = orca.get_table('parcels')['crime_ucr_rate']
C_crime_other = orca.get_table('parcels')['crime_other_rate']

#accessibility based on zone-to-zone travel matrix
@orca.column('zones')
def A_ln_emp_26min_drive_alone(zones, travel_data):
    drvtime = travel_data.to_frame("am_auto_total_time").reset_index()
    zemp = zones.to_frame('employment')
    temp = pd.merge(drvtime,zemp, left_on = 'to_zone_id', right_index = True, how='left' )
    return np.log1p(temp[temp.am_auto_total_time <=26].groupby('from_zone_id').employment.sum().fillna(0))

@orca.column('zones')
def A_ln_emp_50min_transit(zones, travel_data):
    transittime = travel_data.to_frame("am_transit_total_time").reset_index()
    zemp = zones.to_frame('employment')
    temp = pd.merge(transittime,zemp, left_on = 'to_zone_id', right_index = True, how='left' )
    return np.log1p(temp[temp.am_transit_total_time <=50].groupby('from_zone_id').employment.sum().fillna(0))

@orca.column('zones')
def A_ln_retail_emp_15min_drive_alone(zones, travel_data):
    drvtime = travel_data.to_frame("midday_auto_total_time").reset_index()
    zemp = zones.to_frame('employment')
    temp = pd.merge(drvtime,zemp, left_on = 'to_zone_id', right_index = True, how='left' )
    return np.log1p(temp[temp.midday_auto_total_time <=15].groupby('from_zone_id').employment.sum().fillna(0))


A_job_logsum_high_income=orca.merge_tables(target='buildings', tables=['zones', 'parcels', 'buildings'], 
                            columns=["logsum_pop_high_income"])["logsum_pop_high_income"]
A_job_logsum_low_income=orca.merge_tables(target='buildings', tables=['zones', 'parcels', 'buildings'], 
                            columns=["logsum_pop_low_income"])["logsum_pop_low_income"]
A_pop_logsum_high_income=orca.merge_tables(target='buildings', tables=['zones', 'parcels', 'buildings'], 
                            columns=["logsum_pop_high_income"])["logsum_pop_high_income"]
A_pop_logsum_low_income=orca.merge_tables(target='buildings', tables=['zones', 'parcels', 'buildings'], 
                            columns=["logsum_pop_low_income"])["logsum_pop_low_income"]

N_retail_jobs = orca.get_table('nodes_walk')['retail_jobs']



# ========================
P_property_tax=orca.get_table('parcels')['pptytax']


# In[ ]:


orca.get_table('zones').to_frame()


# # Variables haven't been converted from OPUS(urbansim1) to ORCA(urbansim2)

# In[ ]:


## income
 I_disposable_inc = ln_bounded(household.income - (urbansim_parcel.building.unit_price/5.))
 "I_ln_income_less_price_per_unit = ln_bounded(household.income - ((urbansim_parcel.building.unit_price/10.) * urbansim_parcel.building.building_sqft_per_unit))",  
 #"I_ln_income_less_price_per_unit_x_is_condo_residential = ln_bounded(household.income - ((urbansim_parcel.building.unit_price/10.) * urbansim_parcel.building.building_sqft_per_unit)) * urbansim.building.is_condo_residential", # 
 "I_ln_income_less_price_per_unit_x_is_multi_family_residential = ln_bounded(household.income - (urbansim_parcel.building.unit_price/5.)) * washtenaw.building.is_multi_family_residential", # 
 "I_ln_income_less_price_per_unit_x_is_single_family_residential = ln_bounded(household.income - (urbansim_parcel.building.unit_price/5.)) * washtenaw.building.is_single_family_residential", I_ln_income_x_is_new_construction = ln(household.income) * (urbansim_parcel.building.age_masked < 2)
 I_ln_income_x_is_pre1945 = ln(household.income) * (urbansim_parcel.building.age_masked > 60)
 I_ln_income_x_is_single_family_residential = ln(household.income) * washtenaw.building.is_single_family_residential
 I_ln_income_x_ln_average_zonal_income = ln(household.income) * ln(building.disaggregate(urbansim_parcel.zone.average_income))
 I_ln_income_x_ln_lot_size_per_unit = ln(household.income) * ln((building.disaggregate(parcel.parcel_sqft)) / building.residential_units)
 I_ln_income_x_ln_lot_size_less_building_footprint_per_unit = ln(household.income) * ln(((building.disaggregate(parcel.parcel_sqft)) - building.land_area) / building.residential_units)
 I_ln_income_x_ln_lot_size_less_building_footprint_per_unit_x_is_single_family_residential = ln(household.income) * ln(((building.disaggregate(parcel.parcel_sqft)) - building.land_area) / building.residential_units) 
 I_ln_income_x_ln_price_per_sqft = ln(household.income) * ln(urbansim_parcel.building.unit_price)
 I_ln_income_x_ln_sqft_per_unit = ln(household.income) * ln(urbansim_parcel.building.building_sqft_per_unit)
 I_ln_income_x_ln_zonal_pop_den = ln(household.income) * (ln(building.disaggregate(urbansim_parcel.zone.population_per_acre))).astype(float32)
I_ln_income_x_is_multi_family_residential = ln(household.income) * washtenaw.building.is_multi_family_residential
 I_ln_income_x_is_single_family_residential = ln(household.income) * washtenaw.building.is_single_family_residential
I_is_high_income_x_is_single_family_residential=urbansim.household.is_high_income * washtenaw.building.is_single_family_residential
 I_is_mid_income_x_is_single_family_residential=urbansim.household.is_mid_income * washtenaw.building.is_single_family_residential
 I_is_high_income_x_school_quality=urbansim.household.is_high_income * building.disaggregate(school_district.proficient10, intermediates=[parcel])
 I_is_mid_income_x_school_quality=urbansim.household.is_mid_income * building.disaggregate(school_district.proficient10, intermediates=[parcel])
 I_is_low_income_x_school_quality=urbansim.household.is_low_income * building.disaggregate(school_district.proficient10, intermediates=[parcel])
 I_is_high_income_x_crime_rate=urbansim.household.is_high_income * building.disaggregate(city.rate_total, intermediates=[parcel])
 I_is_mid_income_x_crime_rate=urbansim.household.is_mid_income * building.disaggregate(city.rate_total, intermediates=[parcel])
 I_is_low_income_x_crime_rate=urbansim.household.is_low_income * building.disaggregate(city.rate_total, intermediates=[parcel])
 I_is_high_income_x_pptytax=urbansim.household.is_high_income * building.disaggregate(parcel.pptytax)
 I_is_mid_income_x_pptytax=urbansim.household.is_mid_income * building.disaggregate(parcel.pptytax)
 I_is_low_income_x_pptytax=urbansim.household.is_low_income * building.disaggregate(parcel.pptytax)
  I_is_high_income_x_pptytax=urbansim.household.is_high_income * building.disaggregate(parcel.pptytax)
 I_is_mid_income_x_pptytax=urbansim.household.is_mid_income * building.disaggregate(parcel.pptytax)
 I_is_low_income_x_pptytax=urbansim.household.is_low_income * building.disaggregate(parcel.pptytax)
 I_is_high_income_x_percent_high_income=urbansim.household.is_high_income * building.disaggregate(washtenaw.zone.percent_high_income)
 I_is_mid_income_x_percent_mid_income=urbansim.household.is_mid_income * building.disaggregate(washtenaw.zone.percent_mid_income)
 I_is_low_income_x_percent_low_income=urbansim.household.is_low_income * building.disaggregate(washtenaw.zone.percent_low_income)

## size
 I_hh_size_x_ln_zonal_pop_den = household.persons * (ln(building.disaggregate(urbansim_parcel.zone.population_per_acre))).astype(float32)
 I_hh_size_x_ln_sqft_per_unit = household.persons * ln(urbansim_parcel.building.sqft_per_unit)
I_hh_size_3_x_single_family_residential=(household.persons> 2) *washtenaw.building.is_single_family_residential
 I_one_per_x_not_single_family_residential = (household.persons < 2) * numpy.logical_not(washtenaw.building.is_single_family_residential)

## children
I_has_children_x_ln_zonal_pop_den = (household.children > 0) * (ln(building.disaggregate(urbansim_parcel.zone.population_per_acre))).astype(float32)
 I_has_children_x_is_single_family_residential = (household.children > 0) * washtenaw.building.is_single_family_residential
 I_has_children_x_ln_sqft_per_unit = (household.children > 0) * ln(urbansim_parcel.building.sqft_per_unit)
 I_has_children_x_zonal_hh_with_children = (household.children > 0) * building.disaggregate(washtenaw.zone.percent_household_with_children)
 I_has_children_x_n_school_quality=(household.children > 0) * building.disaggregate(school_district.proficient10, intermediates=[parcel])
 I_has_children_x_crime_rate=(household.children > 0) *building.disaggregate(city.rate_total, intermediates=[parcel])

## age
 I_is_young_x_ln_zonal_emp_den = urbansim.household.is_young * (ln(building.disaggregate(urbansim_parcel.zone.number_of_jobs_per_acre))).astype(float32)
 I_is_young_x_ln_zonal_number_of_jobs_of_sector_retail = urbansim.household.is_young * ln_bounded(building.disaggregate(urbansim_parcel.zone.number_of_jobs_of_sector_7))
 I_is_young_x_ln_zonal_number_of_jobs_of_sector_retail_and_food_services = urbansim.household.is_young * ln_bounded(building.disaggregate(urbansim_parcel.zone.number_of_jobs_of_sector_7 + urbansim_parcel.zone.number_of_jobs_of_sector_14))


## race
 I_is_race1_x_zonal_hh_race1=(household.race_id==1) * building.disaggregate(urbansim_parcel.zone.percent_household_race1)
 I_is_race2_x_zonal_hh_race2=(household.race_id==2) * building.disaggregate(urbansim_parcel.zone.percent_household_race2)
 I_is_race3_x_zonal_hh_race3=(household.race_id==3) * building.disaggregate(urbansim_parcel.zone.percent_household_race3)
 I_is_race4_x_zonal_hh_race4=(household.race_id==4) * building.disaggregate(urbansim_parcel.zone.percent_household_race4)
### ACCESSIBILITY
         
A_has_workers_x_ln_emp_45min_hbw_drive_alone = (household.workers > 0) * building.disaggregate(ln_bounded(urbansim_parcel.zone.employment_within_45_minutes_travel_time_hbw_am_drive_alone))
A_has_workers_x_ln_emp_45min_hbw_transit_walk = (household.workers > 0) * building.disaggregate(ln_bounded(urbansim_parcel.zone.employment_within_45_minutes_travel_time_hbw_am_transit_walk))
A_ln_employment_within_45_minutes_travel_time_hbw_am_drive_alone = building.disaggregate(ln_bounded(urbansim_parcel.zone.employment_within_45_minutes_travel_time_hbw_am_drive_alone))
A_ln_employment_within_45_minutes_travel_time_hbw_am_transit_walk = building.disaggregate(ln_bounded(urbansim_parcel.zone.employment_within_45_minutes_travel_time_hbw_am_transit_walk))
A_logsum_accessibility_emp=(household.workers > household.cars)* (washtenaw.building.logsum_work_more_woker_than_car)+(household.workers <= household.cars)* (washtenaw.building.logsum_work_less_woker_than_car)
A_logsum_accessibility_pop=washtenaw.building.logsum_pop_less_woker_than_car


# # variables for future test
