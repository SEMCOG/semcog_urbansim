#!/usr/bin/env python
# coding: utf-8

# import numpy as np, pandas as pd
# from urbansim.models import MNLLocationChoiceModel, RegressionModelGroup, transition
# from developer import developer, feasibility
# from urbansim.urbansim import dataset
# from variables import var_calc
# 
# dset = dataset.Dataset('data//semcog_data.h5')
# vacant_residential_units, vacant_job_spaces = var_calc.calculate(dset)

# ## Base year estimation 

# In[2]:


###HLCM
#Agents for estimation
households_for_estimation = dset.households.loc[np.random.choice(dset.households.index, size=15000, replace=False)]
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
hlcm.fit(households_for_estimation, dset.buildings, households_for_estimation.building_id)
hlcm.report_fit()


###ELCM
#Agents for estimation
jobs_for_estimation = dset.jobs_for_estimation
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
elcm.fit(jobs_for_estimation, dset.buildings, jobs_for_estimation.building_id)
elcm.report_fit()

# elcms = {}
# for sector_id in dset.jobs_for_estimation[dset.jobs_for_estimation.home_based_status==0].groupby('sector_id').size().index.values:
#     print 'SECTOR %s' % sector_id
#     #Agents for estimation
#     jobs_for_estimation = dset.jobs_for_estimation[dset.jobs_for_estimation.sector_id==sector_id]
#     #Filters on agents
#     agent_estimation_filters = ['home_based_status == 0']
#     agents_simulation_filters = None
#     ##Specification
#     patsy_expression = ['np.log1p(non_residential_sqft)',
#                         'np.log1p(improvement_value)',
#                         'jobs_within_30_min',
#                         'popden',
#                         'crime08']
#     patsy_expression = ' + '.join(patsy_expression)
#     #Filters on alternatives
#     estimation_filters = ['building_id > 0']
#     simulation_filters = ['non_residential_sqft>0',
#                           'building_type_id not in [16,17,18,19,20]']
#     ##Instantiate ELCM
#     elcm = MNLLocationChoiceModel(
#         patsy_expression, 10,
#         choosers_fit_filters=agent_estimation_filters, choosers_predict_filters=agents_simulation_filters,
#         alts_fit_filters=estimation_filters, alts_predict_filters=simulation_filters,
#         choice_column='building_id', name='ELCM')
#     ##Estimate
#     elcm.fit(jobs_for_estimation, dset.buildings, jobs_for_estimation.building_id)
#     elcm.report_fit()
#     ##Store sector-specific ELCMs
#     elcms[sector_id] = elcm

###Residential REPM
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
res_price_model = RegressionModelGroup('building_type_id')
for key in group_keys:
    res_price_model.add_model_from_params(key, estimate_filters, simulate_filters,
                              model_expression, ytransform=np.exp)
##Estimate
fits_res = res_price_model.fit(dset.buildings)


###Nonresidential REPM
##Model specification
def patsy_expression():
    patsy_exp = ['I(year_built < 1940)',
                 'year_built',
                 'stories',
                 'np.log1p(non_residential_sqft)',
                 'np.log1p(popden)',
                 'dist_hwy',
                 'dist_road',
                 'crime08',
                 'np.log1p(jobs_within_30_min)']
    patsy_exp = ' + '.join(patsy_exp)
    return 'np.log(unit_price_nonres) ~ ' + patsy_exp
model_expression = patsy_expression()
##Estimation filters
estimate_filters = ['non_residential_sqft > 0',
                    'year_built > 1700',
                    'stories > 0',
                    'tax_exempt == 0',
h                    'unit_price_nonres > 0',
                    'building_type_id > 20']
##Simulation filters
simulate_filters = ['non_residential_sqft > 0',
                    '16 <= building_type_id > 20']
##Segmentation
group_keys = [21,22,23,24,27,28,33,39] ##btype 32 had only 1 observation
nonres_price_model = RegressionModelGroup('building_type_id')
for key in group_keys:
    nonres_price_model.add_model_from_params(key, estimate_filters, simulate_filters,
                              model_expression, ytransform=np.exp)
##Estimate
fits_nonres = nonres_price_model.fit(dset.buildings)


# ## Model Simulation

# In[3]:


for year in range (2011,2020):
    
    vacant_residential_units, vacant_job_spaces = var_calc.calculate(dset)
    
    ##Record county-level hh/job counts
    dset.households['county_id'] = dset.buildings.county_id[dset.households.building_id].values
    dset.jobs['county_id'] = dset.buildings.county_id[dset.jobs.building_id].values
    starting_hh = dset.households.groupby('county_id').size()
    starting_emp = dset.jobs.groupby('county_id').size()
    
    ##Household transition
    ct = dset.fetch('annual_household_control_totals')
    totals_field = ct.reset_index().groupby('year').total_number_of_households.sum()
    ct = pd.DataFrame({'total_number_of_households':totals_field})
    tran = transition.TabularTotalsTransition(ct, 'total_number_of_households')
    model = transition.TransitionModel(tran)
    new, added_hh_idx, new_linked = model.transition(
            dset.households, year, linked_tables={'linked': (dset.persons, 'household_id')})
    dset.households = new
    dset.persons = new_linked['linked']
    
    ##Employment transition
    ct_emp = dset.fetch('annual_employment_control_totals')
    totals_field = ct_emp.reset_index().groupby('year').total_number_of_jobs.sum()
    ct_emp = pd.DataFrame({'total_number_of_jobs':totals_field})
    tran = transition.TabularTotalsTransition(ct_emp, 'total_number_of_jobs')
    model = transition.TransitionModel(tran)
    new, added_jobs_idx, new_linked = model.transition(dset.jobs, year)
    dset.jobs = new
    
    ###HLCM
    new_assignments = hlcm.predict(dset.households.loc[added_hh_idx], vacant_residential_units)
    dset.households.loc[added_hh_idx,'building_id'] = new_assignments
    
    ###ELCM
    new_assignments = elcm.predict(dset.jobs.loc[added_jobs_idx], vacant_job_spaces)
    dset.jobs.loc[added_jobs_idx,'building_id'] = new_assignments
#     job_sectors = dset.jobs.loc[added_jobs_idx, 'sector_id']
#     for sector in np.unique(job_sectors.values):
#         if sector in elcms.keys():
#             print sector
#             idx_by_sector = job_sectors[job_sectors.values==sector]
#             added_jobs_idx_by_sector = idx_by_sector.index
#             elcm = elcms[sector]
#             print elcm
#             new_assignments = elcm.predict(dset.jobs.loc[added_jobs_idx_by_sector], vacant_job_spaces)  ##vacant job spaces needs to be updated
#             dset.jobs.loc[added_jobs_idx_by_sector,'building_id'] = new_assignments
    
    ####REPM
    new_unit_price_res = res_price_model.predict(dset.buildings)
    dset.buildings.loc[new_unit_price_res.index.values, 'unit_price_res'] = new_unit_price_res
    
    new_unit_price_nonres = nonres_price_model.predict(dset.buildings)
    dset.buildings.loc[new_unit_price_nonres.index.values, 'unit_price_nonres'] = new_unit_price_nonres
    
    ##County-level hh/job growth indicator
    dset.households['county_id'] = dset.buildings.county_id[dset.households.building_id].values
    dset.jobs['county_id'] = dset.buildings.county_id[dset.jobs.building_id].values
    ending_hh = dset.households.groupby('county_id').size()
    ending_emp = dset.jobs.groupby('county_id').size()
    print 'HH diffs'
    print ending_hh - starting_hh
    print 'EMP diffs'
    print ending_emp - starting_emp
    
    
    ####PROFORMA
    feasibility.feasibility_run(dset,year)
    
    ##Record starting non-residential sqft by county
    dset.buildings['county_id'] = dset.parcels.county_id[dset.buildings.parcel_id].values
    starting_nonres = dset.buildings.groupby('county_id').non_residential_sqft.sum()
    
    ##Non-residential proformas
    nr_buildings = developer.exec_developer(dset, year, "jobs", "job_spaces",
                   [21,22,23,24,27,28,32,33,39], nonres=True)
    dset.buildings = dset.buildings[['building_id_old', 'building_type_id', 'improvement_value', 'land_area',
         'non_residential_sqft', 'parcel_id', 'residential_units', 'sqft_per_unit',
         'stories', 'tax_exempt', 'year_built']]
    dset.buildings = pd.concat([dset.buildings,nr_buildings])
    
    ##County-level non-residential-sqft growth indicator
    dset.buildings['county_id'] = dset.parcels.county_id[dset.buildings.parcel_id].values
    ending_nonres = dset.buildings.groupby('county_id').non_residential_sqft.sum()
    print 'Nonres sqft diffs'
    print ending_nonres - starting_nonres
    
    ##Record starting residential units by county
    dset.buildings['county_id'] = dset.parcels.county_id[dset.buildings.parcel_id].values
    starting_resunits = dset.buildings.groupby('county_id').residential_units.sum()
    
    ##Residential proformas
    res_buildings = developer.exec_developer(dset, year, "households", 
                                             "residential_units", [16, 17, 18, 19])
    dset.buildings = pd.concat([dset.buildings,res_buildings])
    
    ##County-level residential unit growth indicator
    dset.buildings['county_id'] = dset.parcels.county_id[dset.buildings.parcel_id].values
    ending_resunits = dset.buildings.groupby('county_id').residential_units.sum()
    print 'Resunit diffs'
    print ending_resunits - starting_resunits

