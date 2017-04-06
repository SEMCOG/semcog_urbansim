import numpy as np
import pandas as pd

import orca
from urbansim.utils import misc

import dataset, variables, utils, models
import models


#orca.run(["build_networks"])
#orca.run(["neighborhood_vars"])

vars_to_dummify = ['city_id', 'building_type_id']
vars_to_log = ['non_residential_sqft', 'building_sqft', 'land_area', 'parcel_sqft', 'sqft_per_unit', 
               'parcels_parcel_far', 'sqft_price_nonres']

emp_sectors = np.arange(18) + 1

geographic_levels = [('parcels', 'parcel_id'),
                     ('zones', 'zone_id'),
                     ('nodes_walk', 'nodeid_walk'),
                     ('nodes_drv', 'nodeid_drv')]

@orca.injectable('year')
def year():
    default_year = 2015
    try:
        iter_var = orca.get_injectable('iter_var')
        if iter_var is not None:
            return iter_var
        else:
            return default_year
    except:
        return default_year
    

@orca.column('zones', 'z_total_jobs', cache=True, cache_scope='iteration')
def z_total_jobs(jobs):
    return jobs.zone_id.value_counts()

@orca.column('zones', 'transit_jobs_50min', cache=True, cache_scope='iteration')
def transit_jobs_45min(zones, travel_data):
    td = travel_data.to_frame('am_transit_total_time').reset_index()
    zemp = zones.to_frame('employment')
    temp = pd.merge(td,zemp, left_on = 'to_zone_id', right_index = True, how='left' )
    transit_jobs_45min = temp[temp.am_transit_total_time <=50].groupby('from_zone_id').employment.sum()
    return transit_jobs_45min

@orca.column('zones', 'a_ln_emp_26min_drive_alone', cache=True, cache_scope='iteration')
def a_ln_emp_26min_drive_alone(zones, travel_data):
    drvtime = travel_data.to_frame("am_auto_total_time").reset_index()
    zemp = zones.to_frame('employment')
    temp = pd.merge(drvtime, zemp, left_on ='to_zone_id', right_index = True, how='left' )
    return np.log1p(temp[temp.am_auto_total_time <=26].groupby('from_zone_id').employment.sum().fillna(0))

@orca.column('zones', 'a_ln_emp_50min_transit', cache=True, cache_scope='iteration')
def a_ln_emp_50min_transit(zones, travel_data):
    transittime = travel_data.to_frame("am_transit_total_time").reset_index()
    zemp = zones.to_frame('employment')
    temp = pd.merge(transittime,zemp, left_on = 'to_zone_id', right_index = True, how='left' )
    return np.log1p(temp[temp.am_transit_total_time <=50].groupby('from_zone_id').employment.sum().fillna(0))

@orca.column('zones', 'a_ln_retail_emp_15min_drive_alone', cache=True, cache_scope='iteration')
def a_ln_retail_emp_15min_drive_alone(zones, travel_data):
    drvtime = travel_data.to_frame("midday_auto_total_time").reset_index()
    zemp = zones.to_frame('employment')
    temp = pd.merge(drvtime,zemp, left_on = 'to_zone_id', right_index = True, how='left' )
    return np.log1p(temp[temp.midday_auto_total_time <=15].groupby('from_zone_id').employment.sum().fillna(0))

@orca.column('nodes_walk', 'node_r1500_acre', cache=True, cache_scope='iteration')
def node_r1500_acre(nodes_walk):
    return nodes_walk['node_r1500_sqft'] / 43560

@orca.column('nodes_walk', 'ln_empden', cache=True, cache_scope='iteration')
def ln_empden(nodes_walk):
    return np.log1p(nodes_walk.jobs / nodes_walk.node_r1500_acre).fillna(0)

@orca.column('nodes_walk', 'ln_popden', cache=True, cache_scope='iteration')
def ln_popden(nodes_walk):
    return np.log1p(nodes_walk.population / nodes_walk.node_r1500_acre).fillna(0)

@orca.column('nodes_walk', 'percent_high_income', cache=True, cache_scope='iteration')
def percent_high_income(nodes_walk):
    return np.log1p(nodes_walk.highinc_hhs / nodes_walk.households).fillna(0)

@orca.column('nodes_walk', 'percent_mid_income', cache=True, cache_scope='iteration')
def percent_mid_income(nodes_walk):
    return np.log1p(nodes_walk.midinc_hhs / nodes_walk.households).fillna(0)

@orca.column('nodes_walk', 'percent_low_income', cache=True, cache_scope='iteration')
def percent_low_income(nodes_walk):
    return np.log1p(nodes_walk.lowinc_hhs / nodes_walk.households).fillna(0)

@orca.column('nodes_walk', 'percent_race1', cache=True, cache_scope='iteration')
def percent_race1(nodes_walk):
    return np.log1p(nodes_walk.race_1_hhs / nodes_walk.households).fillna(0)

@orca.column('nodes_walk', 'percent_race2', cache=True, cache_scope='iteration')
def percent_race2(nodes_walk):
    return np.log1p(nodes_walk.race_2_hhs / nodes_walk.households).fillna(0)

@orca.column('nodes_walk', 'percent_race3', cache=True, cache_scope='iteration')
def percent_race3(nodes_walk):
    return np.log1p(nodes_walk.race_3_hhs / nodes_walk.households).fillna(0)

@orca.column('nodes_walk', 'percent_race4', cache=True, cache_scope='iteration')
def percent_race4(nodes_walk):
    return np.log1p(nodes_walk.race_4_hhs / nodes_walk.households).fillna(0)

@orca.column('nodes_walk', 'percent_hh_with_children', cache=True, cache_scope='iteration')
def percent_hh_with_children(nodes_walk):
    return np.log1p(nodes_walk.hhs_with_children / nodes_walk.households).fillna(0)
    
@orca.column('buildings', 'building_age', cache=True, cache_scope='iteration')
def building_age(buildings, year):
    year_built = buildings.year_built
    year_built[year_built < 1600] = year_built[year_built > 1600].mean()
    age = year - year_built
    return age

@orca.column('buildings', 'b_is_pre_1945', cache=True, cache_scope='iteration')
def b_is_pre_1945(buildings):
    return (buildings.year_built < 1945).astype('int32')

@orca.column('buildings', 'b_is_newerthan2010', cache=True, cache_scope='iteration')
def b_is_newerthan2010(buildings):
    return (buildings.year_built > 2010).astype('int32')

@orca.column('buildings', 'b_total_jobs', cache=True, cache_scope='iteration')
def b_total_jobs(jobs, buildings):
    jobs_by_b = jobs.building_id.groupby(jobs.building_id).size()
    return pd.Series(index=buildings.index, data=jobs_by_b).fillna(0)

def make_employment_proportion_variable(sector_id):
    """
    Generate employment proportion of total jobs in building variable. Registers with orca.
    """
    var_name = 'bldg_empratio_%s' % sector_id
    
    @orca.column('buildings', var_name, cache=True, cache_scope='iteration')
    def func():
        buildings = orca.get_table('buildings')
        jobs = orca.get_table('jobs')
        total_jobs = buildings.b_total_jobs
        jobs = jobs.to_frame(jobs.local_columns)
        jobs_sector = jobs[jobs.sector_id == sector_id].building_id.value_counts()
        return (jobs_sector / total_jobs).fillna(0)

    return func

def make_employment_density_variable(sector_id):
    """
    Generate zonal employment density variable. Registers with orca.
    """
    var_name = 'ln_empden_%s' % sector_id
    
    @orca.column('zones', var_name, cache=True, cache_scope='iteration')
    def func():
        zones = orca.get_table('zones')
        jobs = orca.get_table('jobs')
        total_acres = zones.acres
        jobs = jobs.to_frame(jobs.local_columns + ['zone_id'])
        jobs_sector = jobs[jobs.sector_id == sector_id].zone_id.value_counts()
        return np.log1p(jobs_sector / total_acres).fillna(0)

    return func

def make_dummy_variable(geog_var, geog_id):
    """
    Generate dummy variable. Registers with orca.
    """
    var_name = '%s_is_%s' % (geog_var, geog_id)
    
    @orca.column('buildings', var_name, cache=True, cache_scope='iteration')
    def func():
        buildings = orca.get_table('buildings')
        return (buildings[geog_var] == geog_id).astype('int32')

    return func

def make_logged_variable(var_to_log):
    """
    Generate logged variable. Registers with orca.
    """
    var_name = 'b_ln_%s' % var_to_log
    
    @orca.column('buildings', var_name, cache=True, cache_scope='iteration')
    def func():
        buildings = orca.get_table('buildings')
        return np.log1p(buildings[var_to_log]).fillna(0)

    return func

def make_disagg_var(from_geog_name, to_geog_name, var_to_disaggregate, from_geog_id_name, name_based_on_geography=True):
    """
    Generator function for disaggregating variables. Registers with orca.
    """
    if name_based_on_geography:
        var_name = from_geog_name + '_' + var_to_disaggregate
    else:
        var_name = var_to_disaggregate
    @orca.column(to_geog_name, var_name, cache=True, cache_scope='iteration')
    def func():
        print 'Disaggregating %s to %s from %s' % (var_to_disaggregate, to_geog_name, from_geog_name)

        from_geog = orca.get_table(from_geog_name)
        to_geog = orca.get_table(to_geog_name)
        return misc.reindex(from_geog[var_to_disaggregate], to_geog[from_geog_id_name]).fillna(0)

    return func

for dummifiable_var in vars_to_dummify:
#     print dummifiable_var
    var_cat_ids = np.unique(orca.get_table('buildings')[dummifiable_var]).astype('int')
    for var_cat_id in var_cat_ids:
        if var_cat_id > 0:
#             print var_cat_id
            make_dummy_variable(dummifiable_var, var_cat_id)
    
for sector in emp_sectors:
    make_employment_proportion_variable(sector)
    make_employment_density_variable(sector)
            
for geography in geographic_levels:
    geography_name = geography[0]
    geography_id = geography[1]
    if geography_name != 'buildings':
        building_vars = orca.get_table('buildings').columns
        for var in orca.get_table(geography_name).columns:
            if var not in building_vars:
                make_disagg_var(geography_name, 'buildings', var, geography_id)
                
for var_to_log in vars_to_log:
    make_logged_variable(var_to_log)
    
