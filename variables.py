import pandas as pd, numpy as np
from urbansim.utils import misc
import urbansim.sim.simulation as sim
import dataset

#####################
# BUILDINGS VARIABLES
#####################

@sim.column('buildings', 'school_district_id', cache=True)
def school_district_id(buildings, parcels):
    return misc.reindex(parcels.school_district_id, buildings.parcel_id)

@sim.column('buildings', 'general_type', cache=True)
def general_type(buildings, building_type_map):
    return buildings.building_type_id.map(building_type_map).fillna(0)

@sim.column('buildings', '_node_id', cache=True)
def _node_id(buildings, parcels):
    return misc.reindex(parcels._node_id, buildings.parcel_id)

@sim.column('buildings', '_node_id0', cache=True)
def _node_id0(buildings, parcels):
    return misc.reindex(parcels._node_id0, buildings.parcel_id)
        
@sim.column('buildings', 'x', cache=True)
def x(buildings, parcels):
    return misc.reindex(parcels.x, buildings.parcel_id)

@sim.column('buildings', 'y', cache=True)
def y(buildings, parcels):
    return misc.reindex(parcels.y, buildings.parcel_id)
        
@sim.column('buildings', 'dist_hwy', cache=True)
def dist_hwy(buildings, parcels):
    return misc.reindex(parcels.dist_hwy, buildings.parcel_id)

@sim.column('buildings', 'dist_road', cache=True)
def dist_road(buildings, parcels):
    return misc.reindex(parcels.dist_road, buildings.parcel_id)
        
@sim.column('buildings', 'zone_id', cache=True)
def zone_id(buildings, parcels):
    return misc.reindex(parcels.zone_id, buildings.parcel_id)
        
@sim.column('buildings', 'city_id', cache=True)
def city_id(buildings, parcels):
    return misc.reindex(parcels.city_id, buildings.parcel_id)
        
@sim.column('buildings', 'large_area_id', cache=True)
def large_area_id(buildings, parcels):
    return misc.reindex(parcels.large_area_id, buildings.parcel_id)
        
@sim.column('buildings', 'crime08', cache=True)
def crime08(buildings, cities):
    return misc.reindex(cities.crime08, buildings.city_id)
        
@sim.column('buildings', 'popden', cache=True)
def popden(buildings, zones):
    return misc.reindex(zones.popden, buildings.zone_id).fillna(0)
        
@sim.column('buildings', 'building_sqft', cache=True)
def building_sqft(buildings):
    return buildings.non_residential_sqft + buildings.sqft_per_unit*buildings.residential_units
              
@sim.column('buildings', 'building_sqft_per_job', cache=True)
def building_sqft_per_job(buildings, building_sqft_per_job):
    b = pd.DataFrame({'zone_id':buildings.zone_id, 'building_type_id':buildings.building_type_id})
    bsqft_job = building_sqft_per_job.to_frame()
    
    return pd.merge(b,
                    bsqft_job,
                    left_on=['zone_id', 'building_type_id'],
                    right_index=True, how='left').building_sqft_per_job.fillna(0)
        
@sim.column('buildings', 'job_spaces', cache=True)
def job_spaces(buildings):
    job_spaces = buildings.non_residential_sqft / buildings.building_sqft_per_job
    job_spaces[np.isinf(job_spaces)] = np.nan
    job_spaces[job_spaces < 0] = 0
    job_spaces = job_spaces.fillna(0).round().astype('int')
    return job_spaces
        
@sim.column('buildings', 'non_residential_units', cache=True)
def non_residential_units(buildings):
    job_spaces = buildings.non_residential_sqft / buildings.building_sqft_per_job
    job_spaces[np.isinf(job_spaces)] = np.nan
    job_spaces[job_spaces < 0] = 0
    job_spaces = job_spaces.fillna(0).round().astype('int')
    return job_spaces
        
@sim.column('buildings', 'jobs_within_30_min', cache=True)
def jobs_within_30_min(buildings, zones):
    return misc.reindex(zones.jobs_within_30_min, buildings.zone_id).fillna(0)
    
@sim.column('buildings', 'vacant_residential_units')
def vacant_residential_units(buildings, households):
    return buildings.residential_units.sub(
        households.building_id.value_counts(), fill_value=0)

@sim.column('buildings', 'vacant_job_spaces')
def vacant_job_spaces(buildings, jobs):
    return buildings.job_spaces.sub(
        jobs.building_id.value_counts(), fill_value=0)


#####################
# HOUSEHOLDS VARIABLES
#####################

@sim.column('households', 'school_district_id', cache=True)
def school_district_id(households, buildings):
    return misc.reindex(buildings.school_district_id, households.building_id)
        
@sim.column('households', 'income_quartile', cache=True)
def income_quartile(households):
    return pd.Series(pd.qcut(households.income, 4).labels,
                     index=households.index)
        
@sim.column('households', 'zone_id', cache=True)
def zone_id(households, buildings):
    return misc.reindex(buildings.zone_id, households.building_id)
        
@sim.column('households', 'x', cache=True)
def x(households, buildings):
    return misc.reindex(buildings.x, households.building_id)
    
@sim.column('households', 'y', cache=True)
def y(households, buildings):
    return misc.reindex(buildings.y, households.building_id)

@sim.column('households', 'large_area', cache=True)
def large_area(households, buildings):
    return misc.reindex(buildings.large_area_id, households.building_id)
        
@sim.column('households', 'lid', cache=True)
def lid(households):
    return households.large_area_id
       
@sim.column('households', '_node_id', cache=True)
def _node_id(households, buildings):
    return misc.reindex(buildings._node_id, households.building_id)

@sim.column('households', '_node_id0', cache=True)
def _node_id0(households, buildings):
    return misc.reindex(buildings._node_id0, households.building_id)
    
#####################
# PERSONS VARIABLES
#####################
        
@sim.column('persons', 'zone_id', cache=True)
def zone_id(persons, households):
    return misc.reindex(households.zone_id, persons.household_id)
    
@sim.column('persons', 'school_district_id', cache=True)
def school_district_id(persons, households):
    return misc.reindex(households.school_district_id, persons.household_id)


#####################
# JOBS VARIABLES
#####################

@sim.column('jobs', 'zone_id', cache=True)
def zone_id(jobs, buildings):
    return misc.reindex(buildings.zone_id, jobs.building_id)
        
@sim.column('jobs', 'parcel_id', cache=True)
def parcel_id(jobs, buildings):
    return misc.reindex(buildings.parcel_id, jobs.building_id)
        
@sim.column('jobs', 'x', cache=True)
def x(jobs, buildings):
    return misc.reindex(buildings.x, jobs.building_id)
    
@sim.column('jobs', 'y', cache=True)
def y(jobs, buildings):
    return misc.reindex(buildings.y, jobs.building_id)

@sim.column('jobs', 'large_area', cache=True)
def large_area(jobs, buildings):
    return misc.reindex(buildings.large_area_id, jobs.building_id)
    
@sim.column('jobs', 'lid', cache=True)
def lid(jobs):
    return jobs.large_area_id

@sim.column('jobs', '_node_id', cache=True)
def _node_id(jobs, buildings):
    return misc.reindex(buildings._node_id, jobs.building_id)

@sim.column('jobs', '_node_id0', cache=True)
def _node_id0(jobs, buildings):
    return misc.reindex(buildings._node_id0, jobs.building_id)


#####################
# PARCELS VARIABLES
#####################
        
@sim.column('parcels', 'acres', cache=True)
def acres(parcels):
    return parcels.parcel_sqft / 43560
        
@sim.column('parcels', 'x', cache=True)
def x(parcels):
    return parcels.centroid_x
    
@sim.column('parcels', 'y', cache=True)
def y(parcels):
    return parcels.centroid_y
        
@sim.column('parcels', 'allowed', cache=True)
def allowed(parcels):
    df = pd.DataFrame(index=parcels.index)
    df['allowed'] = True
    return df.allowed

def parcel_average_price(use):
    return misc.reindex(sim.get_table('nodes_prices')[use],
                        sim.get_table('parcels')._node_id)
                        
def parcel_is_allowed(form):
    #form_to_btype = sim.get_injectable("form_to_btype")
    parcels = sim.get_table('parcels')
    df = pd.DataFrame(index=parcels.index)
    df['allowed'] = True
    return df.allowed

@sim.column('parcels', 'max_far', cache=True)
def max_far(parcels):
    df = pd.DataFrame(index=parcels.index)
    df['max_far'] = 2.0
    return df.max_far
            
@sim.column('parcels', 'max_height', cache=True)
def max_height(parcels):
    df = pd.DataFrame(index=parcels.index)
    df['max_height'] = 100
    return df.max_height
        
@sim.column('parcels', 'parcel_size', cache=True)
def parcel_size(parcels):
    return parcels.parcel_sqft
        
@sim.column('parcels', 'ave_unit_size')
def ave_unit_size(parcels, nodes):
    if len(nodes) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes.ave_unit_sqft, parcels._node_id)
        
@sim.column('parcels', 'total_units', cache=True)
def total_units(parcels, buildings):
    return buildings.residential_units.groupby(buildings.parcel_id).sum().\
        reindex(parcels.index).fillna(0)

@sim.column('parcels', 'total_job_spaces', cache=True)
def total_job_spaces(parcels, buildings):
    return buildings.job_spaces.groupby(buildings.parcel_id).sum().\
        reindex(parcels.index).fillna(0)

@sim.column('parcels', 'total_sqft', cache=True)
def total_sqft(parcels, buildings):
    return buildings.building_sqft.groupby(buildings.parcel_id).sum().\
        reindex(parcels.index).fillna(0)
            
@sim.column('parcels', 'land_cost')
def land_cost(parcels, nodes_prices):
    if len(nodes_prices) == 0:
        return pd.Series(index=parcels.index)
    return (parcels.total_sqft * parcel_average_price("residential")).\
        reindex(parcels.index).fillna(0)
        
#####################
# ZONES VARIABLES
#####################
                                  
@sim.column('zones', 'popden', cache=True)
def popden(zones, parcels, households):
    return households.persons.groupby(households.zone_id).sum() / parcels.acres.groupby(parcels.zone_id).sum()
        
@sim.column('zones', 'jobs_within_30_min', cache=True)
def jobs_within_30_min(jobs, travel_data):
    j = pd.DataFrame({'zone_id':jobs.zone_id})
    td = travel_data.to_frame()
    return misc.compute_range(td,
                                  j.groupby('zone_id').size(),
                                  "am_single_vehicle_to_work_travel_time",
                                  30, agg=np.sum)