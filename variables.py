import pandas as pd, numpy as np
from urbansim.utils import misc
import dataset

###
import orca
import utils
import pandana as pdna


#####################
# BUILDINGS VARIABLES
#####################

##@orca.column('buildings', 'school_district_id', cache=True)
##def school_district_id(buildings, parcels):
##    return misc.reindex(parcels.school_district_id, buildings.parcel_id)

@orca.column('buildings', cache=True, cache_scope='iteration')
def general_type(buildings, building_type_map):
    return buildings.building_type_id.map(building_type_map).fillna(0)

##@orca.column('buildings', cache=True, cache_scope='iteration')
##def _node_id(buildings, parcels):
##    return misc.reindex(parcels._node_id, buildings.parcel_id)

@orca.column('buildings', cache=True, cache_scope='iteration')
def nodeid_walk(buildings, parcels):
    return misc.reindex(parcels.nodeid_walk, buildings.parcel_id)

@orca.column('buildings', cache=True, cache_scope='iteration')
def nodeid_drv(buildings, parcels):
    return misc.reindex(parcels.nodeid_drv, buildings.parcel_id)


@orca.column('buildings', cache=True, cache_scope='iteration')
def x(buildings, parcels):
    return misc.reindex(parcels.x, buildings.parcel_id)

@orca.column('buildings', cache=True, cache_scope='iteration')
def y(buildings, parcels):
    return misc.reindex(parcels.y, buildings.parcel_id)

##@orca.column('buildings', 'dist_hwy', cache=True)
##def dist_hwy(buildings, parcels):
##    return misc.reindex(parcels.dist_hwy, buildings.parcel_id)
##
##@orca.column('buildings', 'dist_road', cache=True)
##def dist_road(buildings, parcels):
##    return misc.reindex(parcels.dist_road, buildings.parcel_id)

@orca.column('buildings', cache=True, cache_scope='iteration')
def zone_id(buildings, parcels):
    return misc.reindex(parcels.zone_id, buildings.parcel_id)

@orca.column('buildings', cache=True, cache_scope='iteration')
def city_id(buildings, parcels):
    return misc.reindex(parcels.city_id, buildings.parcel_id)

@orca.column('buildings', cache=True, cache_scope='iteration')
def large_area_id(buildings, parcels):
    return misc.reindex(parcels.large_area_id, buildings.parcel_id)

@orca.column('buildings', cache=True, cache_scope='iteration')
def crime08(buildings, cities):
    return misc.reindex(cities.crime08, buildings.city_id).fillna(0)

@orca.column('buildings', cache=True, cache_scope='iteration')
def popden(buildings, zones):
    return misc.reindex(zones.popden, buildings.zone_id).fillna(0)

@orca.column('buildings', cache=True, cache_scope='iteration')
def building_sqft(buildings):
    return buildings.non_residential_sqft + buildings.sqft_per_unit*buildings.residential_units

@orca.column('buildings', cache=True, cache_scope='iteration')
def building_sqft_per_job(buildings, building_sqft_per_job):
    b = pd.DataFrame({'zone_id':buildings.zone_id, 'building_type_id':buildings.building_type_id})
    bsqft_job = building_sqft_per_job.to_frame()

    return pd.merge(b,bsqft_job,
                    left_on=['zone_id', 'building_type_id'],
                    right_index=True, how='left').building_sqft_per_job.fillna(0)

@orca.column('buildings', cache=True, cache_scope='iteration')
def job_spaces(buildings):
    job_spaces = buildings.non_residential_sqft / buildings.building_sqft_per_job
    job_spaces[np.isinf(job_spaces)] = np.nan
    job_spaces[job_spaces < 0] = 0
    job_spaces = job_spaces.fillna(0).round().astype('int')
    return job_spaces

@orca.column('buildings', cache=True, cache_scope='iteration')
def non_residential_units(buildings):
    job_spaces = buildings.non_residential_sqft / buildings.building_sqft_per_job
    job_spaces[np.isinf(job_spaces)] = np.nan
    job_spaces[job_spaces < 0] = 0
    job_spaces = job_spaces.fillna(0).round().astype('int')
    return job_spaces

##@orca.column('buildings', 'jobs_within_30_min', cache=True)
##def jobs_within_30_min(buildings, zones):
##    return misc.reindex(zones.jobs_within_30_min, buildings.zone_id).fillna(0)

@orca.column('buildings')
def vacant_residential_units(buildings, households):
    return buildings.residential_units.sub(
        households.building_id.value_counts(), fill_value=0)

@orca.column('buildings')
def vacant_job_spaces(buildings, jobs):
    return buildings.job_spaces.sub(
        jobs.building_id.value_counts(), fill_value=0)

@orca.column('buildings', cache=True, cache_scope='iteration')
def parcel_sqft(buildings, parcels):
    return misc.reindex(parcels.parcel_sqft, buildings.parcel_id)

@orca.column('buildings', cache=True, cache_scope='iteration')
def school_district_achievement(buildings, parcels):
    return misc.reindex(parcels.school_district_achievement, buildings.parcel_id)

@orca.column('buildings', cache=True, cache_scope='iteration')
def school_district_achievement(buildings, parcels):
    return misc.reindex(parcels.school_district_achievement, buildings.parcel_id)






#####################
# HOUSEHOLDS VARIABLES
#####################
##
##@orca.column('households', 'school_district_id', cache=True)
##def school_district_id(households, buildings):
##    return misc.reindex(buildings.school_district_id, households.building_id)

@orca.column('households', cache=True, cache_scope='iteration')
def income_quartile(households):
    return pd.Series(pd.qcut(households.income, 4,labels=False),
                     index=households.index) + 1

@orca.column('households', cache=True, cache_scope='iteration')
def zone_id(households, buildings):
    return misc.reindex(buildings.zone_id, households.building_id)

@orca.column('households', cache=True, cache_scope='iteration')
def x(households, buildings):
    return misc.reindex(buildings.x, households.building_id)

@orca.column('households', cache=True, cache_scope='iteration')
def y(households, buildings):
    return misc.reindex(buildings.y, households.building_id)

@orca.column('households', cache=True, cache_scope='iteration')
def large_area(households, buildings):
    return misc.reindex(buildings.large_area_id, households.building_id)

@orca.column('households', cache=True, cache_scope='iteration')
def large_area_id(households, buildings):
    return misc.reindex(buildings.large_area_id, households.building_id)

@orca.column('households', cache=True, cache_scope='iteration')
def lid(households):
    return households.large_area_id

@orca.column('households', cache=True, cache_scope='iteration')
def nodeid_walk(households, buildings):
    return misc.reindex(buildings.nodeid_walk, households.building_id)

@orca.column('households', cache=True, cache_scope='iteration')
def nodeid_drv(households, buildings):
    return misc.reindex(buildings.nodeid_drv, households.building_id)


#####################
# PERSONS VARIABLES
#####################

@orca.column('persons', cache=True, cache_scope='iteration')
def zone_id(persons, households):
    return misc.reindex(households.zone_id, persons.household_id)

##@orca.column('persons', 'school_district_id', cache=True)
##def school_district_id(persons, households):
##    return misc.reindex(households.school_district_id, persons.household_id)


#####################
# JOBS VARIABLES
#####################

@orca.column('jobs', cache=True, cache_scope='iteration')
def zone_id(jobs, buildings):
    return misc.reindex(buildings.zone_id, jobs.building_id)

@orca.column('jobs', cache=True, cache_scope='iteration')
def parcel_id(jobs, buildings):
    return misc.reindex(buildings.parcel_id, jobs.building_id)

@orca.column('jobs', cache=True, cache_scope='iteration')
def x(jobs, buildings):
    return misc.reindex(buildings.x, jobs.building_id)

@orca.column('jobs', cache=True, cache_scope='iteration')
def y(jobs, buildings):
    return misc.reindex(buildings.y, jobs.building_id)

@orca.column('jobs', cache=True, cache_scope='iteration')
def large_area_id(jobs, buildings):
    return misc.reindex(buildings.large_area_id, jobs.building_id)

@orca.column('jobs', cache=True, cache_scope='iteration')
def lid(jobs):
    return jobs.large_area_id

@orca.column('jobs', cache=True, cache_scope='iteration')
def nodeid_walk(jobs, buildings):
    return misc.reindex(buildings.nodeid_walk, jobs.building_id)

@orca.column('jobs', cache=True, cache_scope='iteration')
def nodeid_drv(jobs, buildings):
    return misc.reindex(buildings.nodeid_drv, jobs.building_id)


#####################
# PARCELS VARIABLES
#####################

@orca.column('parcels', cache=True, cache_scope='iteration')
def acres(parcels):
    return parcels.parcel_sqft / 43560

@orca.column('parcels', cache=True, cache_scope='iteration')
def x(parcels):
    return parcels.centroid_x

@orca.column('parcels', cache=True, cache_scope='iteration')
def y(parcels):
    return parcels.centroid_y

@orca.column('parcels', cache=True, cache_scope='iteration')
def allowed(parcels):
    df = pd.DataFrame(index=parcels.index)
    df['allowed'] = True
    return df.allowed


def parcel_average_price(use):
    return misc.reindex(orca.get_table('nodes_walk')[use],
                        orca.get_table('parcels').nodeid_walk)


def parcel_is_allowed(form):
    form_to_btype = orca.get_injectable("form_to_btype")
    buildings = orca.get_table("buildings").to_frame(["parcel_id", "building_type_id", "residential_units"])
    lone_house = buildings[
                     (buildings.building_type_id == 81) &
                     (buildings.residential_units == 1)].groupby(by="parcel_id").building_type_id.count() == 1
    orca.add_injectable("lone_house", lone_house)
    zoning = orca.get_table('zoning')
    allowed = [(zoning['type%d' % typ] > 0)
               for typ in form_to_btype[form]]

    s = pd.concat(allowed, axis=1).max(axis=1).\
        reindex(orca.get_table('parcels').index).fillna(False)

    return s.astype("bool") & (~lone_house).reindex(zoning.index, fill_value=True)
    # return s.astype("bool")


@orca.column('parcels', cache=True, cache_scope='iteration')
def max_far(parcels):
    df = pd.DataFrame(index=parcels.index)
    df['max_far'] = orca.get_table('zoning').max_far
    return df.max_far


@orca.column('parcels', cache=True, cache_scope='iteration')
def max_dua(parcels):
    df = pd.DataFrame(index=parcels.index)
    df['max_dua'] = orca.get_table('zoning').max_dua
    return df.max_dua


@orca.column('parcels', cache=True, cache_scope='iteration')
def max_height(parcels):
    df = pd.DataFrame(index=parcels.index)
    df['max_height'] = orca.get_table('zoning').max_height
    return df.max_height


@orca.column('parcels', cache=True, cache_scope='iteration')
def pct_undev(parcels):
    df = pd.DataFrame(index=parcels.index)
    df['pct_undev'] = orca.get_table('zoning').pct_undev
    return df.pct_undev


@orca.column('parcels', cache=True, cache_scope='iteration')
def parcel_size(parcels):
    return parcels.parcel_sqft

@orca.column('parcels')
def ave_unit_size(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_walk.ave_unit_sqft, parcels.nodeid_walk)

@orca.column('parcels', cache=True, cache_scope='iteration')
def total_units(parcels, buildings):
    return buildings.residential_units.groupby(buildings.parcel_id).sum().\
        reindex(parcels.index).fillna(0)

@orca.column('parcels', cache=True, cache_scope='iteration')
def total_job_spaces(parcels, buildings):
    return buildings.job_spaces.groupby(buildings.parcel_id).sum().\
        reindex(parcels.index).fillna(0)

@orca.column('parcels', cache=True, cache_scope='iteration')
def total_sqft(parcels, buildings):
    return buildings.building_sqft.groupby(buildings.parcel_id).sum().\
        reindex(parcels.index).fillna(0)

@orca.column('parcels')
def land_cost(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        return pd.Series(index=parcels.index)
    return (parcels.total_sqft * parcel_average_price("residential")).\
        reindex(parcels.index).fillna(0)

@orca.column('parcels', cache=True, cache_scope='iteration')
def parcel_far(parcels):
    return (parcels.total_sqft/parcels.parcel_sqft).fillna(0)


def parcel_average_price(use):
    if len(orca.get_table('nodes_walk')) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)    
    return misc.reindex(orca.get_table('nodes_walk')[use],
                        orca.get_table('parcels').nodeid_walk)

@orca.column('parcels', cache=True, cache_scope='iteration')
def school_district_achievement(parcels, schools):
    return misc.reindex(parcels['school_id'], schools.to_frame(['dcode', 'totalachievementindex']).\
                        groupby('dcode').totalachievementindex.mean())


@orca.column('parcels', cache=True, cache_scope='iteration')
def drv_nearest_hospital(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_drv').drv_nearest_hospital,
                        parcels.nodeid_drv)

@orca.column('parcels', cache=True, cache_scope='iteration')
def drv_nearest_healthcenter(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_drv').drv_nearest_healthcenter,
                        parcels.nodeid_drv)

@orca.column('parcels', cache=True, cache_scope='iteration')
def drv_nearest_grocery(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_drv').drv_nearest_grocery,
                        parcels.nodeid_drv)

@orca.column('parcels', cache=True, cache_scope='iteration')
def drv_nearest_urgentcare(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_drv').drv_nearest_urgentcare,
                        parcels.nodeid_drv)

@orca.column('parcels', cache=True, cache_scope='iteration')
def drv_nearest_library(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_drv').drv_nearest_library,
                        parcels.nodeid_drv)

@orca.column('parcels', cache=True, cache_scope='iteration')
def drv_nearest_park(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_drv').drv_nearest_park,
                        parcels.nodeid_drv)



@orca.column('parcels', cache=True, cache_scope='iteration')
def walk_nearest_hospital(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_walk').walk_nearest_hospital,
                        parcels.nodeid_walk)  

@orca.column('parcels', cache=True, cache_scope='iteration')
def walk_nearest_grocery(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_walk').walk_nearest_grocery,
                        parcels.nodeid_walk)  


@orca.column('parcels', cache=True, cache_scope='iteration')
def walk_nearest_healthcenter(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_walk').walk_nearest_healthcenter,
                        parcels.nodeid_walk)


@orca.column('parcels', cache=True, cache_scope='iteration')
def walk_nearest_urgentcare(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_walk').walk_nearest_urgentcare,
                        parcels.nodeid_walk)


@orca.column('parcels', cache=True, cache_scope='iteration')
def walk_nearest_library(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_walk').walk_nearest_library,
                        parcels.nodeid_walk)

@orca.column('parcels', cache=True, cache_scope='iteration')
def walk_nearest_park(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_walk').walk_nearest_park,
                        parcels.nodeid_walk)  


@orca.column('parcels', cache=True, cache_scope='forever')
def crime_ucr_rate(crime_rates):
    return crime_rates['ucr_crime_rate']


@orca.column('parcels', cache=True, cache_scope='forever')
def crime_other_rate(crime_rates):
    return crime_rates['other_crime_rate']


#####################
# ZONING VARIABLES
#####################
##@orca.column('zoning', cache=True, cache_scope='iteration')
##def ave_unit_size(zoning, parcels):
##    return misc.reindex(parcels.ave_unit_size, zoning.parcel_id)


#####################
# ZONES VARIABLES
#####################

@orca.column('zones', cache=True, cache_scope='iteration')
def popden(zones, parcels, households):
    return households.persons.groupby(households.zone_id).sum() / parcels.acres.groupby(parcels.zone_id).sum()

##@orca.column('zones', 'jobs_within_30_min', cache=True)
##def jobs_within_30_min(jobs, travel_data):
##    j = pd.DataFrame({'zone_id':jobs.zone_id})
##    td = travel_data.to_frame()
##    zone_ids = np.unique(td.reset_index().to_zone_id)
##    return misc.compute_range(td,
##                                  j.groupby('zone_id').size().reindex(index = zone_ids).fillna(0),
##                                  "am_single_vehicle_to_work_travel_time",
##                                  30, agg=np.sum)

@orca.column('zones', cache=True, cache_scope='iteration')
def households(zones, households):
    print type(households)
    return households.zone_id.groupby(households.zone_id).size()

@orca.column('zones', cache=True, cache_scope='iteration')
def population(zones, households):
    return households.persons.groupby(households.zone_id).sum()

@orca.column('zones', cache=True, cache_scope='iteration')
def employment(zones, jobs, travel_data):
    td = travel_data.to_frame()
    zone_ids = np.unique(td.reset_index().to_zone_id)
    j = pd.DataFrame({'zone_id':jobs.zone_id})
    return j.groupby('zone_id').size().reindex(index = zone_ids).fillna(0)

@orca.column('zones', cache=True, cache_scope='iteration')
def retail_jobs(zones, jobs, travel_data):
    td = travel_data.to_frame()
    zone_ids = np.unique(td.reset_index().to_zone_id)
    j = pd.DataFrame({'zone_id':jobs.zone_id})
    return j.loc[j.sector_id==5,:].groupby('zone_id').size().reindex(index = zone_ids).fillna(0)


@orca.column('zones', cache=True, cache_scope='iteration')
def empden(zones, parcels, households):
    return zones.employment / parcels.acres.groupby(parcels.zone_id).sum()



def logsum_based_accessibility(travel_data, zones, name_attribute, spatial_var):
    td = travel_data.to_frame()
    zones = zones.to_frame(['population', 'employment'])

    td = td.reset_index()
    zones = zones.reset_index()
    unique_zone_ids = np.unique(zones.zone_id.values)

    zones.index = zones.index.values+1
    zone_id_xref = dict(zip(zones.zone_id, zones.index.values))
    apply_xref = lambda x: zone_id_xref[x]

    td =  td[td.from_zone_id.isin(unique_zone_ids)]
    td =  td[td.to_zone_id.isin(unique_zone_ids)]

    td['from_zone_id2'] = td.from_zone_id.apply(apply_xref)
    td['to_zone_id2'] = td.to_zone_id.apply(apply_xref)

    rows = td['from_zone_id2']
    cols = td['to_zone_id2']

    logsums = 0*np.ones((rows.max()+1, cols.max()+1), dtype=td[name_attribute].dtype)
    logsums.put(indices=rows * logsums.shape[1] + cols, values = td[name_attribute])

    population = zones[spatial_var].values
    population = population[np.newaxis, :]

    zone_ids = zones.index.values
    zone_matrix = population * np.exp(logsums[zone_ids,:][:,zone_ids])
    zone_matrix[np.isnan(zone_matrix)] = 0
    results = pd.Series(zone_matrix.sum(axis=1), index = zones.index.values)
    zones['logsum_var'] = results
    zones = zones.reset_index().set_index('zone_id')
    return zones.logsum_var

@orca.column('zones', cache=True, cache_scope='iteration')
def logsum_pop_more_worker_than_car(zones, travel_data):
    name_attribute = 'logsum0'
    spatial_var = 'population'
    return logsum_based_accessibility(travel_data, zones, name_attribute, spatial_var)

@orca.column('zones', cache=True, cache_scope='iteration')
def logsum_pop_less_worker_than_car(zones, travel_data):
    name_attribute = 'logsum1'
    spatial_var = 'population'
    return logsum_based_accessibility(travel_data, zones, name_attribute, spatial_var)

@orca.column('zones', cache=True, cache_scope='iteration')
def logsum_work_more_worker_than_car(zones, travel_data):
    name_attribute = 'logsum0'
    spatial_var = 'employment'
    return logsum_based_accessibility(travel_data, zones, name_attribute, spatial_var)

@orca.column('zones', cache=True, cache_scope='iteration')
def logsum_work_less_worker_than_car(zones, travel_data):
    name_attribute = 'logsum1'
    spatial_var = 'employment'
    return logsum_based_accessibility(travel_data, zones, name_attribute, spatial_var)



#####################
# TRANSIT VARIABLES
#####################

@orca.column('transit_stops', cache=True, cache_scope='iteration')
def nodeid_walk(transit_stops):
    return orca.get_injectable('net_walk').get_node_ids(transit_stops['point_x'], transit_stops['point_y'])



#####################
# SCHOOL VARIABLES
#####################

@orca.column('schools', cache=True, cache_scope='iteration')
def nodeid_drv(schools):
    return orca.get_injectable('net_drv').get_node_ids(schools['point_x'], schools['point_y'])

@orca.column('schools', cache=True, cache_scope='iteration')
def nodeid_walk(schools, parcels):
    return orca.get_injectable('net_walk').get_node_ids(schools['point_x'], schools['point_y'])



#####################
# NODES_DRV VARIABLES
#####################

#net_drv = orca.get_injectable('net_drv')

#'GroceryStores', 'HealthCenters', 'Hospitals', 'Libraries',       'Park_Entrance_points', 'UrgentCare'

def get_nearest(net, dfpoi, cats, searchdis, numpoi, maxdis):
    net.set_pois(''.join(cats), dfpoi['point_x'], dfpoi['point_y'])
    return net.nearest_pois(searchdis, ''.join(cats), num_pois=numpoi, max_distance=maxdis)[1]

    
@orca.column('nodes_drv', cache=True, cache_scope='iteration')
def drv_nearest_hospital(nodes_drv, poi):
    cats = ['Hospitals']
    t = poi.to_frame()[poi.category.isin(cats)]
    return get_nearest(orca.get_injectable('net_drv'), t, cats, 15, 1, 16 )


@orca.column('nodes_drv', cache=True, cache_scope='iteration')
def drv_nearest_grocery(nodes_drv, poi):
    cats = ['GroceryStores']
    t = poi.to_frame()[poi.category.isin(cats)]    
    return get_nearest(orca.get_injectable('net_drv'), t, cats, 15, 1, 16 )

@orca.column('nodes_drv', cache=True, cache_scope='iteration')
def drv_nearest_healthcenter(nodes_drv, poi):
    cats = ['HealthCenters']
    t = poi.to_frame()[poi.category.isin(cats)]    
    return get_nearest(orca.get_injectable('net_drv'), t, cats, 15, 1, 16 )


@orca.column('nodes_drv', cache=True, cache_scope='iteration')
def drv_nearest_library(nodes_drv, poi):
    cats = ['Libraries']
    t = poi.to_frame()[poi.category.isin(cats)]    
    return get_nearest(orca.get_injectable('net_drv'), t, cats, 15, 1, 16 )


@orca.column('nodes_drv', cache=True, cache_scope='iteration')
def drv_nearest_park(nodes_drv, poi):
    cats = ['Park_Entrance_points']
    t = poi.to_frame()[poi.category.isin(cats)]    
    return get_nearest(orca.get_injectable('net_drv'), t, cats, 15, 1, 16 )


@orca.column('nodes_drv', cache=True, cache_scope='iteration')
def drv_nearest_urgentcare(nodes_drv, poi):
    cats = ['UrgentCare']
    t = poi.to_frame()[poi.category.isin(cats)] 
    return get_nearest(orca.get_injectable('net_drv'), t, cats, 15, 1, 16 )




       
#####################
# NODES_WALK VARIABLES
#####################


@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def walk_nearest_hospital(nodes_walk, poi):
    cats = ['Hospitals']
    t = poi.to_frame()[poi.category.isin(cats)]
    print 't',t
    return get_nearest(orca.get_injectable('net_walk'), t, cats, 7920 , 1, 7921 )

@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def walk_nearest_grocery(nodes_walk, poi):
    cats = ['GroceryStores']
    t = poi.to_frame()[poi.category.isin(cats)]    
    return get_nearest(orca.get_injectable('net_walk'), t, cats, 7920 , 1, 7921)

@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def walk_nearest_healthcenter(nodes_walk, poi):
    cats = ['HealthCenters']
    t = poi.to_frame()[poi.category.isin(cats)]     
    return get_nearest(orca.get_injectable('net_walk'), t, cats, 7920 , 1, 7921 )

@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def walk_nearest_library(nodes_walk, poi):
    cats = ['Libraries']
    t = poi.to_frame()[poi.category.isin(cats)]     
    return get_nearest(orca.get_injectable('net_walk'), t, cats, 7920 , 1, 7921 )

@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def walk_nearest_park(nodes_walk, poi):
    cats = ['Park_Entrance_points']
    t = poi.to_frame()[poi.category.isin(cats)]    
    return get_nearest(orca.get_injectable('net_walk'), t, cats, 7920 , 1, 7921 )

@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def walk_nearest_urgentcare(nodes_walk, poi):
    cats = ['UrgentCare']
    t = poi.to_frame()[poi.category.isin(cats)]     
    return get_nearest(orca.get_injectable('net_walk'), t, cats, 7920 , 1, 7921 )











