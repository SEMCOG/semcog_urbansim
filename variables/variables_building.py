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
    return buildings.non_residential_sqft + buildings.sqft_per_unit * buildings.residential_units


@orca.column('buildings', cache=True, cache_scope='iteration')
def building_sqft_per_job(buildings, building_sqft_per_job):
    b = pd.DataFrame({'zone_id': buildings.zone_id, 'building_type_id': buildings.building_type_id})
    bsqft_job = building_sqft_per_job.to_frame()

    return pd.merge(b, bsqft_job,
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
def crime_ucr_rate(buildings, parcels):
    return misc.reindex(parcels.crime_ucr_rate, buildings.parcel_id).fillna(0)


@orca.column('buildings', cache=True, cache_scope='iteration')
def crime_other_rate(buildings, parcels):
    return misc.reindex(parcels.crime_other_rate, buildings.parcel_id).fillna(0)


### accessibilities auto
@orca.column('buildings', cache=True, cache_scope='iteration')
def drv_nearest_hospital(buildings, parcels):
    return misc.reindex(parcels.drv_nearest_hospital, buildings.parcel_id)


@orca.column('buildings', cache=True, cache_scope='iteration')
def drv_nearest_healthcenter(buildings, parcels):
    return misc.reindex(parcels.drv_nearest_healthcenter, buildings.parcel_id)


@orca.column('buildings', cache=True, cache_scope='iteration')
def drv_nearest_grocery(buildings, parcels):
    return misc.reindex(parcels.drv_nearest_grocery, buildings.parcel_id)


@orca.column('buildings', cache=True, cache_scope='iteration')
def drv_nearest_urgentcare(buildings, parcels):
    return misc.reindex(parcels.drv_nearest_urgentcare, buildings.parcel_id)


@orca.column('buildings', cache=True, cache_scope='iteration')
def drv_nearest_library(buildings, parcels):
    return misc.reindex(parcels.drv_nearest_library, buildings.parcel_id)


@orca.column('buildings', cache=True, cache_scope='iteration')
def drv_nearest_park(buildings, parcels):
    return misc.reindex(parcels.drv_nearest_park, buildings.parcel_id)


### accessibilities walk
@orca.column('buildings', cache=True, cache_scope='iteration')
def walk_nearest_hospital(buildings, parcels):
    return misc.reindex(parcels.walk_nearest_hospital, buildings.parcel_id)


@orca.column('buildings', cache=True, cache_scope='iteration')
def walk_nearest_grocery(buildings, parcels):
    return misc.reindex(parcels.walk_nearest_grocery, buildings.parcel_id)


@orca.column('buildings', cache=True, cache_scope='iteration')
def walk_nearest_healthcenter(buildings, parcels):
    return misc.reindex(parcels.walk_nearest_healthcenter, buildings.parcel_id)


@orca.column('buildings', cache=True, cache_scope='iteration')
def walk_nearest_urgentcare(buildings, parcels):
    return misc.reindex(parcels.walk_nearest_urgentcare, buildings.parcel_id)


@orca.column('buildings', cache=True, cache_scope='iteration')
def walk_nearest_library(buildings, parcels):
    return misc.reindex(parcels.walk_nearest_library, buildings.parcel_id)


@orca.column('buildings', cache=True, cache_scope='iteration')
def walk_nearest_park(buildings, parcels):
    return misc.reindex(parcels.walk_nearest_park, buildings.parcel_id)
