import numpy as np
import orca
import pandas as pd
from urbansim.utils import misc


#####################
# BUILDINGS VARIABLES
#####################

# @orca.column('buildings', 'school_district_id', cache=True)
# def school_district_id(buildings, parcels):
#     return misc.reindex(parcels.school_district_id, buildings.parcel_id)

@orca.column('buildings', cache=True, cache_scope='iteration')
def hedonic_id(buildings):
    hedonic_id = buildings.large_area_id * 100 + buildings.building_type_id
    hedonic_id.loc[buildings.building_type_id.isin([24, 32, 42, 43, 52, 53, 61, 62])] = buildings.building_type_id
    hedonic_id.loc[hedonic_id == 571] = 371
    hedonic_id.loc[hedonic_id == 584] = 384
    return hedonic_id


@orca.column('buildings', cache=True, cache_scope='iteration')
def general_type(buildings, building_type_map):
    return buildings.building_type_id.map(building_type_map).fillna(0)


# @orca.column('buildings', cache=True, cache_scope='iteration')
# def _node_id(buildings, parcels):
#     return misc.reindex(parcels._node_id, buildings.parcel_id)

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


@orca.column('buildings', cache=True, cache_scope='iteration')
def zone_id(buildings, parcels):
    return misc.reindex(parcels.zone_id, buildings.parcel_id)


@orca.column('buildings', cache=True, cache_scope='iteration')
def city_id(buildings, parcels):
    return misc.reindex(parcels.city_id, buildings.parcel_id)


@orca.column('buildings', cache=True, cache_scope='iteration')
def semmcd(buildings, parcels):
    return misc.reindex(parcels.semmcd, buildings.parcel_id)


@orca.column('buildings', cache=True, cache_scope='iteration')
def large_area_id(buildings, parcels):
    return misc.reindex(parcels.large_area_id, buildings.parcel_id)


@orca.column('buildings', cache=True, cache_scope='iteration')
def popden(buildings, zones):
    return misc.reindex(zones.popden, buildings.zone_id).fillna(0)


@orca.column('buildings', cache=True, cache_scope='iteration')
def residential_sqft(buildings):
    return buildings.sqft_per_unit * buildings.residential_units


@orca.column('buildings', cache=True, cache_scope='iteration')
def building_sqft(buildings):
    return buildings.non_residential_sqft.astype(int) + buildings.residential_sqft


@orca.column('buildings', cache=True, cache_scope='iteration')
def building_sqft_per_job(buildings, building_sqft_per_job):
    b = buildings.to_frame(["building_type_id"])
    bsqft_job = building_sqft_per_job.to_frame()

    return pd.merge(b, bsqft_job,
                    left_on=['building_type_id'],
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
    return buildings.job_spaces


@orca.column('buildings', cache=True, cache_scope='iteration')
def jobs_within_30_min(buildings, zones):
    return misc.reindex(zones.jobs_within_30_min, buildings.zone_id).fillna(0)


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


@orca.column('buildings', 'building_age', cache=True, cache_scope='iteration')
def building_age(buildings, year):
    year_built = buildings.year_built
    year_built[year_built < 1600] = year_built[year_built > 1600].mean()
    age = year - year_built
    return age


@orca.column('buildings', 'building_age_gt_50', cache=True, cache_scope='iteration')
def building_age_gt_50(buildings):
    return (buildings.building_age > 50).astype('int32')


@orca.column('buildings', 'building_age_gt_70', cache=True, cache_scope='iteration')
def building_age_gt_70(buildings):
    return (buildings.building_age > 70).astype('int32')


@orca.column('buildings', 'building_age_gt_80', cache=True, cache_scope='iteration')
def building_age_gt_80(buildings):
    return (buildings.building_age > 80).astype('int32')


@orca.column('buildings', 'building_age_gt_90', cache=True, cache_scope='iteration')
def building_age_gt_90(buildings):
    return (buildings.building_age > 90).astype('int32')


@orca.column('buildings', 'building_age_gt_100', cache=True, cache_scope='iteration')
def building_age_gt_100(buildings):
    return (buildings.building_age > 100).astype('int32')


@orca.column('buildings', 'building_age_le_10', cache=True, cache_scope='iteration')
def building_age_le_10(buildings):
    return (buildings.building_age < 10).astype('int32')


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


### Variable generation functions

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
    # print var_name

    @orca.column('buildings', var_name, cache=True, cache_scope='iteration')
    def func():
        buildings = orca.get_table('buildings')
        return np.log1p(buildings[var_to_log]).fillna(0)

    return func


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


geographic_levels = [('parcels', 'parcel_id'),
                     ('zones', 'zone_id')]
vars_to_dummify = ['city_id', 'building_type_id']
vars_to_log = ['non_residential_sqft', 'building_sqft', 'land_area', 'parcel_sqft', 'sqft_per_unit',
               'parcels_parcel_far', 'sqft_price_nonres']

for geography in geographic_levels:
    geography_name = geography[0]
    geography_id = geography[1]
    if geography_name != 'buildings':
        building_vars = orca.get_table('buildings').columns
        for var in orca.get_table(geography_name).columns:
            if var not in building_vars:
                make_disagg_var(geography_name, 'buildings', var, geography_id)

for dummifiable_var in vars_to_dummify:
    var_cat_ids = np.unique(orca.get_table('buildings')[dummifiable_var]).astype('int')
    for var_cat_id in var_cat_ids:
        if var_cat_id > 0:
            make_dummy_variable(dummifiable_var, var_cat_id)

for var_to_log in vars_to_log:
    make_logged_variable(var_to_log)

emp_sectors = np.arange(18) + 1
for sector in emp_sectors:
    make_employment_proportion_variable(sector)
