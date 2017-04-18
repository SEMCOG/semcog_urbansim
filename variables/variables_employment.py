import orca
import numpy as np
import pandas as pd
from urbansim.utils import misc


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
# NON-JOB EMPLOYMENT VARIABLES
#####################

@orca.column('buildings', 'b_total_jobs', cache=True, cache_scope='iteration')
def b_total_jobs(jobs, buildings):
    jobs_by_b = jobs.building_id.groupby(jobs.building_id).size()
    return pd.Series(index=buildings.index, data=jobs_by_b).fillna(0)

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

emp_sectors = np.arange(18) + 1
for sector in emp_sectors:
    make_employment_proportion_variable(sector)
    make_employment_density_variable(sector)
