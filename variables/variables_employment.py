import orca
import numpy as np
import pandas as pd
from urbansim.utils import misc


#####################
# JOBS VARIABLES
#####################

@orca.column('jobs', cache=True, cache_scope='iteration')
def slid(jobs):
    return (jobs.sector_id*100000 + jobs.large_area_id).fillna(0).astype('int')


@orca.column('jobs', cache=True, cache_scope='iteration')
def zone_id(jobs, buildings):
    return misc.reindex(buildings.zone_id, jobs.building_id)

@orca.column('jobs', cache=True, cache_scope='iteration')
def school_id(jobs, buildings):
    return misc.reindex(buildings.school_id, jobs.building_id)


@orca.column('jobs', cache=True, cache_scope='iteration')
def city_id(jobs, buildings):
    return misc.reindex(buildings.city_id, jobs.building_id)

@orca.column('jobs', cache=True, cache_scope='iteration')
def mi_house_id(jobs, buildings):
    return misc.reindex(buildings.mi_house_id, jobs.building_id)

@orca.column('jobs', cache=True, cache_scope='iteration')
def mi_senate_id(jobs, buildings):
    return misc.reindex(buildings.mi_senate_id, jobs.building_id)

@orca.column('jobs', cache=True, cache_scope='iteration')
def us_congress_id(jobs, buildings):
    return misc.reindex(buildings.us_congress_id, jobs.building_id)

# #35
# @orca.column('jobs', cache=True, cache_scope='iteration')
# def b_city_id(jobs, buildings):
#     return misc.reindex(buildings.b_city_id, jobs.building_id)


@orca.column('jobs', cache=True, cache_scope='iteration')
def semmcd(jobs, buildings):
    return misc.reindex(buildings.semmcd, jobs.building_id)


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
def nodeid_walk(jobs, buildings):
    return misc.reindex(buildings.nodeid_walk, jobs.building_id)


@orca.column('jobs', cache=True, cache_scope='iteration')
def nodeid_drv(jobs, buildings):
    return misc.reindex(buildings.nodeid_drv, jobs.building_id)

### ELCM NN dummy variables definition
@orca.column('jobs', cache=True, cache_scope='iteration')
def home_based_homebased(jobs):
    return jobs.home_based_status == 1
    
@orca.column('jobs', cache=True, cache_scope='iteration')
def home_based_nonhomebased(jobs):
    return jobs.home_based_status == 0

@orca.column('jobs', cache=True, cache_scope='iteration')
def job_sector_sector3(jobs):
    return jobs.sector_id == 3

@orca.column('jobs', cache=True, cache_scope='iteration')
def job_sector_sector6(jobs):
    return jobs.sector_id == 6 

@orca.column('jobs', cache=True, cache_scope='iteration')
def job_sector_sector10(jobs):
    return jobs.sector_id == 10 

@orca.column('jobs', cache=True, cache_scope='iteration')
def job_sector_sector11(jobs):
    return jobs.sector_id == 11

@orca.column('jobs', cache=True, cache_scope='iteration')
def job_sector_sector14(jobs):
    return jobs.sector_id == 14

@orca.column('jobs', cache=True, cache_scope='iteration')
def job_sector_sector9(jobs):
    return jobs.sector_id == 9

@orca.column('jobs', cache=True, cache_scope='iteration')
def job_sector_sector4(jobs):
    return jobs.sector_id == 4

@orca.column('jobs', cache=True, cache_scope='iteration')
def job_sector_sector2(jobs):
    return jobs.sector_id == 2

@orca.column('jobs', cache=True, cache_scope='iteration')
def job_sector_sector5(jobs):
    return jobs.sector_id == 5

@orca.column('jobs', cache=True, cache_scope='iteration')
def job_sector_sector16(jobs):
    return jobs.sector_id == 16

@orca.column('jobs', cache=True, cache_scope='iteration')
def job_sector_sector17(jobs):
    return jobs.sector_id == 17

@orca.column('jobs', cache=True, cache_scope='iteration')
def job_sector_sector8(jobs):
    return jobs.sector_id == 8