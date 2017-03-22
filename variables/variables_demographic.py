import orca
import pandas as pd
from urbansim.utils import misc


#####################
# HOUSEHOLDS VARIABLES
#####################
##
##@orca.column('households', 'school_district_id', cache=True)
##def school_district_id(households, buildings):
##    return misc.reindex(buildings.school_district_id, households.building_id)

@orca.column('households', cache=True, cache_scope='iteration')
def income_quartile(households):
    return pd.Series(pd.qcut(households.income, 4, labels=False),
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
