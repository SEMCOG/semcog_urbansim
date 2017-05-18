import orca
import numpy as np
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


@orca.column('households', 'ln_income', cache=True, cache_scope='iteration')
def ln_income(households):
    return np.log1p(households.income)


@orca.column('households', 'low_income', cache=True, cache_scope='iteration')
def low_income(households):
    return (households.income_quartile == 1).astype('int32')


@orca.column('households', 'mid_income', cache=True, cache_scope='iteration')
def mid_income(households):
    return (households.income_quartile.isin([2, 3])).astype('int32')


@orca.column('households', 'high_income', cache=True, cache_scope='iteration')
def high_income(households):
    return (households.income_quartile == 4).astype('int32')


@orca.column('households', 'hhsize_gt_2', cache=True, cache_scope='iteration')
def hhsize_gt_2(households):
    return (households.persons > 2).astype('int32')


@orca.column('households', 'hhsize_is_1', cache=True, cache_scope='iteration')
def hhsize_is_1(households):
    return (households.persons == 1).astype('int32')


@orca.column('households', 'has_children', cache=True, cache_scope='iteration')
def has_children(households):
    return (households.children > 0).astype('int32')


@orca.column('households', 'is_young', cache=True, cache_scope='iteration')
def has_children(households):
    return (households.age_of_head < 35).astype('int32')


@orca.column('households', 'is_race1', cache=True, cache_scope='iteration')
def is_race1(households):
    return (households.race_id == 1).astype('int32')


@orca.column('households', 'is_race2', cache=True, cache_scope='iteration')
def is_race2(households):
    return (households.race_id == 2).astype('int32')


@orca.column('households', 'is_race3', cache=True, cache_scope='iteration')
def is_race3(households):
    return (households.race_id == 3).astype('int32')


@orca.column('households', 'is_race4', cache=True, cache_scope='iteration')
def is_race4(households):
    return (households.race_id == 4).astype('int32')


@orca.column('households', 'has_workers', cache=True, cache_scope='iteration')
def has_workers(households):
    return (households.workers > 0).astype('int32')


@orca.column('households', 'workers_gt_cars', cache=True, cache_scope='iteration')
def workers_gt_cars(households):
    return (households.workers > households.cars).astype('int32')


@orca.column('households', 'workers_lte_cars', cache=True, cache_scope='iteration')
def workers_lte_cars(households):
    return (households.workers <= households.cars).astype('int32')


#####################
# PERSONS VARIABLES
#####################

@orca.column('persons', cache=True, cache_scope='iteration')
def zone_id(persons, households):
    return misc.reindex(households.zone_id, persons.household_id)

##@orca.column('persons', 'school_district_id', cache=True)
##def school_district_id(persons, households):
##    return misc.reindex(households.school_district_id, persons.household_id)
