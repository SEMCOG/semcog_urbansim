import orca
import numpy as np
import pandas as pd
from urbansim.utils import misc


#####################
# HOUSEHOLDS VARIABLES
#####################


## @orca.column('households', cache=True)
## def school_district_id(households, buildings):
##     return misc.reindex(buildings.school_district_id, households.building_id)


@orca.column('households', cache=True, cache_scope='iteration')
def qlid(households):
    return (households.income_quartile*100000 + households.large_area_id).fillna(0).astype('int')


@orca.column('households', cache=True, cache_scope='iteration')
def income_quartile(households):
    return pd.Series(pd.qcut(households.income, 4, labels=False),
                     index=households.index) + 1


orca.add_injectable("household_type_map", {
    1: 'income_quartile ==1 & persons <=2 & age_of_head >= 65',
    2: 'income_quartile ==1 & persons <=2 & age_of_head >= 35 & age_of_head < 65',
    3: 'income_quartile ==1 & persons <=2 & age_of_head < 35',
    4: 'income_quartile ==1 & persons > 2 & age_of_head >= 65',
    5: 'income_quartile ==1 & persons > 2 & age_of_head >=35 & age_of_head < 65',
    6: 'income_quartile ==1 & persons > 2 & age_of_head < 35',
    7: 'income_quartile in [2,3] & persons <=2 & age_of_head >= 65',
    8: 'income_quartile in [2,3] & persons <=2 & age_of_head >= 35 & age_of_head < 65',
    9: 'income_quartile in [2,3] & persons <=2 & age_of_head < 35',
    10: 'income_quartile in [2,3] & persons > 2 & age_of_head >= 65',
    11: 'income_quartile in [2,3] & persons > 2 & age_of_head >=35 & age_of_head < 65',
    12: 'income_quartile in [2,3] & persons > 2 & age_of_head < 35',
    13: 'income_quartile ==4 & persons <=2 & age_of_head >= 65',
    14: 'income_quartile ==4 & persons <=2 & age_of_head >= 35 & age_of_head < 65',
    15: 'income_quartile ==4 & persons <=2 & age_of_head < 35',
    16: 'income_quartile ==4 & persons > 2 & age_of_head >= 65',
    17: 'income_quartile ==4 & persons > 2 & age_of_head >= 35 & age_of_head < 65',
    18: 'income_quartile ==4 & persons > 2 & age_of_head < 35'
})


@orca.column('households', cache=True, cache_scope='iteration')
def household_type(households, household_type_map):
    df = households.to_frame(['income_quartile', 'age_of_head', 'persons'])
    df['household_type'] = 0
    for i, q in household_type_map.iteritems():
        idx = df.query(q).index.values
        df.loc[idx, 'household_type'] = i
    return df.household_type.fillna(0)


@orca.column('households', cache=True, cache_scope='iteration')
def hh_type_large_area_id(households):
    return (households.household_type*10000 + households.large_area_id).fillna(0).astype('int')


@orca.column('households', cache=True, cache_scope='iteration')
def zone_id(households, buildings):
    return misc.reindex(buildings.zone_id, households.building_id)


@orca.column('households', cache=True, cache_scope='iteration')
def b_zone_id(households, buildings):
    return misc.reindex(buildings.b_zone_id, households.building_id)


@orca.column('households', cache=True, cache_scope='iteration')
def city_id(households, buildings):
    return misc.reindex(buildings.city_id, households.building_id)


@orca.column('households', cache=True, cache_scope='iteration')
def b_city_id(households, buildings):
    return misc.reindex(buildings.b_city_id, households.building_id)


@orca.column('households', cache=True, cache_scope='iteration')
def semmcd(households, buildings):
    return misc.reindex(buildings.semmcd, households.building_id)


@orca.column('households', cache=True, cache_scope='iteration')
def x(households, buildings):
    return misc.reindex(buildings.x, households.building_id)


@orca.column('households', cache=True, cache_scope='iteration')
def y(households, buildings):
    return misc.reindex(buildings.y, households.building_id)


@orca.column('households', cache=True, cache_scope='iteration')
def lid(households):
    # todo: remove and fix
    return households.large_area_id


@orca.column('households', cache=True, cache_scope='iteration')
def nodeid_walk(households, buildings):
    return misc.reindex(buildings.nodeid_walk, households.building_id)


@orca.column('households', cache=True, cache_scope='iteration')
def nodeid_drv(households, buildings):
    return misc.reindex(buildings.nodeid_drv, households.building_id)


@orca.column('households', cache=True, cache_scope='iteration')
def ln_income(households):
    return np.log1p(households.income)


@orca.column('households', cache=True, cache_scope='iteration')
def low_income(households):
    return (households.income_quartile == 1).astype('int32')


@orca.column('households', cache=True, cache_scope='iteration')
def mid_income(households):
    return (households.income_quartile.isin([2, 3])).astype('int32')


@orca.column('households', cache=True, cache_scope='iteration')
def high_income(households):
    return (households.income_quartile == 4).astype('int32')


@orca.column('households', cache=True, cache_scope='iteration')
def hhsize_gt_2(households):
    return (households.persons > 2).astype('int32')


@orca.column('households', cache=True, cache_scope='iteration')
def hhsize_gt_3(households):
    return (households.persons > 3).astype('int32')


@orca.column('households', cache=True, cache_scope='iteration')
def hhsize_is_1(households):
    return (households.persons == 1).astype('int32')

@orca.column('households', cache=True, cache_scope='iteration')
def hhsize_lt_3(households):
    return (households.persons < 3).astype('int32')

@orca.column('households', cache=True, cache_scope='iteration')
def has_children(households):
    return (households.children > 0).astype('int32')


@orca.column('households', cache=True, cache_scope='iteration')
def is_young(households):
    return (households.age_of_head < 35).astype('int32')


@orca.column('households', cache=True, cache_scope='iteration')
def is_race1(households):
    return (households.race_id == 1).astype('int32')


@orca.column('households', cache=True, cache_scope='iteration')
def is_race2(households):
    return (households.race_id == 2).astype('int32')


@orca.column('households', cache=True, cache_scope='iteration')
def is_race3(households):
    return (households.race_id == 3).astype('int32')


@orca.column('households', cache=True, cache_scope='iteration')
def is_race4(households):
    return (households.race_id == 4).astype('int32')


@orca.column('households', cache=True, cache_scope='iteration')
def has_workers(households):
    return (households.workers > 0).astype('int32')


@orca.column('households', cache=True, cache_scope='iteration')
def workers_gt_cars(households):
    return (households.workers > households.cars).astype('int32')


@orca.column('households', cache=True, cache_scope='iteration')
def workers_lte_cars(households):
    return (households.workers <= households.cars).astype('int32')


#####################
# PERSONS VARIABLES
#####################

@orca.column('persons', cache=True, cache_scope='iteration')
def zone_id(persons, households):
    return misc.reindex(households.zone_id, persons.household_id)


@orca.column('persons', cache=True, cache_scope='iteration')
def b_zone_id(persons, households):
    return misc.reindex(households.b_zone_id, persons.household_id)


@orca.column('persons', cache=True, cache_scope='iteration')
def city_id(households, persons):
    return misc.reindex(households.city_id, persons.household_id)


@orca.column('persons', cache=True, cache_scope='iteration')
def b_city_id(households, persons):
    return misc.reindex(households.b_city_id, persons.household_id)


@orca.column('persons', cache=True, cache_scope='iteration')
def semmcd(households, persons):
    return misc.reindex(households.semmcd, persons.household_id)


@orca.column('persons', cache=True, cache_scope='iteration')
def large_area_id(households, persons):
    return misc.reindex(households.large_area_id, persons.household_id)

##@orca.column('persons', cache=True)
##def school_district_id(persons, households):
##    return misc.reindex(households.school_district_id, persons.household_id)


#####################
# GQ VARIABLES
#####################

@orca.column('group_quarters', cache=True, cache_scope='iteration')
def zone_id(group_quarters, buildings):
    return misc.reindex(buildings.zone_id, group_quarters.building_id)


@orca.column('group_quarters', cache=True, cache_scope='iteration')
def b_zone_id(group_quarters, buildings):
    return misc.reindex(buildings.b_zone_id, group_quarters.building_id)


@orca.column('group_quarters', cache=True, cache_scope='iteration')
def city_id(group_quarters, buildings):
    return misc.reindex(buildings.city_id, group_quarters.building_id)


@orca.column('group_quarters', cache=True, cache_scope='iteration')
def b_city_id(group_quarters, buildings):
    return misc.reindex(buildings.b_city_id, group_quarters.building_id)


@orca.column('group_quarters', cache=True, cache_scope='iteration')
def semmcd(group_quarters, buildings):
    return misc.reindex(buildings.semmcd, group_quarters.building_id)


@orca.column('group_quarters', cache=True, cache_scope='iteration')
def large_area_id(group_quarters, buildings):
    return misc.reindex(buildings.large_area_id, group_quarters.building_id)
