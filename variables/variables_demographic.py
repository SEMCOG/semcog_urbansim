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


@orca.column("households", cache=True, cache_scope="iteration")
def qlid(households):
    return (
        (households.income_quartile * 100000 + households.large_area_id)
        .fillna(0)
        .astype("int")
    )


@orca.column("households", cache=True, cache_scope="iteration")
def income_quartile(households):
    return (
        pd.Series(pd.qcut(households.income, 4, labels=False), index=households.index)
        + 1
    )


@orca.column("households", cache=True, cache_scope="iteration")
def income_tercile(households):
    """Region-wide income thirds, backing the lowinc / midinc / highinc segments.

    Recomputed each iteration, like income_quartile, so a segment always holds
    a third of households: the HLCM is trained on relative position in the
    income distribution, and fixed dollar thresholds would drift over a 30-year
    horizon as incomes grow.
    """
    return (
        pd.Series(pd.qcut(households.income, 3, labels=False), index=households.index)
        + 1
    )


orca.add_injectable(
    "household_type_map",
    {
        1: "income_quartile ==1 & persons <=2 & age_of_head >= 65",
        2: "income_quartile ==1 & persons <=2 & age_of_head >= 35 & age_of_head < 65",
        3: "income_quartile ==1 & persons <=2 & age_of_head < 35",
        4: "income_quartile ==1 & persons > 2 & age_of_head >= 65",
        5: "income_quartile ==1 & persons > 2 & age_of_head >=35 & age_of_head < 65",
        6: "income_quartile ==1 & persons > 2 & age_of_head < 35",
        7: "income_quartile in [2,3] & persons <=2 & age_of_head >= 65",
        8: "income_quartile in [2,3] & persons <=2 & age_of_head >= 35 & age_of_head < 65",
        9: "income_quartile in [2,3] & persons <=2 & age_of_head < 35",
        10: "income_quartile in [2,3] & persons > 2 & age_of_head >= 65",
        11: "income_quartile in [2,3] & persons > 2 & age_of_head >=35 & age_of_head < 65",
        12: "income_quartile in [2,3] & persons > 2 & age_of_head < 35",
        13: "income_quartile ==4 & persons <=2 & age_of_head >= 65",
        14: "income_quartile ==4 & persons <=2 & age_of_head >= 35 & age_of_head < 65",
        15: "income_quartile ==4 & persons <=2 & age_of_head < 35",
        16: "income_quartile ==4 & persons > 2 & age_of_head >= 65",
        17: "income_quartile ==4 & persons > 2 & age_of_head >= 35 & age_of_head < 65",
        18: "income_quartile ==4 & persons > 2 & age_of_head < 35",
    },
)


@orca.column("households", cache=True, cache_scope="iteration")
def household_type(households, household_type_map):
    df = households.to_frame(["income_quartile", "age_of_head", "persons"])
    df["household_type"] = 0
    for i, q in household_type_map.items():
        idx = df.query(q).index.values
        df.loc[idx, "household_type"] = i
    return df.household_type.fillna(0)


@orca.column("households", cache=True, cache_scope="iteration")
def hh_type_large_area_id(households):
    return (
        (households.household_type * 10000 + households.large_area_id)
        .fillna(0)
        .astype("int")
    )


@orca.column("households", cache=True, cache_scope="iteration")
def geoid(households, buildings):
    return misc.reindex(buildings.geoid, households.building_id)


@orca.column("households", cache=True, cache_scope="iteration")
def zone_id(households, buildings):
    return misc.reindex(buildings.zone_id, households.building_id)


@orca.column("households", cache=True, cache_scope="iteration")
def tract_id(households, buildings):
    return misc.reindex(buildings.tract_id, households.building_id)


@orca.column("households", cache=True, cache_scope="iteration")
def school_id(households, buildings):
    return misc.reindex(buildings.school_id, households.building_id)


@orca.column("households", cache=True, cache_scope="iteration")
def mi_house_id(households, buildings):
    return misc.reindex(buildings.mi_house_id, households.building_id)


@orca.column("households", cache=True, cache_scope="iteration")
def mi_senate_id(households, buildings):
    return misc.reindex(buildings.mi_senate_id, households.building_id)


@orca.column("households", cache=True, cache_scope="iteration")
def us_congress_id(households, buildings):
    return misc.reindex(buildings.us_congress_id, households.building_id)


# #35 keep it for now as required by the refiner model
# @orca.column("households", cache=True, cache_scope="iteration")
# def b_zone_id(households, buildings):
#     return misc.reindex(buildings.b_zone_id, households.building_id)


@orca.column("households", cache=True, cache_scope="iteration")
def sp_filter(households, buildings):
    return misc.reindex(buildings.sp_filter, households.building_id)


@orca.column("households", cache=True, cache_scope="iteration")
def city_id(households, buildings):
    return misc.reindex(buildings.city_id, households.building_id)


# #35 keep it for now as required by the refiner model
# @orca.column("households", cache=True, cache_scope="iteration")
# def b_city_id(households, buildings):
#     return misc.reindex(buildings.b_city_id, households.building_id)


@orca.column("households", cache=False, cache_scope="iteration")
def semmcd(households, buildings):
    return misc.reindex(buildings.semmcd, households.building_id)


@orca.column("households", cache=True, cache_scope="iteration")
def x(households, buildings):
    return misc.reindex(buildings.x, households.building_id)


@orca.column("households", cache=True, cache_scope="iteration")
def y(households, buildings):
    return misc.reindex(buildings.y, households.building_id)


@orca.column("households", cache=True, cache_scope="iteration")
def lid(households):
    # todo: remove and fix
    return households.large_area_id


@orca.column("households", cache=True, cache_scope="iteration")
def nodeid_walk(households, buildings):
    return misc.reindex(buildings.nodeid_walk, households.building_id)


@orca.column("households", cache=True, cache_scope="iteration")
def nodeid_drv(households, buildings):
    return misc.reindex(buildings.nodeid_drv, households.building_id)


@orca.column("households", cache=True, cache_scope="iteration")
def ln_income(households):
    return np.log1p(households.income)


@orca.column("households", cache=True, cache_scope="iteration")
def low_income(households):
    return (households.income_quartile == 1).astype("int32")


@orca.column("households", cache=True, cache_scope="iteration")
def mid_income(households):
    return (households.income_quartile.isin([2, 3])).astype("int32")


@orca.column("households", cache=True, cache_scope="iteration")
def high_income(households):
    return (households.income_quartile == 4).astype("int32")


@orca.column("households", cache=True, cache_scope="iteration")
def hhsize_gt_2(households):
    return (households.persons > 2).astype("int32")


@orca.column("households", cache=True, cache_scope="iteration")
def hhsize_gt_3(households):
    return (households.persons > 3).astype("int32")


@orca.column("households", cache=True, cache_scope="iteration")
def hhsize_is_1(households):
    return (households.persons == 1).astype("int32")


@orca.column("households", cache=True, cache_scope="iteration")
def hhsize_lt_3(households):
    return (households.persons < 3).astype("int32")


@orca.column("households", cache=True, cache_scope="iteration")
def has_children(households):
    return (households.children > 0).astype("int32")


@orca.column("households", cache=True, cache_scope="iteration")
def has_cars(households):
    return (households.cars > 0).astype("int32")


@orca.column("households", cache=True, cache_scope="iteration")
def no_car(households):
    return (households.cars == 0).astype("int32")


@orca.column("households", cache=True, cache_scope="iteration")
def is_young(households):
    return (households.age_of_head < 35).astype("int32")


@orca.column("households", cache=True, cache_scope="iteration")
def is_senior(households):
    return (households.age_of_head >= 65).astype("int32")


@orca.column("households", cache=True, cache_scope="iteration")
def is_large(households):
    return (households.persons > 4).astype("int32")


@orca.column("households", cache=True, cache_scope="iteration")
def is_race1(households):
    return (households.race_id == 1).astype("int32")


@orca.column("households", cache=True, cache_scope="iteration")
def is_race2(households):
    return (households.race_id == 2).astype("int32")


@orca.column("households", cache=True, cache_scope="iteration")
def is_race3(households):
    return (households.race_id == 3).astype("int32")


@orca.column("households", cache=True, cache_scope="iteration")
def is_race4(households):
    return (households.race_id == 4).astype("int32")


@orca.column("households", cache=True, cache_scope="iteration")
def has_workers(households):
    return (households.workers > 0).astype("int32")


@orca.column("households", cache=True, cache_scope="iteration")
def workers_gt_cars(households):
    return (households.workers > households.cars).astype("int32")


@orca.column("households", cache=True, cache_scope="iteration")
def workers_lte_cars(households):
    return (households.workers <= households.cars).astype("int32")

@orca.column("households", cache=True, cache_scope="iteration")
def seniors(persons, households):
    persons = persons.to_frame(["household_id", "age"])
    # Count persons aged 65+ per household
    seniors_count = persons[persons.age >= 65].groupby("household_id").size()
    # Reindex to include all households, fill missing with 0
    return seniors_count.reindex(households.index).fillna(0).astype(int)

@orca.column("households", cache=True, cache_scope="iteration")
def with_seniors(households):
    households = households.to_frame(["seniors"])
    return (households.seniors > 0).astype('int8')

@orca.column("households", cache=True, cache_scope="iteration")
def without_seniors(households):
    households = households.to_frame(["seniors"])
    return (households.seniors <= 0).astype('int8')

@orca.column("households", cache=True, cache_scope="iteration")
def senior_with_seniors(households):
    return (households.with_seniors == 1).astype('int8')

@orca.column("households", cache=True, cache_scope="iteration")
def senior_without_seniors(households):
    return (households.without_seniors == 1).astype('int8')

#####################
# PERSONS VARIABLES
#####################


@orca.column("persons", cache=True, cache_scope="iteration")
def zone_id(persons, households):
    return misc.reindex(households.zone_id, persons.household_id)

@orca.column("persons", cache=True, cache_scope="iteration")
def tract_id(persons, households):
    return misc.reindex(households.tract_id, persons.household_id)

@orca.column("persons", cache=True, cache_scope="iteration")
def school_id(persons, households):
    return misc.reindex(households.school_id, persons.household_id)

@orca.column("persons", cache=True, cache_scope="iteration")
def mi_house_id(persons, households):
    return misc.reindex(households.mi_house_id, persons.household_id)


@orca.column("persons", cache=True, cache_scope="iteration")
def mi_senate_id(persons, households):
    return misc.reindex(households.mi_senate_id, persons.household_id)


@orca.column("persons", cache=True, cache_scope="iteration")
def us_congress_id(persons, households):
    return misc.reindex(households.us_congress_id, persons.household_id)


# #35 keep it for now as required by the refiner model
# @orca.column("persons", cache=True, cache_scope="iteration")
# def b_zone_id(persons, households):
#     return misc.reindex(households.b_zone_id, persons.household_id)


@orca.column("persons", cache=True, cache_scope="iteration")
def city_id(households, persons):
    return misc.reindex(households.city_id, persons.household_id)


# #35 keep it for now as required by the refiner model
# @orca.column("persons", cache=True, cache_scope="iteration")
# def b_city_id(households, persons):
#     return misc.reindex(households.b_city_id, persons.household_id)


@orca.column("persons", cache=True, cache_scope="iteration")
def semmcd(households, persons):
    return misc.reindex(households.semmcd, persons.household_id)


@orca.column("persons", cache=True, cache_scope="iteration")
def large_area_id(households, persons):
    return misc.reindex(households.large_area_id, persons.household_id)


##@orca.column('persons', cache=True)
##def school_district_id(persons, households):
##    return misc.reindex(households.school_district_id, persons.household_id)


#####################
# GQ VARIABLES
#####################


@orca.column("group_quarters", cache=True, cache_scope="iteration")
def zone_id(group_quarters, buildings):
    return misc.reindex(buildings.zone_id, group_quarters.building_id)

@orca.column("group_quarters", cache=True, cache_scope="iteration")
def tract_id(group_quarters, buildings):
    return misc.reindex(buildings.tract_id, group_quarters.building_id)

@orca.column("group_quarters", cache=True, cache_scope="iteration")
def school_id(group_quarters, buildings):
    return misc.reindex(buildings.school_id, group_quarters.building_id)

@orca.column("group_quarters", cache=True, cache_scope="iteration")
def mi_house_id(group_quarters, buildings):
    return misc.reindex(buildings.mi_house_id, group_quarters.building_id)


@orca.column("group_quarters", cache=True, cache_scope="iteration")
def mi_senate_id(group_quarters, buildings):
    return misc.reindex(buildings.mi_senate_id, group_quarters.building_id)


@orca.column("group_quarters", cache=True, cache_scope="iteration")
def us_congress_id(group_quarters, buildings):
    return misc.reindex(buildings.us_congress_id, group_quarters.building_id)

# @orca.column("group_quarters", cache=True, cache_scope="iteration")
# def b_zone_id(group_quarters, buildings):
#     return misc.reindex(buildings.b_zone_id, group_quarters.building_id)


@orca.column("group_quarters", cache=True, cache_scope="iteration")
def city_id(group_quarters, buildings):
    return misc.reindex(buildings.city_id, group_quarters.building_id)


# #35
# @orca.column('group_quarters', cache=True, cache_scope='iteration')
# def b_city_id(group_quarters, buildings):
#     return misc.reindex(buildings.b_city_id, group_quarters.building_id)


@orca.column("group_quarters", cache=True, cache_scope="iteration")
def semmcd(group_quarters, buildings):
    return misc.reindex(buildings.semmcd, group_quarters.building_id)


@orca.column("group_quarters", cache=True, cache_scope="iteration")
def large_area_id(group_quarters, buildings):
    return misc.reindex(buildings.large_area_id, group_quarters.building_id)


@orca.column("households", cache=True, cache_scope="iteration")
def residential_units(households, buildings):
    return misc.reindex(buildings.residential_units, households.building_id)


@orca.column("households", cache=True, cache_scope="iteration")
def year_built(households, buildings):
    return misc.reindex(buildings.year_built, households.building_id)


@orca.column("households", cache=True, cache_scope="iteration")
def mcd_model_quota(households, buildings):
    return misc.reindex(buildings.mcd_model_quota, households.building_id)

@orca.column("households", cache=True, cache_scope="iteration")
def hh_size_1p(households):
    return (households.persons==1).astype('int8')

@orca.column("households", cache=True, cache_scope="iteration")
def hh_size_2p(households):
    return (households.persons==2).astype('int8')

@orca.column("households", cache=True, cache_scope="iteration")
def hh_size_3p(households):
    return (households.persons==3).astype('int8')

@orca.column("households", cache=True, cache_scope="iteration")
def hh_size_4pp(households):
    return (households.persons>=4).astype('int8')

@orca.column("households", cache=True, cache_scope="iteration")
def children_has_children(households):
    return (households.children>0).astype('int8')

@orca.column("households", cache=True, cache_scope="iteration")
def children_no_children(households):
    return (households.children==0).astype('int8')

@orca.column("households", cache=True, cache_scope="iteration")
def ownership_own(households):
    return households.ten.lt(3).astype('int8')

@orca.column("households", cache=True, cache_scope="iteration")
def ownership_rent(households):
    return households.ten.ge(3).astype('int8')

@orca.column("households", cache=True, cache_scope="iteration")
def aoh_lt35(households):
    return households.age_of_head.lt(35).astype('int8')

@orca.column("households", cache=True, cache_scope="iteration")
def aoh_lt65(households):
    return (households.age_of_head.ge(35) & households.age_of_head.lt(65)).astype('int8')

@orca.column("households", cache=True, cache_scope="iteration")
def aoh_65p(households):
    return households.age_of_head.ge(65).astype('int8')

@orca.column("households", cache=True, cache_scope="iteration")
def incomeqt_incqt1(households):
    return (households.income_quartile == 1).astype('int8')

@orca.column("households", cache=True, cache_scope="iteration")
def incomeqt_incqt2(households):
    return (households.income_quartile == 2).astype('int8')

@orca.column("households", cache=True, cache_scope="iteration")
def incomeqt_incqt3(households):
    return (households.income_quartile == 3).astype('int8')

@orca.column("households", cache=True, cache_scope="iteration")
def incomeqt_incqt4(households):
    return (households.income_quartile == 4).astype('int8')

# HLCM income segments -- thirds, not the quartile grouping. These must stay in
# step with forecast_estimation.utils.add_hh_indicators, which cuts the same way
# at estimation time; the quartile grouping below gave 25/50/25 while estimation
# assumed equal segments. income_quartile above is unchanged and still backs
# household_type_map and the output indicators.
@orca.column("households", cache=True, cache_scope="iteration")
def income_highinc(households):
    return (households.income_tercile == 3).astype('int8')

@orca.column("households", cache=True, cache_scope="iteration")
def income_midinc(households):
    return (households.income_tercile == 2).astype('int8')

@orca.column("households", cache=True, cache_scope="iteration")
def income_lowinc(households):
    return (households.income_tercile == 1).astype('int8')


#####################
# TRAVEL SURVEY VARIABLES (households)
# Block-group behavioral vars carried down parcel -> building -> household so
# household models can read them. Mirrors the parcel/building registrations.
#####################

from variables.variables_parcel import SURVEY_VARS as _SURVEY_VARS


def _make_household_survey_var(var_name):
    """Register one household-level travel survey column via building broadcast."""

    @orca.column("households", var_name, cache=True, cache_scope="forever")
    def _col(households, buildings):
        if var_name not in buildings.columns:
            return pd.Series(np.nan, index=households.index)
        return misc.reindex(buildings[var_name], households.building_id).fillna(0)

    return _col


for _sv in _SURVEY_VARS:
    _make_household_survey_var(_sv)