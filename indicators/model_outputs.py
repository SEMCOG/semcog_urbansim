import numpy as np
import orca
import pandas as pd
from urbansim.utils import misc


@orca.column("parcels", cache=True, cache_scope="iteration")
def parcel_is_allowed_residential():
    import variables

    return variables.parcel_is_allowed("residential")


@orca.column("parcels", cache=True, cache_scope="iteration")
def parcel_is_allowed_demolition():
    import variables

    return variables.parcel_is_allowed()


@orca.column("households", cache=True, cache_scope="iteration")
def seniors(persons):
    persons = persons.to_frame(["household_id", "age"])
    return persons[persons.age >= 65].groupby("household_id").size()


def make_indicators(tab, geo_id):
    @orca.column(tab, cache=True, cache_scope="iteration")
    def hh(households):
        households = households.to_frame([geo_id])
        return households.groupby(geo_id).size()

    @orca.column(tab, cache=True, cache_scope="iteration")
    def hh_pop(households):
        households = households.to_frame([geo_id, "persons"])
        return households.groupby(geo_id).persons.sum()

    @orca.column(tab, cache=True, cache_scope="iteration")
    def gq_pop(group_quarters):
        group_quarters = group_quarters.to_frame([geo_id])
        return group_quarters.groupby(geo_id).size()

    @orca.column(tab, cache=True, cache_scope="iteration")
    def pop():
        df = orca.get_table(tab)
        df = df.to_frame(["hh_pop", "gq_pop"]).fillna(0)
        return df.hh_pop + df.gq_pop

    @orca.column(tab, cache=True, cache_scope="iteration")
    def housing_units(buildings):
        buildings = buildings.to_frame([geo_id, "residential_units"])
        return buildings.groupby(geo_id).residential_units.sum()

    @orca.column(tab, cache=True, cache_scope="iteration")
    def hu_filter(buildings):
        buildings = buildings.to_frame([geo_id, "hu_filter"])
        return buildings.groupby(geo_id).hu_filter.sum()

    @orca.column(tab, cache=True, cache_scope="iteration")
    def parcel_is_allowed_residential(parcels):
        parcels = parcels.to_frame([geo_id, "parcel_is_allowed_residential"])
        return parcels.groupby(geo_id).parcel_is_allowed_residential.sum()

    @orca.column(tab, cache=True, cache_scope="iteration")
    def parcel_is_allowed_demolition(parcels):
        parcels = parcels.to_frame([geo_id, "parcel_is_allowed_demolition"])
        return parcels.groupby(geo_id).parcel_is_allowed_demolition.sum()

    @orca.column(tab, cache=True, cache_scope="iteration")
    def job_spaces(buildings):
        buildings = buildings.to_frame([geo_id, "job_spaces"])
        return buildings.groupby(geo_id).job_spaces.sum()

    @orca.column(tab, cache=True, cache_scope="iteration")
    def nonres_sqft(buildings, form_to_btype):
        buildings = buildings.to_frame([geo_id, "non_residential_sqft"])
        # hold for change
        # exclude all residential building_type
        # buildings = buildings[~buildings.building_type_id.isin(form_to_btype["residential"])]
        return buildings.groupby(geo_id).non_residential_sqft.sum()

    @orca.column(tab, cache=True, cache_scope="iteration")
    def res_sqft(buildings):
        buildings = buildings.to_frame([geo_id, "residential_sqft"])
        return buildings.groupby(geo_id).residential_sqft.sum()

    def make_building_sqft_type(i):
        @orca.column(
            tab,
            "building_sqft_type_" + str(i).zfill(2),
            cache=True,
            cache_scope="iteration",
        )
        def res_sqft_type(buildings):
            buildings = buildings.to_frame(
                [geo_id, "building_type_id", "building_sqft"]
            )
            return (
                buildings[buildings.building_type_id == i]
                .groupby(geo_id)
                .building_sqft.sum()
            )

    for i in orca.get_table("building_types").index:
        make_building_sqft_type(i)

    @orca.column(tab, cache=True, cache_scope="iteration")
    def buildings(buildings):
        buildings = buildings.to_frame([geo_id])
        return buildings.groupby(geo_id).size()

    @orca.column(tab, cache=True, cache_scope="iteration")
    def household_size():
        df = orca.get_table(tab)
        df = df.to_frame(["hh", "hh_pop"])
        return df.hh_pop / df.hh

    @orca.column(tab, cache=True, cache_scope="iteration")
    def vacant_units():
        df = orca.get_table(tab)
        df = df.to_frame(["hh", "housing_units"]).fillna(0)
        return df.housing_units - df.hh

    @orca.column(tab, cache=True, cache_scope="iteration")
    def res_vacancy_rate():
        df = orca.get_table(tab)
        df = df.to_frame(["vacant_units", "housing_units"]).fillna(0)
        return df.vacant_units / df.housing_units

    @orca.column(tab, cache=True, cache_scope="iteration")
    def nonres_vacancy_rate():
        df = orca.get_table(tab)
        df = df.to_frame(["jobs_total", "jobs_home_based", "job_spaces"]).fillna(0)
        return 1.0 - (df.jobs_total - df.jobs_home_based) / df.job_spaces

    @orca.column(tab, cache=True, cache_scope="iteration")
    def incomes_1(households):
        households = households.to_frame([geo_id, "income_quartile"])
        return households[households.income_quartile == 1].groupby(geo_id).size()

    @orca.column(tab, cache=True, cache_scope="iteration")
    def pct_incomes_1():
        df = orca.get_table(tab)
        df = df.to_frame(["incomes_1", "hh"])
        return 1.0 * df.incomes_1 / df.hh

    @orca.column(tab, cache=True, cache_scope="iteration")
    def incomes_2(households):
        households = households.to_frame([geo_id, "income_quartile"])
        return households[households.income_quartile == 2].groupby(geo_id).size()

    @orca.column(tab, cache=True, cache_scope="iteration")
    def pct_incomes_2():
        df = orca.get_table(tab)
        df = df.to_frame(["incomes_2", "hh"])
        return 1.0 * df.incomes_2 / df.hh

    @orca.column(tab, cache=True, cache_scope="iteration")
    def incomes_3(households):
        households = households.to_frame([geo_id, "income_quartile"])
        return households[households.income_quartile == 3].groupby(geo_id).size()

    @orca.column(tab, cache=True, cache_scope="iteration")
    def pct_incomes_3():
        df = orca.get_table(tab)
        df = df.to_frame(["incomes_3", "hh"])
        return 1.0 * df.incomes_3 / df.hh

    @orca.column(tab, cache=True, cache_scope="iteration")
    def incomes_4(households):
        households = households.to_frame([geo_id, "income_quartile"])
        return households[households.income_quartile == 4].groupby(geo_id).size()

    @orca.column(tab, cache=True, cache_scope="iteration")
    def pct_incomes_4():
        df = orca.get_table(tab)
        df = df.to_frame(["incomes_4", "hh"])
        return 1.0 * df.incomes_4 / df.hh

    @orca.column(tab, cache=True, cache_scope="iteration")
    def with_children(households):
        households = households.to_frame([geo_id, "children"])
        return households[households.children > 0].groupby(geo_id).size()

    @orca.column(tab, cache=True, cache_scope="iteration")
    def pct_with_children():
        df = orca.get_table(tab)
        df = df.to_frame(["with_children", "hh"])
        return 1.0 * df.with_children / df.hh

    @orca.column(tab, cache=True, cache_scope="iteration")
    def without_children(households):
        households = households.to_frame([geo_id, "children"])
        return households[households.children.fillna(0) == 0].groupby(geo_id).size()

    @orca.column(tab, cache=True, cache_scope="iteration")
    def with_seniors(households):
        households = households.to_frame([geo_id, "seniors"])
        return households[households.seniors > 0].groupby(geo_id).size()

    @orca.column(tab, cache=True, cache_scope="iteration")
    def pct_with_seniors():
        df = orca.get_table(tab)
        df = df.to_frame(["with_seniors", "hh"])
        return 1.0 * df.with_seniors / df.hh

    @orca.column(tab, cache=True, cache_scope="iteration")
    def without_seniors(households):
        households = households.to_frame([geo_id, "seniors"])
        return households[households.seniors.fillna(0) == 0].groupby(geo_id).size()

    def make_hh_size(r, plus=False):
        hh_name = "hh_size_" + str(r)
        if plus:
            hh_name += "p"

        @orca.column(tab, hh_name, cache=True, cache_scope="iteration")
        def hh_size(households):
            households = households.to_frame([geo_id, "persons"])
            return (
                households[
                    (households.persons == r) | (plus & (households.persons > r))
                ]
                .groupby(geo_id)
                .size()
            )

        w_child = "with_children_" + hh_name

        @orca.column(tab, w_child, cache=True, cache_scope="iteration")
        def hh_size(households):
            households = households.to_frame([geo_id, "persons", "children"])
            return (
                households[
                    ((households.persons == r) | (plus & (households.persons > r)))
                    & (households.children > 0)
                ]
                .groupby(geo_id)
                .size()
            )

        wo_child = "without_children_" + hh_name

        @orca.column(tab, wo_child, cache=True, cache_scope="iteration")
        def hh_size(households):
            households = households.to_frame([geo_id, "persons", "children"])
            return (
                households[
                    ((households.persons == r) | (plus & (households.persons > r)))
                    & (households.children.fillna(0) == 0)
                ]
                .groupby(geo_id)
                .size()
            )

    make_hh_size(1)
    make_hh_size(2)
    make_hh_size(3)
    make_hh_size(3, True)
    make_hh_size(4, True)

    def make_hh_size_age(r, a, b, plus=False):
        hh_name = "hh_size_" + str(r)
        if plus:
            hh_name += "p"
        hh_name += "_age_" + str(a).zfill(2) + "_" + str(b).zfill(2)

        @orca.column(tab, hh_name, cache=True, cache_scope="iteration")
        def hh_size(households):
            households = households.to_frame([geo_id, "persons", "age_of_head"])
            return (
                households[
                    ((households.persons == r) | (plus & (households.persons > r)))
                    & ((households.age_of_head >= a) & (households.age_of_head <= b))
                ]
                .groupby(geo_id)
                .size()
            )

    make_hh_size_age(1, 15, 34)
    make_hh_size_age(1, 35, 44)
    make_hh_size_age(1, 65, np.inf)
    make_hh_size_age(2, 15, 34, True)
    make_hh_size_age(2, 35, 44, True)
    make_hh_size_age(2, 65, np.inf, True)

    @orca.column(tab, cache=True, cache_scope="iteration")
    def hh_no_car_or_lt_workers(households):
        households = households.to_frame([geo_id, "cars", "workers"])
        return (
            households[(households.cars == 0) | (households.cars < households.workers)]
            .groupby(geo_id)
            .size()
        )

    @orca.column(tab, cache=True, cache_scope="iteration")
    def pct_hh_no_car_or_lt_workers():
        df = orca.get_table(tab)
        df = df.to_frame(["hh_no_car_or_lt_workers", "hh"])
        return 1.0 * df.hh_no_car_or_lt_workers / df.hh

    def make_pop_race(r):
        hh_name = "hh_pop_race_" + str(r)

        @orca.column(tab, hh_name, cache=True, cache_scope="iteration")
        def hh_pop_race(persons):
            persons = persons.to_frame([geo_id, "race_id"])
            return persons[persons.race_id == r].groupby(geo_id).size()

        @orca.column(tab, "pct_" + hh_name, cache=True, cache_scope="iteration")
        def pct_hh_pop_race():
            df = orca.get_table(tab)
            df = df.to_frame([hh_name, "hh_pop"])
            return 1.0 * df[hh_name] / df.hh_pop

        gq_name = "gq_pop_race_" + str(r)

        @orca.column(tab, gq_name, cache=True, cache_scope="iteration")
        def gq_pop_race(group_quarters):
            group_quarters = group_quarters.to_frame([geo_id, "race_id"])
            return group_quarters[group_quarters.race_id == r].groupby(geo_id).size()

        @orca.column(tab, "pct_" + gq_name, cache=True, cache_scope="iteration")
        def pct_gq_pop_race():
            df = orca.get_table(tab)
            df = df.to_frame([gq_name, "gq_pop"])
            return 1.0 * df[gq_name] / df.gq_pop

        name = "pop_race_" + str(r)

        @orca.column(tab, name, cache=True, cache_scope="iteration")
        def pop_race():
            df = orca.get_table(tab)
            df = df.to_frame([gq_name, hh_name]).fillna(0)
            return df[gq_name] + df[hh_name]

        @orca.column(tab, "pct_" + name, cache=True, cache_scope="iteration")
        def pct_pop_race():
            df = orca.get_table(tab)
            df = df.to_frame([name, "pop"])
            return 1.0 * df[name] / df["pop"]

    for r in [1, 2, 3, 4]:
        make_pop_race(r)

    def make_pop_age(a, b):
        hh_name = "hh_pop_age_" + str(a).zfill(2) + "_" + str(b).zfill(2)

        @orca.column(tab, hh_name, cache=True, cache_scope="iteration")
        def hh_pop_age(persons):
            persons = persons.to_frame([geo_id, "age"])
            return (
                persons[(persons.age >= a) & (persons.age <= b)].groupby(geo_id).size()
            )

        gq_name = "gq_pop_age_" + str(a).zfill(2) + "_" + str(b).zfill(2)

        @orca.column(tab, gq_name, cache=True, cache_scope="iteration")
        def gq_pop_age(group_quarters):
            group_quarters = group_quarters.to_frame([geo_id, "age"])
            return (
                group_quarters[(group_quarters.age >= a) & (group_quarters.age <= b)]
                .groupby(geo_id)
                .size()
            )

        @orca.column(
            tab,
            "pop_age_" + str(a).zfill(2) + "_" + str(b).zfill(2),
            cache=True,
            cache_scope="iteration",
        )
        def pop_age():
            df = orca.get_table(tab)
            df = df.to_frame([hh_name, gq_name]).fillna(0)
            return df[hh_name] + df[gq_name]

    for (a, b) in [
        (00, 4),
        (5, 17),
        (18, 24),
        (18, 64),
        (25, 34),
        (35, 64),
        (65, np.inf),
        (00, 17),
        (25, 44),
        (25, 54),
        (25, 64),
        (45, 64),
        (55, 64),
        (65, 84),
        (85, np.inf),
        (35, 59),
        (60, 64),
        (65, 74),
        (75, np.inf),
    ]:
        make_pop_age(a, b)

    @orca.column(tab, cache=True, cache_scope="iteration")
    def hh_pop_age_median(persons):
        persons = persons.to_frame([geo_id, "age"])
        return persons.groupby(geo_id).median()

    @orca.column(tab, cache=True, cache_scope="iteration")
    def pct_hh_pop_age_05_17():
        df = orca.get_table(tab)
        df = df.to_frame(["hh_pop_age_05_17", "hh_pop"])
        return 1.0 * df.hh_pop_age_05_17 / df.hh_pop

    @orca.column(tab, cache=True, cache_scope="iteration")
    def pct_hh_pop_age_65_inf():
        df = orca.get_table(tab)
        df = df.to_frame(["hh_pop_age_65_inf", "hh_pop"])
        return 1.0 * df.hh_pop_age_65_inf / df.hh_pop

    @orca.column(tab, cache=True, cache_scope="iteration")
    def jobs_total(jobs):
        jobs = jobs.to_frame([geo_id])
        return jobs.groupby(geo_id).size()

    @orca.column(tab, cache=True, cache_scope="iteration")
    def jobs_home_based(jobs):
        jobs = jobs.to_frame([geo_id, "home_based_status"])
        return jobs[jobs.home_based_status == 1].groupby(geo_id).size()

    @orca.column(tab, cache=True, cache_scope="iteration")
    def jobs_home_based_in_nonres(jobs, buildings):
        jobs = jobs.to_frame(["building_id", "home_based_status"])
        jobs_count = jobs[jobs.home_based_status == 1].groupby("building_id").size()
        buildings = buildings.to_frame([geo_id, "residential_sqft"])
        buildings["jobs_count"] = jobs_count
        return (
            buildings[buildings.residential_sqft.fillna(0) <= 0]
            .groupby(geo_id)
            .jobs_count.sum()
            .fillna(0)
        )

    def make_job_sector_ind(i):
        @orca.column(
            tab, "jobs_sec_" + str(i).zfill(2), cache=True, cache_scope="iteration"
        )
        def jobs_sec_id(jobs):
            jobs = jobs.to_frame([geo_id, "sector_id"])
            return jobs[jobs.sector_id == i].groupby(geo_id).size()

        @orca.column(
            tab,
            "jobs_sec_" + str(i).zfill(2) + "_home_based",
            cache=True,
            cache_scope="iteration",
        )
        def jobs_sec_id_home(jobs):
            jobs = jobs.to_frame([geo_id, "sector_id", "home_based_status"])
            return (
                jobs[(jobs.sector_id == i) & (jobs.home_based_status == 1)]
                .groupby(geo_id)
                .size()
            )

    for i in orca.get_table("employment_sectors").index:
        make_job_sector_ind(i)
    # for i in range(1, 19):
    #     make_job_sector_ind(i)


@orca.column("parcels", cache=True, cache_scope="iteration")
def whatnot_id(parcels, whatnots, interesting_parcel_ids):
    parcels = parcels.to_frame(["large_area_id", "us_congress_id", "mi_senate_id", "mi_house_id", "city_id", "school_id", "zone_id"])
    parcels["parcel_id"] = parcels.index
    parcels.index.name = None
    parcels.loc[parcels.parcel_id.isin(interesting_parcel_ids), "parcel_id"] = 0

    whatnots = whatnots.to_frame(
        ["large_area_id", "us_congress_id", "mi_senate_id", "mi_house_id", "city_id", "school_id", "zone_id"]
    ).reset_index()
    m = pd.merge(
        parcels, whatnots, "left", ["large_area_id", "us_congress_id", "mi_senate_id", "mi_house_id", "city_id", "school_id", "zone_id"],
    )
    return m.whatnot_id


@orca.column("buildings", cache=True, cache_scope="iteration")
def whatnot_id(buildings, whatnots, interesting_parcel_ids):
    buildings = buildings.to_frame(
        ["large_area_id", "us_congress_id", "mi_senate_id", "mi_house_id", "city_id", "school_id", "zone_id"]
    ).reset_index()
    buildings.loc[buildings.parcel_id.isin(interesting_parcel_ids), "parcel_id"] = 0
    whatnots = whatnots.to_frame(
        ["large_area_id", "us_congress_id", "mi_senate_id", "mi_house_id", "city_id", "school_id", "zone_id"]
    ).reset_index()
    m = pd.merge(
        buildings,
        whatnots,
        "left",
        ["large_area_id", "us_congress_id", "mi_senate_id", "mi_house_id", "city_id", "school_id", "zone_id"],
    )
    return m.set_index("building_id").whatnot_id


@orca.column("jobs", cache=True, cache_scope="iteration")
def whatnot_id(jobs, buildings):
    return misc.reindex(buildings.whatnot_id, jobs.building_id)


@orca.column("households", cache=True, cache_scope="iteration")
def whatnot_id(households, buildings):
    return misc.reindex(buildings.whatnot_id, households.building_id)


@orca.column("persons", cache=True, cache_scope="iteration")
def whatnot_id(households, persons):
    return misc.reindex(households.whatnot_id, persons.household_id)


@orca.column("group_quarters", cache=True, cache_scope="iteration")
def whatnot_id(group_quarters, buildings):
    return misc.reindex(buildings.whatnot_id, group_quarters.building_id)


orca.add_injectable(
    "form_to_btype",
    {
        "residential": [81, 82, 83],
        "industrial": [31, 32, 33],
        "retail": [21, 65],
        "office": [23],
        "medical": [51, 52, 53],
        "entertainment": [61, 63, 91],
        # "mixedresidential": [21, 81, 82, 83],
        # "mixedoffice": [23, 81, 82, 83],
    },
)


# === indicators ===
def list_indicators():
    main = ["hh", "hh_pop", "gq_pop", "pop", "jobs_total", "housing_units"]
    hh = [
        ## size
        "household_size",
        "hh_size_1",
        "hh_size_2",
        "hh_size_3",
        "hh_size_3p",
        "hh_size_4p",
        ## income
        "incomes_1",
        "incomes_2",
        "incomes_3",
        "incomes_4",
        "pct_incomes_1",
        "pct_incomes_2",
        "pct_incomes_3",
        "pct_incomes_4",
        ## children
        "with_children",
        "without_children",
        "pct_with_children",
        ## senior
        "with_seniors",
        "without_seniors",
        "pct_with_seniors",
        ## chilren and size
        "with_children_hh_size_1",
        "with_children_hh_size_2",
        "with_children_hh_size_3",
        "with_children_hh_size_3p",
        "with_children_hh_size_4p",
        "without_children_hh_size_1",
        "without_children_hh_size_2",
        "without_children_hh_size_3",
        "without_children_hh_size_3p",
        "without_children_hh_size_4p",
        ## size and age
        "hh_size_1_age_15_34",
        "hh_size_1_age_35_44",
        "hh_size_1_age_65_inf",
        "hh_size_2p_age_15_34",
        "hh_size_2p_age_35_44",
        "hh_size_2p_age_65_inf",
        ## car worker
        "hh_no_car_or_lt_workers",
        "pct_hh_no_car_or_lt_workers",
    ]
    hh_pop = [
        ## race
        "hh_pop_race_1",
        "hh_pop_race_2",
        "hh_pop_race_3",
        "hh_pop_race_4",
        "pct_hh_pop_race_1",
        "pct_hh_pop_race_2",
        "pct_hh_pop_race_3",
        "pct_hh_pop_race_4",
        ## age
        "hh_pop_age_00_04",
        "hh_pop_age_05_17",
        "hh_pop_age_18_24",
        "hh_pop_age_25_34",
        "hh_pop_age_35_64",
        "hh_pop_age_65_inf",
        "hh_pop_age_18_64",
        "hh_pop_age_00_17",
        "hh_pop_age_25_44",
        "hh_pop_age_25_64",
        "hh_pop_age_45_64",
        "hh_pop_age_65_84",
        "hh_pop_age_85_inf",
        "hh_pop_age_35_59",
        "hh_pop_age_60_64",
        "hh_pop_age_65_74",
        "hh_pop_age_75_inf",
        "hh_pop_age_median",
        "pct_hh_pop_age_05_17",
        "pct_hh_pop_age_65_inf",
    ]
    pop = [
        ## race
        "pop_race_1",
        "pop_race_2",
        "pop_race_3",
        "pop_race_4",
        "pct_pop_race_1",
        "pct_pop_race_2",
        "pct_pop_race_3",
        "pct_pop_race_4",
        ## age
        "pop_age_00_04",
        "pop_age_05_17",
        "pop_age_18_24",
        "pop_age_25_34",
        "pop_age_35_64",
        "pop_age_65_inf",
        "pop_age_18_64",
        "pop_age_00_17",
        "pop_age_25_44",
        "pop_age_45_64",
        "pop_age_65_84",
        "pop_age_85_inf",
        "pop_age_35_59",
        "pop_age_60_64",
        "pop_age_65_74",
        "pop_age_75_inf",
        "pop_age_25_54",
        "pop_age_55_64",
        "pop_age_25_64",
    ]
    gq_pop = [
        ## race
        "gq_pop_race_1",
        "gq_pop_race_2",
        "gq_pop_race_3",
        "gq_pop_race_4",
        "pct_gq_pop_race_1",
        "pct_gq_pop_race_2",
        "pct_gq_pop_race_3",
        "pct_gq_pop_race_4",
        ## age
        "gq_pop_age_00_04",
        "gq_pop_age_05_17",
        "gq_pop_age_18_24",
        "gq_pop_age_25_34",
        "gq_pop_age_35_64",
        "gq_pop_age_65_inf",
        "gq_pop_age_18_64",
        "gq_pop_age_00_17",
        "gq_pop_age_25_44",
        "gq_pop_age_25_64",
        "gq_pop_age_45_64",
        "gq_pop_age_65_84",
        "gq_pop_age_85_inf",
        "gq_pop_age_35_59",
        "gq_pop_age_60_64",
        "gq_pop_age_65_74",
        "gq_pop_age_75_inf",
    ]
    ## job by sector
    sec_ids = sorted(set(orca.get_table("jobs").sector_id))
    job = [f"jobs_sec_{str(i).zfill(2)}" for i in sec_ids]
    job_home_based = ["jobs_home_based_in_nonres", "jobs_home_based"] + [
        s + "_home_based" for s in job
    ]
    ## land use
    parcel_building = [
        "parcel_is_allowed_residential",
        "parcel_is_allowed_demolition",
        "hu_filter",
        "buildings",
        "vacant_units",
        "job_spaces",
        "res_sqft",
        "nonres_sqft",
        "res_vacancy_rate",
        "nonres_vacancy_rate",
    ]
    ## building sqft
    btype_ids = sorted(set(orca.get_table("buildings").building_type_id))
    building_sqft = [f"building_sqft_type_{str(i).zfill(2)}" for i in btype_ids]

    return (
        main
        + hh
        + hh_pop
        + pop
        + gq_pop
        + job
        + job_home_based
        + parcel_building
        + building_sqft
    )
