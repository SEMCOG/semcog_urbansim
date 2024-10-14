import numpy as np
import orca
import pandas as pd
from urbansim.utils import misc

@orca.table(cache=True)
def buildings(store):
    df = store["buildings"]
    df = df.fillna(0)
    # Todo: combine two sqft prices into one and set non use sqft price to 0
    df["market_value"] = 0
    df["sqft_price_nonres"] = df.market_value * 1.0 / 0.7 / df.non_residential_sqft
    df.loc[df.sqft_price_nonres > 1000, "sqft_price_nonres"] = 0
    df.loc[df.sqft_price_nonres < 0, "sqft_price_nonres"] = 0
    df["sqft_price_res"] = (
        df.market_value
        * 1.0
        / 0.7
        / (df.sqft_per_unit.astype(int) * df.residential_units)
    )
    df.loc[df.sqft_price_res > 1000, "sqft_price_res"] = 0
    df.loc[df.sqft_price_res < 0, "sqft_price_res"] = 0
    df.fillna(0, inplace=True)

    df["mcd_model_quota"] = 0

    df = pd.merge(
        df,
        store["parcels"][["city_id"]],
        left_on="parcel_id",
        right_index=True,
        how="left",
    )
    df["city_id"] = df["city_id"].fillna(0)
    df["hu_filter"] = 0
    cites = [551, 1155, 1100, 3130, 6020, 6040]
    sample = df[df.residential_units > 0]
    sample = sample[~(sample.index.isin(store["households"].building_id))]
    # #35
    for c in sample.city_id.unique():
        frac = 0.8 if c in cites else 0
        # #35
        df.loc[
            sample[sample.city_id == c].sample(frac=frac, replace=False).index.values,
            "hu_filter",
        ] = 1

    # TODO, this is placeholder. will update with special emp buildings lookup later

    df[
        "sp_filter"
    ] = 0  # special filter: for event location/buildings, landmark buildings, etc
    # !!important set pseudo buildings to -2 sp_filter
    df.loc[df.index > 90000000, "sp_filter"] = -2

    df["event_id"] = 0  # also add event_id for event reference

    return df


@orca.table(cache=True)
def households(hh, buildings):
    df = hh
    b = buildings.to_frame(["large_area_id", "residential_units"])
    b = b[b.large_area_id.isin({161.0, 3.0, 5.0, 125.0, 99.0, 115.0, 147.0, 93.0})]
    df.loc[df.building_id == -1, "building_id"] = np.random.choice(
        b.index.values, (df.building_id == -1).sum()
    )

    bid_to_la = {
        1: 3, 2: 125, 3:99, 4: 161, 5: 115, 6: 147, 7: 93, 8: 5
    }
    idx_invalid_building_id = np.in1d(df.building_id, b.index.values) == False
    hh_to_assign = df.loc[idx_invalid_building_id, "building_id"]
    for bid, laid in bid_to_la.items():
        local_hh = hh_to_assign[hh_to_assign//1000000 == bid]
        # sample la hu
        df.loc[local_hh.index, 'building_id'] = np.random.choice(
            b[(b.large_area_id==laid)&(b.residential_units>0)].index.values, local_hh.size
        )

    df["large_area_id"] = misc.reindex(b.large_area_id, df.building_id,)

    # dtype optimization
    df["workers"] = df["workers"].fillna(0).astype(np.int8)
    df["children"] = df["children"].fillna(0).astype(np.int8)
    df["persons"] = df["persons"].astype(np.int8)
    df["cars"] = df["cars"].astype(np.int8)
    df["race_id"] = df["race_id"].astype(np.int8)
    df["income"] = df["income"].astype(np.int32)
    df["age_of_head"] = df["age_of_head"].astype(np.int8)
    df["large_area_id"] = df["large_area_id"].astype(np.uint8)
    return df.fillna(0)


@orca.table(cache=True)
def persons(p):
    df = p
    df["relate"] = df["relate"].astype(np.int8)
    df["age"] = df["age"].astype(np.int8)
    df["worker"] = df["worker"].astype(np.int8)
    df["sex"] = df["sex"].astype(np.int8)
    df["race_id"] = df["race_id"].astype(np.int8)
    df["member_id"] = df["member_id"].astype(np.int8)
    df["household_id"] = df["household_id"].astype(np.int64)
    return df


@orca.table(cache=True)
def jobs(store, buildings):
    df = store["jobs"]
    b = buildings.to_frame(["large_area_id"])
    b = b[b.large_area_id.isin({161.0, 3.0, 5.0, 125.0, 99.0, 115.0, 147.0, 93.0})]
    df.loc[df.building_id == -1, "building_id"] = np.random.choice(
        b.index.values, (df.building_id == -1).sum()
    )
    idx_invalid_building_id = np.in1d(df.building_id, b.index.values) == False
    df.loc[idx_invalid_building_id, "building_id"] = np.random.choice(
        b.index.values, idx_invalid_building_id.sum()
    )
    df["large_area_id"] = misc.reindex(b.large_area_id, df.building_id)
    return df.fillna(0)


@orca.table(cache=True)
def parcels(store, zoning):
    parcels_df = store["parcels"]
    # concat pseudo buildings parcels
    #  based on zoning.is_developable, adjust parcels pct_undev
    pct_undev = zoning.pct_undev.copy()
    # Parcel is NOT developable, leave as is unless events are present (173,616 parcels)
    pct_undev[zoning.is_developable == 0] = 100
    # Parcel is developable, but refer to the field “percent_undev” for how much of the parcel is actually developable (1,791,169 parcels)
    # Parcel is developable, but contains underground storage tanks
    pct_undev[zoning.is_developable == 2] += 10
    parcels_df["pct_undev"] = pct_undev.clip(0, 100).astype("int16")
    parcels_df["pct_undev"] = parcels_df["pct_undev"].fillna(0)
    return parcels_df

@orca.table(cache=True)
def base_job_space(buildings):
    return buildings.jobs_non_home_based.to_frame("base_job_space")

@orca.table(cache=True)
def building_to_zone_baseyear():
    # baseyear building_id to zone_id mapping
    # fix the issue where one parcel could have multiple TAZ zone
    return pd.read_csv('data/building_to_zone_baseyear_2020_shrink.csv').set_index('building_id')


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
def whatnot_id(parcels, whatnots,):
    parcels = parcels.to_frame(["large_area_id", "city_id", "zone_id"])
    parcels.index.name = None

    whatnots = whatnots.to_frame(
        ["large_area_id", "city_id", "zone_id"]
    ).reset_index()
    m = pd.merge(
        parcels, whatnots, "left", ["large_area_id", "city_id", "zone_id"],
    )
    return m.whatnot_id


@orca.column("buildings", cache=True, cache_scope="iteration")
def whatnot_id(buildings, whatnots,):
    buildings = buildings.to_frame(
        ["large_area_id", "city_id", "zone_id"]
    ).reset_index()
    whatnots = whatnots.to_frame(
        ["large_area_id", "city_id", "zone_id"]
    ).reset_index()
    m = pd.merge(
        buildings,
        whatnots,
        "left",
        ["large_area_id", "city_id", "zone_id"],
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
    main = ["hh", "hh_pop", "housing_units"]
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
    ## job by sector
    sec_ids = sorted(set(orca.get_table("jobs").sector_id))
    job = [f"jobs_sec_{str(i).zfill(2)}" for i in sec_ids]
    job_home_based = ["jobs_home_based_in_nonres", "jobs_home_based"] + [
        s + "_home_based" for s in job
    ]
    ## land use
    parcel_building = [
        # "parcel_is_allowed_residential",
        # "parcel_is_allowed_demolition",
        # "hu_filter",
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
        + parcel_building
        + building_sqft
    )
