import numpy as np
import orca
import pandas as pd
from urbansim.utils import misc
import variables


#####################
# BUILDINGS VARIABLES
#####################

# @orca.column('buildings', cache=True)
# def school_district_id(buildings, parcels):
#     return misc.reindex(parcels.school_district_id, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def hedonic_id(buildings):
    hedonic_id = buildings.large_area_id * 100 + buildings.building_type_id
    hedonic_id.loc[
        # buildings.building_type_id.isin([24, 32, 42, 43, 52, 53, 61, 62])
        buildings.building_type_id.isin([11, 13, 14, 32, 41, 42, 52, 53, 61, 63, 71, 84, 91, 92, 93, 94, 95])
    ] = buildings.building_type_id
    # hedonic_id.loc[hedonic_id == 571] = 371
    # hedonic_id.loc[hedonic_id == 584] = 384
    return hedonic_id


@orca.column("buildings", cache=True, cache_scope="iteration")
def general_type(buildings, building_type_map):
    return buildings.building_type_id.map(building_type_map)


@orca.column("buildings", cache=True)
def is_medical(buildings):
    return (buildings.general_type == "Medical").astype("int")


@orca.column("buildings", cache=True)
def is_tcu(buildings):
    return (buildings.general_type == "TCU").astype("int")


@orca.column("buildings", cache=True)
def is_institutional(buildings):
    return (buildings.general_type == "Institutional").astype("int")


@orca.column("buildings", cache=True)
def is_retail(buildings):
    return (buildings.general_type == "Retail").astype("int")


@orca.column("buildings", cache=True)
def is_office(buildings):
    return (buildings.general_type == "Office").astype("int")


@orca.column("buildings", cache=True)
def is_industrial(buildings):
    return (buildings.general_type == "Industrial").astype("int")


@orca.column("buildings", cache=True, cache_scope="iteration")
def gq_building(buildings, group_quarters):
    return 1 * buildings.index.isin(group_quarters.building_id)


# @orca.column('buildings', cache=True, cache_scope='iteration')
# def _node_id(buildings, parcels):
#     return misc.reindex(parcels._node_id, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def nodeid_walk(buildings, parcels):
    return misc.reindex(parcels.nodeid_walk, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def nodeid_drv(buildings, parcels):
    return misc.reindex(parcels.nodeid_drv, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def x(buildings, parcels):
    return misc.reindex(parcels.x.fillna(-1), buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def y(buildings, parcels):
    return misc.reindex(parcels.y.fillna(-1), buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def zone_id(buildings, parcels):
    return misc.reindex(parcels.zone_id, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def large_hh_city(buildings, parcels):
    large_hh_city = buildings.parcel_id * 0
    p = parcels.to_frame(parcels.local_columns)
    b = buildings.to_frame(buildings.local_columns)
    lh_index = b.loc[
        (
            b.parcel_id.isin(
                p.loc[
                    p.city_id.isin([547, 552, 553, 1025, 1030, 1090, 6135, 5090])
                ].index
            )
        )
    ].index
    large_hh_city.loc[lh_index] = 1
    return large_hh_city


@orca.column("buildings", cache=True, cache_scope="iteration")
def small_hh_city(buildings, parcels):
    small_hh_city = buildings.parcel_id * 0
    p = parcels.to_frame(parcels.local_columns)
    b = buildings.to_frame(buildings.local_columns)
    lh_index = b.loc[
        (
            b.parcel_id.isin(
                p.loc[p.city_id.isin([508, 515, 527, 529, 532, 536, 538, 551])].index
            )
        )
    ].index
    small_hh_city.loc[lh_index] = 1
    return small_hh_city


@orca.column("buildings", cache=True, cache_scope="iteration")
def city_id(buildings, parcels):
    return misc.reindex(parcels.city_id, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def semmcd(buildings, parcels):
    return misc.reindex(parcels.semmcd, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def large_area_id(buildings, parcels):
    return misc.reindex(parcels.large_area_id, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def county_id(buildings, parcels):
    return misc.reindex(parcels.county_id, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def geoid(buildings, parcels):
    # geoid = parcels[['county_id', 'census_bg_id']].apply(lambda x: 26*10000000000 + x.county_id*10000000 + x.census_bg_id, axis=1)
    geoid = 26 * 10000000000 + parcels.county_id * 10000000 + parcels.census_bg_id
    return misc.reindex(geoid.fillna(0).astype(int), buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def popden(buildings, zones):
    return misc.reindex(zones.popden, buildings.zone_id).fillna(0)


@orca.column("buildings", cache=True, cache_scope="iteration")
def residential_sqft(buildings):
    return buildings.sqft_per_unit * buildings.residential_units


@orca.column("buildings", cache=True, cache_scope="iteration")
def building_sqft(buildings):
    return buildings.non_residential_sqft.astype(int) + buildings.residential_sqft


@orca.column("buildings", cache=True, cache_scope="iteration")
def building_sqft_per_job(buildings, building_sqft_per_job):
    b = buildings.to_frame(["building_type_id"])
    bsqft_job = building_sqft_per_job.to_frame()

    return pd.merge(
        b, bsqft_job, left_on=["building_type_id"], right_index=True, how="left"
    ).building_sqft_per_job.fillna(0)


@orca.column("buildings", cache=True, cache_scope="iteration")
def job_spaces(buildings, base_job_space):
    job_spaces = buildings.non_residential_sqft / buildings.building_sqft_per_job
    job_spaces[np.isinf(job_spaces)] = np.nan
    job_spaces[job_spaces < 0] = 0
    job_spaces = job_spaces.fillna(0).round().astype("int")

    jobs = buildings.jobs_non_home_based
    jobs = jobs.reindex(job_spaces.index).fillna(0).round().astype("int")
    jobs = jobs[jobs > job_spaces]
    job_spaces.loc[jobs.index] = jobs

    base_job_space = (
        base_job_space.base_job_space.reindex(job_spaces.index)
        .fillna(0)
        .round()
        .astype("int")
    )
    base_job_space = base_job_space[base_job_space > job_spaces]
    job_spaces.loc[base_job_space.index] = base_job_space

    orca.add_table("base_job_space", job_spaces.to_frame("base_job_space"))

    return job_spaces


@orca.column("buildings", cache=True, cache_scope="iteration")
def non_residential_units(buildings):
    return buildings.job_spaces


@orca.column("buildings", cache=True, cache_scope="iteration")
def jobs_within_30_min(buildings, zones):
    return misc.reindex(zones.jobs_within_30_min, buildings.zone_id).fillna(0)


@orca.column("buildings")
def vacant_residential_units(buildings, households):
    return buildings.residential_units.sub(
        households.building_id.value_counts(), fill_value=0
    )


@orca.column("buildings")
def vacant_job_spaces(buildings, jobs):
    return buildings.job_spaces.sub(jobs.building_id.value_counts(), fill_value=0)


@orca.column("buildings", cache=True, cache_scope="iteration")
def parcel_sqft(buildings, parcels):
    return misc.reindex(parcels.parcel_sqft, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def school_district_achievement(buildings, parcels):
    return misc.reindex(parcels.school_district_achievement, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def crime_ucr_rate(buildings, parcels):
    return misc.reindex(parcels.crime_ucr_rate, buildings.parcel_id).fillna(0)


@orca.column("buildings", cache=True, cache_scope="iteration")
def crime_other_rate(buildings, parcels):
    return misc.reindex(parcels.crime_other_rate, buildings.parcel_id).fillna(0)


### accessibilities auto
@orca.column("buildings", cache=True, cache_scope="iteration")
def drv_nearest_hospital(buildings, parcels):
    return misc.reindex(parcels.drv_nearest_hospital, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def drv_nearest_healthcenter(buildings, parcels):
    return misc.reindex(parcels.drv_nearest_healthcenter, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def drv_nearest_grocery(buildings, parcels):
    return misc.reindex(parcels.drv_nearest_grocery, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def drv_nearest_urgentcare(buildings, parcels):
    return misc.reindex(parcels.drv_nearest_urgentcare, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def drv_nearest_library(buildings, parcels):
    return misc.reindex(parcels.drv_nearest_library, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def drv_nearest_park(buildings, parcels):
    return misc.reindex(parcels.drv_nearest_park, buildings.parcel_id)


### accessibilities walk
@orca.column("buildings", cache=True, cache_scope="iteration")
def walk_nearest_hospital(buildings, parcels):
    return misc.reindex(parcels.walk_nearest_hospital, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def walk_nearest_grocery(buildings, parcels):
    return misc.reindex(parcels.walk_nearest_grocery, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def walk_nearest_healthcenter(buildings, parcels):
    return misc.reindex(parcels.walk_nearest_healthcenter, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def walk_nearest_urgentcare(buildings, parcels):
    return misc.reindex(parcels.walk_nearest_urgentcare, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def walk_nearest_library(buildings, parcels):
    return misc.reindex(parcels.walk_nearest_library, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def walk_nearest_park(buildings, parcels):
    return misc.reindex(parcels.walk_nearest_park, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def building_age(buildings, year):
    year_built = buildings.year_built
    year_built[year_built < 1600] = year_built[year_built > 1600].mean()
    age = year - year_built
    return age


@orca.column("buildings", cache=True, cache_scope="iteration")
def building_age_gt_50(buildings):
    return (buildings.building_age > 50).astype("int32")


@orca.column("buildings", cache=True, cache_scope="iteration")
def building_age_gt_70(buildings):
    return (buildings.building_age > 70).astype("int32")


@orca.column("buildings", cache=True, cache_scope="iteration")
def building_age_gt_80(buildings):
    return (buildings.building_age > 80).astype("int32")


@orca.column("buildings", cache=True, cache_scope="iteration")
def building_age_gt_90(buildings):
    return (buildings.building_age > 90).astype("int32")


@orca.column("buildings", cache=True, cache_scope="iteration")
def building_age_gt_100(buildings):
    return (buildings.building_age > 100).astype("int32")


@orca.column("buildings", cache=True, cache_scope="iteration")
def building_age_le_10(buildings):
    return (buildings.building_age < 10).astype("int32")


@orca.column("buildings", cache=True, cache_scope="iteration")
def b_is_pre_1945(buildings):
    return (buildings.year_built < 1945).astype("int32")


@orca.column("buildings", cache=True, cache_scope="iteration")
def b_is_newerthan2015(buildings):
    return (buildings.year_built > 2015).astype("int32")


@orca.column("buildings", cache=True, cache_scope="iteration")
def b_total_jobs(jobs, buildings):
    jobs_by_b = jobs.building_id.groupby(jobs.building_id).size()
    return pd.Series(index=buildings.index, data=jobs_by_b).fillna(0)


@orca.column("buildings", cache=True, cache_scope="iteration")
def b_total_households(households, buildings):
    hh_by_b = households.building_id.groupby(households.building_id).size()
    return pd.Series(index=buildings.index, data=hh_by_b).fillna(0)


@orca.column("buildings", cache=True, cache_scope="iteration")
def jobs_home_based(jobs):
    jobs = jobs.to_frame(["building_id", "home_based_status"])
    return jobs[jobs.home_based_status == 1].groupby("building_id").size()


@orca.column("buildings", cache=True, cache_scope="iteration")
def jobs_non_home_based(jobs, buildings):
    jobs = jobs.to_frame(["building_id", "home_based_status"])
    return pd.Series(
        index=buildings.index, 
        data=jobs[jobs.home_based_status == 0].groupby("building_id").size()
    ).fillna(0)


@orca.column("buildings", cache=True, cache_scope="iteration")
def mean_zonal_hhsize(buildings, households):
    buildings = buildings.to_frame(["zone_id"])
    households = households.to_frame(["zone_id", "persons"])
    buildings["mean_zonal_hhsize"] = misc.reindex(
        households.groupby("zone_id").persons.mean(), buildings.zone_id
    )
    buildings.mean_zonal_hhsize = buildings.mean_zonal_hhsize.fillna(
        buildings.mean_zonal_hhsize.mean()
    )
    return buildings.mean_zonal_hhsize


@orca.column("buildings", cache=True, cache_scope="iteration")
def mode_income_quartile(buildings, households):
    buildings = buildings.to_frame(["zone_id"])
    households = households.to_frame(["zone_id", "income_quartile"])
    mode_income_quartile = households.income_quartile.groupby(households.zone_id).agg(
        lambda x: x.value_counts().index[0]
    )
    buildings["mode_income_quartile"] = misc.reindex(
        mode_income_quartile, buildings.zone_id
    )
    buildings.mode_income_quartile = buildings.mode_income_quartile.fillna(0)
    return buildings.mode_income_quartile


@orca.column("buildings", cache=True, cache_scope="iteration")
def mode_income_quartile_is_1(buildings):
    return (buildings.mode_income_quartile == 1).astype("int")


@orca.column("buildings", cache=True, cache_scope="iteration")
def mode_income_quartile_is_2(buildings):
    return (buildings.mode_income_quartile == 2).astype("int")


@orca.column("buildings", cache=True, cache_scope="iteration")
def mode_income_quartile_is_3(buildings):
    return (buildings.mode_income_quartile == 3).astype("int")


@orca.column("buildings", cache=True, cache_scope="iteration")
def mode_income_quartile_is_4(buildings):
    return (buildings.mode_income_quartile == 4).astype("int")


### Variable generation functions


def make_dummy_variable(geog_var, geog_id):
    """
    Generate dummy variable. Registers with orca.
    """
    var_name = "%s_is_%s" % (geog_var, geog_id)

    @orca.column("buildings", var_name, cache=True, cache_scope="iteration")
    def func():
        buildings = orca.get_table("buildings")
        return (buildings[geog_var] == geog_id).astype("int32")

    return func


def make_logged_variable(var_to_log):
    """
    Generate logged variable. Registers with orca.
    """
    var_name = "b_ln_%s" % var_to_log
    # print var_name

    @orca.column("buildings", var_name, cache=True, cache_scope="iteration")
    def func():
        buildings = orca.get_table("buildings")
        return np.log1p(buildings[var_to_log]).fillna(0)

    return func


def make_employment_proportion_variable(sector_id):
    """
    Generate employment proportion of total jobs in building variable. Registers with orca.
    """
    var_name = "bldg_empratio_%s" % sector_id

    @orca.column("buildings", var_name, cache=True, cache_scope="iteration")
    def func():
        buildings = orca.get_table("buildings")
        jobs = orca.get_table("jobs")
        total_jobs = buildings.b_total_jobs
        jobs = jobs.to_frame(jobs.local_columns)
        jobs_sector = jobs[jobs.sector_id == sector_id].building_id.value_counts()
        return (jobs_sector / total_jobs).fillna(0)


def make_disagg_var(
    from_geog_name,
    to_geog_name,
    var_to_disaggregate,
    from_geog_id_name,
    name_based_on_geography=True,
):
    """
    Generator function for disaggregating variables. Registers with orca.
    """
    if name_based_on_geography:
        var_name = from_geog_name + "_" + var_to_disaggregate
    else:
        var_name = var_to_disaggregate

    @orca.column(to_geog_name, var_name, cache=True, cache_scope="iteration")
    def func():
        print(
            "Disaggregating {} to {} from {}".format(
                var_to_disaggregate, to_geog_name, from_geog_name
            )
        )

        from_geog = orca.get_table(from_geog_name)
        to_geog = orca.get_table(to_geog_name)
        return misc.reindex(
            from_geog[var_to_disaggregate], to_geog[from_geog_id_name]
        ).fillna(0)

    return func


geographic_levels = [("parcels", "parcel_id"), ("zones", "zone_id")]
vars_to_dummify = ["city_id", "building_type_id"]
vars_to_log = [
    "non_residential_sqft",
    "building_sqft",
    "land_area",
    "parcel_sqft",
    "sqft_per_unit",
    "parcels_parcel_far",
    "sqft_price_nonres",
    "sqft_price_res",
    "market_value",
    "mcd_model_quota",
]

for geography in geographic_levels:
    geography_name = geography[0]
    geography_id = geography[1]
    if geography_name != "buildings":
        building_vars = orca.get_table("buildings").columns
        for var in orca.get_table(geography_name).columns:
            if var not in building_vars:
                make_disagg_var(geography_name, "buildings", var, geography_id)

for dummifiable_var in vars_to_dummify:
    var_cat_ids = np.unique(orca.get_table("buildings")[dummifiable_var]).astype("int")
    for var_cat_id in var_cat_ids:
        if var_cat_id > 0:
            make_dummy_variable(dummifiable_var, var_cat_id)

for var_to_log in vars_to_log:
    make_logged_variable(var_to_log)

emp_sectors = np.arange(18) + 1
for sector in emp_sectors:
    make_employment_proportion_variable(sector)


@orca.column("buildings", cache=True, cache_scope="iteration")
def ln_empden(buildings, zones):
    return np.log1p(misc.reindex(zones.empden, buildings.zone_id).fillna(0))


@orca.column("buildings", cache=True, cache_scope="iteration")
def zone_mean_age_of_head(buildings, zones):
    return misc.reindex(zones.mean_age_of_head, buildings.zone_id).fillna(0)


@orca.column("buildings", cache=True, cache_scope="iteration")
def zone_prop_race_1(buildings, zones):
    return misc.reindex(zones.prop_race_1, buildings.zone_id).fillna(0)


@orca.column("buildings", cache=True, cache_scope="iteration")
def zone_prop_race_2(buildings, zones):
    return misc.reindex(zones.prop_race_2, buildings.zone_id).fillna(0)


@orca.column("buildings", cache=True, cache_scope="iteration")
def zone_prop_race_3(buildings, zones):
    return misc.reindex(zones.prop_race_3, buildings.zone_id).fillna(0)


@orca.column("buildings", cache=True, cache_scope="iteration")
def zone_prop_race_4(buildings, zones):
    return misc.reindex(zones.prop_race_4, buildings.zone_id).fillna(0)


@orca.column("buildings", cache=True, cache_scope="iteration")
def ln_residential_units(buildings):
    return np.log1p(buildings.residential_units)


@orca.column("buildings", cache=True, cache_scope="iteration")
def census_bg_id(buildings, parcels):
    return misc.reindex(parcels.census_bg_id, buildings.parcel_id).fillna(0)


@orca.column("buildings", cache=True, cache_scope="iteration")
def zone_id(buildings, parcels):
    return misc.reindex(parcels.zone_id, buildings.parcel_id).fillna(0)


@orca.column("buildings", cache=True, cache_scope="iteration")
def city_id(buildings, parcels):
    return misc.reindex(parcels.city_id, buildings.parcel_id).fillna(0)


@orca.column("buildings", cache=True)
def hu_filter(buildings, households, parcels):
    """ move hu_filter code from dataset.py to here """
    buildings = buildings.local
    series = pd.Series([0 for _ in range(len(buildings))], index=buildings.index)
    city_id = misc.reindex(parcels.city_id, buildings.parcel_id).fillna(0)
    cites = [551, 1155, 1100, 3130, 6020, 6040]
    sample = buildings[buildings.residential_units > 0]
    sample = sample[~(sample.index.isin(households.building_id))]
    for c in city_id.unique():
        frac = 0.9 if c in cites else 0.5
        sampled_indexes = (
            sample[sample.index.isin(city_id[city_id == c].index)]
            .sample(frac=frac, replace=False)
            .index
        )
        series[series.index.isin(sampled_indexes)] = 1
    # print('hu_filter', series)
    return series


def standardize(series):
    if pd.api.types.is_numeric_dtype(series):
        return (series - series.mean()) / series.std()
    else:
        return series


def register_standardized_variable(table_name, column_to_s):
    """
    Register standardized variable with orca.
    Parameters
    ----------
    table_name : str
        Name of the orca table that this column is part of.
    column_to_ln : str
        Name of the orca column to standardize.
    Returns
    -------
    column_func : function
    """
    new_col_name = "st_" + column_to_s

    @orca.column(table_name, new_col_name, cache=True, cache_scope="iteration")
    def column_func():
        return standardize(orca.get_table(table_name)[column_to_s])

    return column_func


for var in orca.get_table("buildings").columns:
    if var == "general_type":
        # skip general_type, which stalls the process when loading st column
        continue
    register_standardized_variable("buildings", var)

