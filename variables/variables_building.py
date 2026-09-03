import numpy as np
import orca
import pandas as pd
from urbansim.utils import misc
import variables
import lcm_utils


#####################
# BUILDINGS VARIABLES
#####################

# @orca.column('buildings', cache=True)
# def school_district_id(buildings, parcels):
#     return misc.reindex(parcels.school_district_id, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def hedonic_id(buildings):
    """
    Calculate hedonic_id with building type aggregation to solve sampling issues.

    Aggregation Strategy (Strategy 1):
    - Institutional types (11, 13, 14, 92, 93) → 11
    - TCU subtypes (41, 42, 95) → 41
    - Medical subtypes (51, 52, 53) → 51
    - Entertainment + Hospitality (61, 63, 65, 91) → 61

    This increases sample sizes for rare building types while maintaining
    area segmentation where data supports it.
    """
    # Create a copy of building_type_id for mapping
    btype_mapped = buildings.building_type_id.copy()

    # Aggregate Institutional types (11, 13, 14, 92, 93) → 11
    # Includes: various institutional buildings
    btype_mapped[btype_mapped.isin([13, 14, 92, 93])] = 11

    # Aggregate TCU subtypes (41, 42, 95) → 41
    # TCU = Transportation, Communication, Utilities (parking, delivery services, etc.)
    btype_mapped[btype_mapped.isin([42, 95])] = 41

    # Aggregate Medical subtypes (51, 52, 53) → 51
    # Includes: hospitals, clinics, medical offices
    btype_mapped[btype_mapped.isin([52, 53])] = 51

    # Aggregate Entertainment + Hospitality (61, 63, 65, 91) → 61
    # Includes: entertainment venues, hotels, restaurants
    btype_mapped[btype_mapped.isin([63, 65, 91])] = 61

    # Calculate hedonic_id with mapped building types
    hedonic_id = buildings.large_area_id * 100 + btype_mapped

    # Use all-building model for types that don't have enough samples for area segmentation
    # These building types use building_type_id as hedonic_id directly (area 0)
    all_building_btypes = [
        11,  # Institutional (aggregated from 11, 13, 14, 92, 93)
        32,  # Industrial (rare variant, uses all buildings)
        41,  # TCU (aggregated from 41, 42, 95, uses all buildings)
        51,  # Medical (aggregated from 51, 52, 53 - may have area models where samples allow)
        61,  # Entertainment+Hospitality (aggregated from 61, 63, 65, 91)
        71,  # Others
        84,  # Residential (shouldn't be in non-res but keeping for safety)
        94,  # Other commercial
        96,  # Data Center (regional constant-price fallback)
    ]

    # For types marked as all-building, use building_type_id directly (area 0)
    hedonic_id.loc[btype_mapped.isin(all_building_btypes)] = btype_mapped

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
    return (buildings.sqft_per_unit.astype("int64")
            * buildings.residential_units.astype("int64"))


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
    # clip by 0 to prevent negative vacant job spaces
    return buildings.job_spaces.sub(jobs.building_id.value_counts(), fill_value=0).clip(lower=0)


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
def bike_nearest_grocery(buildings, parcels):
    return misc.reindex(parcels.bike_nearest_grocery, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def bike_nearest_library(buildings, parcels):
    return misc.reindex(parcels.bike_nearest_library, buildings.parcel_id)


@orca.column("buildings", cache=True, cache_scope="iteration")
def bike_nearest_park(buildings, parcels):
    return misc.reindex(parcels.bike_nearest_park, buildings.parcel_id)

@orca.column("buildings", cache=True, cache_scope="iteration")
def building_age(buildings, year):
    # Retrieve year_built and city_id series
    year_built = buildings.year_built
    city_id = buildings.city_id
    # Define a mask for invalid year_built entries
    invalid_mask = (year_built < 1600) | (year_built > 2100) | year_built.isna()
    # Calculate the median year_built for each city_id group
    median_year_built_by_city_id = year_built.groupby(city_id).transform('median').astype(int)
    # Replace invalid year_built entries with the median of their city_id group
    year_built = year_built.where(~invalid_mask, median_year_built_by_city_id)
    # Calculate building age
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
def building_age_le_20(buildings):
    return (buildings.building_age < 20).astype("int32")


@orca.column("buildings", cache=True, cache_scope="iteration")
def b_is_pre_1945(buildings):
    return (buildings.year_built < 1945).astype("int32")


@orca.column("buildings", cache=True, cache_scope="iteration")
def building_age_le_3(buildings):
    return (buildings.building_age < 3).astype("int32")


@orca.column("buildings", cache=True, cache_scope="iteration")
def b_is_newerthan2015(buildings):
    return (buildings.year_built > 2015).astype("int32")


@orca.column("buildings", cache=True, cache_scope="iteration")
def b_is_new(buildings):
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
        data=jobs[jobs.home_based_status == 0].groupby("building_id").size(),
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

def make_employment_taz_proportion_variable(sector_id):
    """
    Generate employment proportion of total jobs by sector in TAZ 
    reindex to buildings level. Registers with orca.
    issue #65
    """
    var_name = "taz_empratio_%s" % sector_id

    @orca.column("buildings", var_name, cache=True, cache_scope="iteration")
    def func():
        buildings = orca.get_table("buildings")
        jobs = orca.get_table("jobs")
        # total_jobs = buildings.b_total_jobs
        jobs = jobs.to_frame(['sector_id', 'zone_id'])
        # calculate total jobs by TAZ
        total_jobs = jobs.groupby('zone_id').size()
        # filter jobs by sector
        jobs_sector = jobs[jobs.sector_id == sector_id].zone_id.value_counts()
        # calculate proportion of jobs by TAZ
        taz_empratio = (jobs_sector / total_jobs).fillna(0)
        # make sure the value is between 0 and 1
        taz_empratio = taz_empratio.clip(lower=0, upper=1)
        # reindex to buildings
        return misc.reindex(taz_empratio, buildings.zone_id).fillna(0)

def make_household_tract_proportion_variable(hh_types):
    """
    Generate household type *interaction* proportion of total households in tract,
    reindexed to the buildings level. Registers with orca.

    * hh_types -- single string or tuple of strings (e.g., 'with_children' or ('with_children', 'with_senior'))
    """
    if isinstance(hh_types, str):
        hh_types = (hh_types,)
    var_name = "tract_hh_type_ratio_" + "_".join(hh_types)
    @orca.column("buildings", var_name, cache=True, cache_scope="iteration")
    def func():
        buildings = orca.get_table("buildings")
        hh = orca.get_table("households")
        hh_df = hh.to_frame(list(hh_types) + ['tract_id'])
        # Total number of households per TAZ
        total_hh = hh_df.groupby('tract_id').size()
        # Households satisfying all hh_type == 1
        mask = np.logical_and.reduce([hh_df[hh_type] == 1 for hh_type in hh_types])
        hh_type_count = hh_df.loc[mask, 'tract_id'].value_counts()
        # Proportion calculation
        tract_hhtype_ratio = (hh_type_count / total_hh).fillna(0).clip(0, 1)
        # Reindex to buildings
        return misc.reindex(tract_hhtype_ratio, buildings.tract_id).fillna(0)

def make_building_employment_variable(sector_id):
    """
    Generate jobs by sectors in building variable. Registers with orca.
    """
    var_name = "bldg_jobs_sector_%s" % sector_id

    @orca.column("buildings", var_name, cache=True, cache_scope="iteration")
    def func():
        jobs = orca.get_table("jobs")
        jobs = jobs.to_frame(jobs.local_columns)
        jobs_sector = jobs[jobs.sector_id == sector_id].building_id.value_counts()
        return jobs_sector.fillna(0)

def make_employment_node_ratio_variable(sector_id):
    """
    Generate jobs by sectors in building variable. Registers with orca.
    """
    var_name = "nodes_walk_job_ratio_sector_%s" % sector_id
    node_walk_varname = "nodes_walk_sector%s_jobs" % sector_id

    @orca.column("buildings", var_name, cache=True, cache_scope="iteration")
    def func():
        buildings = orca.get_table("buildings").to_frame(['nodes_walk_jobs', node_walk_varname])
        node_total_jobs = buildings["nodes_walk_jobs"]
        node_sector_jobs = buildings[node_walk_varname]
        return (node_sector_jobs / node_total_jobs).fillna(0)

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
    make_building_employment_variable(sector)
    make_employment_node_ratio_variable(sector)
    make_employment_taz_proportion_variable(sector)

# taz_segments will be like
# [("children_has_children", "ownership_own", "aoh_lt35"), ...]
taz_segments = lcm_utils.get_hlcm_segment()
for seg in taz_segments:
    make_household_tract_proportion_variable(seg)

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
def tract_id(buildings, parcels):
    return misc.reindex(parcels.tract_id, buildings.parcel_id).fillna(0)


@orca.column("buildings", cache=True, cache_scope="iteration")
def maz_id(buildings):
    # Initialized from parcel MAZ plus base-year overrides in dataset.buildings;
    # forecast-created buildings store their own weighted MAZ draw.
    return buildings.local["maz_id"]


@orca.column("buildings", cache=True, cache_scope="iteration")
def zone_id(buildings, micro_zones):
    # TAZ is DERIVED from MAZ through the crosswalk, never assigned independently -- that
    # guarantees MAZ nests inside TAZ. Replaces the retired building_to_zone_baseyear CSV
    # (parcel-default + direct-TAZ override), at finer MAZ resolution.
    return buildings.maz_id.map(micro_zones.zone_id).fillna(0)


@orca.column("buildings", cache=True, cache_scope="iteration")
def school_id(buildings, parcels):
    return misc.reindex(parcels.school_id, buildings.parcel_id).fillna(0)


@orca.column("buildings", cache=True, cache_scope="iteration")
def mi_house_id(buildings, parcels):
    return misc.reindex(parcels.mi_house_id, buildings.parcel_id).fillna(0)


@orca.column("buildings", cache=True, cache_scope="iteration")
def mi_senate_id(buildings, parcels):
    return misc.reindex(parcels.mi_senate_id, buildings.parcel_id).fillna(0)

    
@orca.column("buildings", cache=True, cache_scope="iteration")
def us_congress_id(buildings, parcels):
    return misc.reindex(parcels.us_congress_id, buildings.parcel_id).fillna(0)
    

@orca.column("buildings", cache=True, cache_scope="iteration")
def city_id(buildings, parcels):
    return misc.reindex(parcels.city_id, buildings.parcel_id).fillna(0)


# @orca.column("buildings", cache=True, cache_scope="forever")
# def hu_filter(buildings, households, parcels):
#     """ move hu_filter code from dataset.py to here """
#     buildings = buildings.local
#     series = pd.Series([0 for _ in range(len(buildings))], index=buildings.index)
#     city_id = misc.reindex(parcels.city_id, buildings.parcel_id).fillna(0)
#     cites = [551, 1155, 1100, 3130, 6020, 6040]
#     sample = buildings[buildings.residential_units > 0]
#     sample = sample[~(sample.index.isin(households.building_id))]
#     for c in city_id.unique():
#         # sample 90% for cites list and 0 other cities
#         frac = 0.1 if c in cites else 0
#         sampled_indexes = (
#             sample[sample.index.isin(city_id[city_id == c].index)]
#             .sample(frac=frac, replace=False)
#             .index
#         )
#         # assign 1 to HU to block them from hlcm
#         series[series.index.isin(sampled_indexes)] = 1
#     return series


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

## Accessibility Variable Generator Function
def make_bld_accessibility_var(column_name, indicator_table_name, fillna_value):
    """
    Generator function to create and register an Orca column for
    accessibility variables
    """
    @orca.column("buildings", column_name, cache=True, cache_scope="iteration")
    def func(buildings):
        # Select the correct indicator table based on the indicator_table_name passed 
        # to the generator function, which is retained in this closure.
        if indicator_table_name == "accessibility_walk_indicator_by_parcel":
            indicator_table = orca.get_table("accessibility_walk_indicator_by_parcel")
        elif indicator_table_name == "accessibility_bike_indicator_by_parcel":
            indicator_table = orca.get_table("accessibility_bike_indicator_by_parcel")
        elif indicator_table_name == "accessibility_drive_indicator_by_parcel":
            indicator_table = orca.get_table("accessibility_drive_indicator_by_parcel")
        else:
            raise ValueError(f"Unknown indicator table: {indicator_table_name}")
        # Replicate the core logic: get column, reindex to parcels, fill NaN
        return misc.reindex(indicator_table[column_name], buildings.parcel_id).fillna(fillna_value)
    return func

### Load transportation accessibility variables for walk, bike and drive
# defined in assumptions.py
NEAR_MAX_VARS = orca.get_injectable("NEAR_MAX_VARS")
CUMULATIVE_VARS = orca.get_injectable("CUMULATIVE_VARS")

# Define Near-Max Variables
for mode, config in NEAR_MAX_VARS.items():
    indicator_table = config["indicator_table"]
    fillna_val = config["fillna_val"]
    # Loop over the list of full column names
    for column_name in config["column_names"]:
        # Call the generator function to create and register the Orca column
        make_bld_accessibility_var(
            column_name=column_name,
            indicator_table_name=indicator_table,
            fillna_value=fillna_val,
        )

# Define Cumulative Variables
for mode, config in CUMULATIVE_VARS.items():
    indicator_table = config["indicator_table"]
    fillna_val = config["fillna_val"]
    # Loop over the list of full column names
    for column_name in config["column_names"]:
        # Call the generator function to create and register the Orca column
        make_bld_accessibility_var(
            column_name=column_name,
            indicator_table_name=indicator_table,
            fillna_value=fillna_val,
        )

#####################
# DEMOLITION SCORING VARIABLES
#####################

@orca.column("buildings", cache=True, cache_scope="iteration")
def impr_value_per_sqft(buildings, parcels):
    """Parcel improvement value per building sqft — low value flags blight risk."""
    bldgimpr = misc.reindex(parcels.bldgimprval, buildings.parcel_id).fillna(0)
    total_sqft = (
        buildings.residential_units * buildings.sqft_per_unit
        + buildings.non_residential_sqft
    ).clip(lower=1)
    return (bldgimpr / total_sqft).clip(lower=0, upper=500)


def _leave_one_out_price(
    neighborhood_mean, observations, own_price, is_price_observation
):
    """Remove a qualifying building's own price from a node price average."""
    result = neighborhood_mean.copy()
    has_peer = is_price_observation & (observations > 1)
    result.loc[has_peer] = (
        (neighborhood_mean.loc[has_peer] * observations.loc[has_peer]
         - own_price.loc[has_peer])
        / (observations.loc[has_peer] - 1)
    )
    # A qualifying building with no other price observation has no local peer
    # price.  Do not fall back to its own target value.
    result.loc[is_price_observation & ~has_peer] = 0
    return result.fillna(0)


@orca.column("buildings", cache=True, cache_scope="iteration")
def nodes_walk_residential_excl_self(buildings):
    """Residential node price average with the focal building excluded."""
    own_price = buildings.sqft_price_res
    is_price_observation = (
        buildings.building_type_id.between(81, 84)
        & own_price.gt(0)
        & own_price.lt(650)
    )
    return _leave_one_out_price(
        buildings.nodes_walk_residential,
        buildings.nodes_walk_residential_price_observations,
        own_price,
        is_price_observation,
    )


def _register_nonres_price_excl_self(raw_name, observation_name, general_type=None):
    """Register one non-residential node price mean with focal building removed."""
    column_name = f"nodes_walk_{raw_name}_excl_self"

    @orca.column("buildings", column_name, cache=True, cache_scope="iteration")
    def price_excl_self(buildings):
        if general_type is None:
            is_price_observation = buildings.building_type_id.between(21, 71)
        else:
            is_price_observation = buildings.general_type.eq(general_type)
        return _leave_one_out_price(
            getattr(buildings, f"nodes_walk_{raw_name}"),
            getattr(buildings, f"nodes_walk_{observation_name}"),
            buildings.sqft_price_nonres,
            is_price_observation,
        )


_register_nonres_price_excl_self(
    "ave_nonres_sqft_price", "ave_nonres_sqft_price_observations"
)
for _price_type in ["Retail", "Office", "Industrial", "Medical", "Entertainment", "Hospitality"]:
    _price_name = _price_type.lower()
    _register_nonres_price_excl_self(
        _price_name, f"{_price_name}_price_observations", _price_type
    )


@orca.column("buildings", cache=True, cache_scope="iteration")
def land_to_impr_ratio(buildings, parcels):
    """Land value divided by improvement value — high ratio flags teardown pressure."""
    land = misc.reindex(parcels.landvalue, buildings.parcel_id).fillna(0)
    impr = misc.reindex(parcels.bldgimprval, buildings.parcel_id).fillna(1).clip(lower=1)
    return (land / impr).clip(lower=0, upper=50)


@orca.column("buildings", cache=True, cache_scope="iteration")
def res_vacancy_rate(buildings):
    """Proportion of residential units without a placed household (0–1)."""
    vac = buildings.vacant_residential_units.clip(lower=0)
    rate = (vac / buildings.residential_units.clip(lower=1)).clip(0, 1)
    return rate.where(buildings.residential_units > 0, 0.0)


#####################
# TRAVEL SURVEY VARIABLES (buildings)
# Inherited from parcel-level BG aggregates via parcel_id.
# Static — cache_scope='forever'.
#####################

from variables.variables_parcel import SURVEY_VARS as _SURVEY_VARS


def _make_building_survey_var(var_name):
    """Register one building-level travel survey column via parcel broadcast."""

    @orca.column("buildings", var_name, cache=True, cache_scope="forever")
    def _col(buildings, parcels):
        if var_name not in parcels.columns:
            return pd.Series(np.nan, index=buildings.index)
        return misc.reindex(parcels[var_name], buildings.parcel_id).fillna(0)

    return _col


for _sv in _SURVEY_VARS:
    _make_building_survey_var(_sv)
