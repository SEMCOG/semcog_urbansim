import numpy as np
import orca
import pandas as pd
from urbansim.utils import misc

#####################
# Census Tract VARIABLES
#####################

@orca.column("census_tracts", cache=True, cache_scope="iteration")
def popden(parcels, households):
    return (
        households.persons.groupby(households.tract_id).sum()
        / parcels.acres.groupby(parcels.tract_id).sum().clip(lower=1)
    ).fillna(0)


@orca.column("census_tracts", cache=True)
def jobs_within_30_min(jobs, parcels, travel_data):
    j = pd.DataFrame({"zone_id": jobs.tract_id})
    td = travel_data.to_frame()
    zone_ids = np.unique(td.reset_index().to_zone_id)
    zone_var = misc.compute_range(
        td,
        j.groupby("zone_id").size().reindex(index=zone_ids).fillna(0),
        "am_auto_total_time",
        30,
        agg=np.sum,
    )
    # aggregate back to parcels and then tracts
    parcel_var = misc.reindex(zone_var, parcels.zone_id)
    # now aggregate to tracts using average
    tract_var = parcel_var.groupby(parcels.tract_id).mean()
    return tract_var.fillna(0)


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def households(households, census_tracts):
    return households.tract_id.groupby(households.tract_id).size().reindex(census_tracts.index).fillna(0)


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def population(households, census_tracts):
    return households.persons.groupby(households.tract_id).sum().reindex(census_tracts.index).fillna(0)


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def employment(jobs, parcels, travel_data, census_tracts):
    td = travel_data.to_frame()
    zone_id = np.unique(td.reset_index().to_zone_id)
    j = pd.DataFrame({"zone_id": jobs.zone_id})
    zone_var = j.groupby("zone_id").size().reindex(index=zone_id).fillna(0)
    # aggregate back to parcels and then tracts
    parcel_var = misc.reindex(zone_var, parcels.zone_id)
    # now aggregate to tracts using average
    tract_var = parcel_var.groupby(parcels.tract_id).mean()
    return tract_var.reindex(census_tracts.index).fillna(0)


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def retail_jobs(jobs, parcels, travel_data, census_tracts):
    td = travel_data.to_frame()
    zone_id = np.unique(td.reset_index().to_zone_id)
    j = pd.DataFrame({"zone_id": jobs.zone_id, "sector_id": jobs.sector_id})
    zone_var = (
        j.loc[j.sector_id == 5, :]
        .groupby("zone_id")
        .size()
        .reindex(index=zone_id)
        .fillna(0)
    )
    # aggregate back to parcels and then tracts
    parcel_var = misc.reindex(zone_var, parcels.zone_id)
    # now aggregate to tracts using average
    tract_var = parcel_var.groupby(parcels.tract_id).mean()
    return tract_var.reindex(census_tracts.index).fillna(0)


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def empden(
    census_tracts, parcels
):
    return (census_tracts.employment / parcels.acres.groupby(parcels.tract_id).sum()).reindex(census_tracts.index).fillna(0)


def logsum_based_accessibility(travel_data, zones, name_attribute, spatial_var):
    td = travel_data.to_frame()
    zones = zones.to_frame(["population", "employment"])

    td = td.reset_index()
    zones = zones.reset_index()
    unique_zone_ids = np.unique(zones.zone_id.values)

    zones.index = zones.index.values + 1
    zone_id_xref = dict(list(zip(zones.zone_id, zones.index.values)))
    apply_xref = lambda x: zone_id_xref[x]

    td = td[td.from_zone_id.isin(unique_zone_ids)]
    td = td[td.to_zone_id.isin(unique_zone_ids)]

    td["from_zone_id2"] = td.from_zone_id.apply(apply_xref)
    td["to_zone_id2"] = td.to_zone_id.apply(apply_xref)

    rows = td["from_zone_id2"]
    cols = td["to_zone_id2"]

    logsums = 0 * np.ones(
        (rows.max() + 1, cols.max() + 1), dtype=td[name_attribute].dtype
    )
    logsums.put(indices=rows * logsums.shape[1] + cols, values=td[name_attribute])

    population = zones[spatial_var].values
    population = population[np.newaxis, :]

    zone_ids = zones.index.values
    zone_matrix = population * np.exp(logsums[zone_ids, :][:, zone_ids])
    zone_matrix[np.isnan(zone_matrix)] = 0
    results = pd.Series(zone_matrix.sum(axis=1), index=zones.index.values)
    zones["logsum_var"] = results
    zones = zones.reset_index().set_index("zone_id")
    return zones.logsum_var


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def logsum_pop_high_income(zones, parcels, travel_data):
    name_attribute = "am_work_highinc_logsum"
    spatial_var = "population"
    zone_var = logsum_based_accessibility(travel_data, zones, name_attribute, spatial_var)
    # aggregate back to parcels and then tracts
    parcel_var = misc.reindex(zone_var, parcels.zone_id)
    # now aggregate to tracts using average
    tract_var = parcel_var.groupby(parcels.tract_id).mean()
    return tract_var


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def logsum_pop_mid_income(zones, parcels, travel_data):
    name_attribute = "am_work_midinc_logsum"
    spatial_var = "population"
    zone_var = logsum_based_accessibility(travel_data, zones, name_attribute, spatial_var)
    # aggregate back to parcels and then tracts
    parcel_var = misc.reindex(zone_var, parcels.zone_id)
    # now aggregate to tracts using average
    tract_var = parcel_var.groupby(parcels.tract_id).mean()
    return tract_var


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def logsum_pop_low_income(zones, parcels, travel_data):
    name_attribute = "am_work_lowinc_logsum"
    spatial_var = "population"
    zone_var = logsum_based_accessibility(travel_data, zones, name_attribute, spatial_var)
    # aggregate back to parcels and then tracts
    parcel_var = misc.reindex(zone_var, parcels.zone_id)
    # now aggregate to tracts using average
    tract_var = parcel_var.groupby(parcels.tract_id).mean()
    return tract_var


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def logsum_job_high_income(zones, parcels, travel_data):
    name_attribute = "am_work_highinc_logsum"
    spatial_var = "employment"
    zone_var = logsum_based_accessibility(travel_data, zones, name_attribute, spatial_var)
    # aggregate back to parcels and then tracts
    parcel_var = misc.reindex(zone_var, parcels.zone_id)
    # now aggregate to tracts using average
    tract_var = parcel_var.groupby(parcels.tract_id).mean()
    return tract_var


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def logsum_job_mid_income(zones, parcels, travel_data):
    name_attribute = "am_work_midinc_logsum"
    spatial_var = "employment"
    zone_var = logsum_based_accessibility(travel_data, zones, name_attribute, spatial_var)
    # aggregate back to parcels and then tracts
    parcel_var = misc.reindex(zone_var, parcels.zone_id)
    # now aggregate to tracts using average
    tract_var = parcel_var.groupby(parcels.tract_id).mean()
    return tract_var


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def logsum_job_low_income(zones, parcels, travel_data):
    name_attribute = "am_work_lowinc_logsum"
    spatial_var = "employment"
    zone_var = logsum_based_accessibility(travel_data, zones, name_attribute, spatial_var)
    # aggregate back to parcels and then tracts
    parcel_var = misc.reindex(zone_var, parcels.zone_id)
    # now aggregate to tracts using average
    tract_var = parcel_var.groupby(parcels.tract_id).mean()
    return tract_var


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def z_total_jobs(jobs, census_tracts):
    return jobs.tract_id.value_counts().reindex(census_tracts.index).fillna(0)

@orca.column("census_tracts", cache=True, cache_scope="iteration")
def transit_jobs_60min(census_tracts, parcels, travel_data):
    td = travel_data.to_frame(["am_transit_total_time"]).reset_index()
    zemp = census_tracts.to_frame(["employment"])
    temp = pd.merge(td, zemp, left_on="to_zone_id", right_index=True, how="left")
    zone_var = (
        temp[temp.am_transit_total_time <= 60].groupby("from_zone_id").employment.sum()
    )
    # aggregate back to parcels and then tracts
    parcel_var = misc.reindex(zone_var, parcels.zone_id)
    # now aggregate to tracts using average
    tract_var = parcel_var.groupby(parcels.tract_id).mean()
    return tract_var.fillna(0)

@orca.column("census_tracts", cache=True, cache_scope="iteration")
def transit_jobs_50min(census_tracts, parcels, travel_data):
    td = travel_data.to_frame(["am_transit_total_time"]).reset_index()
    zemp = census_tracts.to_frame(["employment"])
    temp = pd.merge(td, zemp, left_on="to_zone_id", right_index=True, how="left")
    zone_var = (
        temp[temp.am_transit_total_time <= 50].groupby("from_zone_id").employment.sum()
    )
    # aggregate back to parcels and then tracts
    parcel_var = misc.reindex(zone_var, parcels.zone_id)
    # now aggregate to tracts using average
    tract_var = parcel_var.groupby(parcels.tract_id).mean()
    return tract_var.fillna(0)


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def transit_jobs_30min(census_tracts, parcels, travel_data):
    td = travel_data.to_frame(["am_transit_total_time"]).reset_index()
    zemp = census_tracts.to_frame(["employment"])
    temp = pd.merge(td, zemp, left_on="to_zone_id", right_index=True, how="left")
    zone_var = (
        temp[temp.am_transit_total_time <= 30].groupby("from_zone_id").employment.sum()
    )
    # aggregate back to parcels and then tracts
    parcel_var = misc.reindex(zone_var, parcels.zone_id)
    # now aggregate to tracts using average
    tract_var = parcel_var.groupby(parcels.tract_id).mean()
    return tract_var.fillna(0)


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def a_ln_emp_26min_drive_alone(census_tracts, parcels, travel_data):
    drvtime = travel_data.to_frame(["am_auto_total_time"]).reset_index()
    zemp = census_tracts.to_frame(["employment"])
    temp = pd.merge(drvtime, zemp, left_on="to_zone_id", right_index=True, how="left")
    zone_var = np.log1p(
        temp[temp.am_auto_total_time <= 26]
        .groupby("from_zone_id")
        .employment.sum()
        .fillna(0)
    )
    # aggregate back to parcels and then tracts
    parcel_var = misc.reindex(zone_var, parcels.zone_id)
    # now aggregate to tracts using average
    tract_var = parcel_var.groupby(parcels.tract_id).mean()
    return tract_var.fillna(0)


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def a_ln_emp_50min_transit(census_tracts, parcels, travel_data):
    transittime = travel_data.to_frame(["am_transit_total_time"]).reset_index()
    zemp = census_tracts.to_frame(["employment"])
    temp = pd.merge(
        transittime, zemp, left_on="to_zone_id", right_index=True, how="left"
    )
    zone_var = np.log1p(
        temp[temp.am_transit_total_time <= 50]
        .groupby("from_zone_id")
        .employment.sum()
        .fillna(0)
    )
    # aggregate back to parcels and then tracts
    parcel_var = misc.reindex(zone_var, parcels.zone_id)
    # now aggregate to tracts using average
    tract_var = parcel_var.groupby(parcels.tract_id).mean()
    return tract_var.fillna(0)


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def a_ln_retail_emp_15min_drive_alone(census_tracts, parcels, travel_data):
    drvtime = travel_data.to_frame(["midday_auto_total_time"]).reset_index()
    zemp = census_tracts.to_frame(["employment"])
    temp = pd.merge(drvtime, zemp, left_on="to_zone_id", right_index=True, how="left")
    zone_var = np.log1p(
        temp[temp.midday_auto_total_time <= 15]
        .groupby("from_zone_id")
        .employment.sum()
        .fillna(0)
    )
    # aggregate back to parcels and then tracts
    parcel_var = misc.reindex(zone_var, parcels.zone_id)
    # now aggregate to tracts using average
    tract_var = parcel_var.groupby(parcels.tract_id).mean()
    return tract_var.fillna(0)


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def percent_vacant_job_spaces(buildings, parcels):
    buildings = buildings.to_frame(
        buildings.local_columns + ["job_spaces", "vacant_job_spaces", "zone_id"]
    )
    job_spaces = buildings.groupby("zone_id").job_spaces.sum()
    vacant_job_spaces = buildings.groupby("zone_id").vacant_job_spaces.sum()

    zone_var = (
        (vacant_job_spaces * 1.0 / job_spaces)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    # aggregate back to parcels and then tracts
    parcel_var = misc.reindex(zone_var, parcels.zone_id)
    # now aggregate to tracts using average
    tract_var = parcel_var.groupby(parcels.tract_id).mean()
    return tract_var.fillna(0)


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def avg_sqft_price_res(buildings, census_tracts):
    buildings = buildings.to_frame( ["sqft_price_res", "tract_id"])
    sqft_price_res = buildings.groupby("tract_id").sqft_price_res.median()
    return sqft_price_res.reindex(census_tracts.index).fillna(0)


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def avg_sqft_price_res(buildings, census_tracts):
    buildings = buildings.to_frame( ["sqft_price_res", "tract_id"])
    sqft_price_res = buildings.groupby("tract_id").sqft_price_res.median()
    return sqft_price_res.reindex(census_tracts.index).fillna(0)


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def crime_ucr_rate(buildings, census_tracts):
    buildings = buildings.to_frame( ["crime_ucr_rate", "tract_id"])
    crime_ucr_rate = buildings.groupby("tract_id").crime_ucr_rate.mean()
    return crime_ucr_rate.reindex(census_tracts.index).fillna(0)


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def crime_other_rate(buildings, census_tracts):
    buildings = buildings.to_frame( ["crime_other_rate", "tract_id"])
    crime_other_rate = buildings.groupby("tract_id").crime_other_rate.mean()
    return crime_other_rate.reindex(census_tracts.index).fillna(0)


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def percent_vacant_residential_units(buildings, parcels):
    buildings = buildings.to_frame(
        buildings.local_columns + ["vacant_residential_units", "zone_id"]
    )
    du = buildings.groupby("zone_id").residential_units.sum()
    vacant_du = buildings.groupby("zone_id").vacant_residential_units.sum()

    zone_var = (vacant_du * 1.0 / du).replace([np.inf, -np.inf], np.nan).fillna(0)
    # aggregate back to parcels and then tracts
    parcel_var = misc.reindex(zone_var, parcels.zone_id)
    # now aggregate to tracts using average
    tract_var = parcel_var.groupby(parcels.tract_id).mean()
    return tract_var.fillna(0)


def make_employment_density_variable(sector_id):
    """
    Generate zonal employment density variable. Registers with orca.
    """
    var_name = "ln_empden_%s" % sector_id

    @orca.column("census_tracts", var_name, cache=True, cache_scope="iteration")
    def func():
        zones = orca.get_table("zones")
        jobs = orca.get_table("jobs")
        parcels = orca.get_table("parcels")
        total_acres = zones.acres
        jobs = jobs.to_frame(jobs.local_columns + ["zone_id"])
        jobs_sector = jobs[jobs.sector_id == sector_id].zone_id.value_counts()
        zone_var = np.log1p(jobs_sector / total_acres).fillna(0)
        # aggregate back to parcels and then tracts
        parcel_var = misc.reindex(zone_var, parcels.zone_id)
        # now aggregate to tracts using average
        tract_var = parcel_var.groupby(parcels.tract_id).mean()
        return tract_var

    return func


emp_sectors = np.arange(18) + 1
for sector in emp_sectors:
    make_employment_density_variable(sector)


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def mean_age_of_head(households):
    return households.age_of_head.groupby(households.tract_id).mean()


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def prop_race_1(census_tracts, households):
    households = households.to_frame(["race_id", "tract_id"])
    return (households.query("race_id == 1").groupby("tract_id").size() / census_tracts.households).reindex(census_tracts.index).fillna(0)


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def prop_race_2(census_tracts, households):
    households = households.to_frame(["race_id", "tract_id"])
    return (households.query("race_id == 2").groupby("tract_id").size() / census_tracts.households).reindex(census_tracts.index).fillna(0)


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def prop_race_3(census_tracts, households):
    households = households.to_frame(["race_id", "tract_id"])
    return (households.query("race_id == 3").groupby("tract_id").size() / census_tracts.households).reindex(census_tracts.index).fillna(0)


@orca.column("census_tracts", cache=True, cache_scope="iteration")
def prop_race_4(census_tracts, households):
    households = households.to_frame(["race_id", "tract_id"])
    return (households.query("race_id == 4").groupby("tract_id").size() / census_tracts.households).reindex(census_tracts.index).fillna(0)


##########  Parcel vars to add for proforma calibration



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


geographic_levels = [("census_tracts", "zone_id")]

for geography in geographic_levels:
    geography_name = geography[0]
    geography_id = geography[1]
    if geography_name != "parcels":
        parcel_vars = orca.get_table("parcels").columns
        for var in orca.get_table(geography_name).columns:
            if var not in parcel_vars:
                make_disagg_var(geography_name, "parcels", var, geography_id)


def standardize(series):
    if series.dtype != object:
        series = (series - series.mean()) / series.std()
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


for var in orca.get_table("parcels").columns:
    register_standardized_variable("parcels", var)



#####################
# TRAVEL SURVEY VARIABLES (census tracts)
# Base-year behavioral variables from SEMCOG Regional Travel Survey.
# Aggregated from block groups via parcel-count-weighted crosswalk.
# Static — cache_scope='forever'.
#####################

from variables.variables_parcel import SURVEY_VARS as _SURVEY_VARS, SURVEY_PARCEL_VARS as _SURVEY_PARCEL_VARS


def _make_tract_survey_var(var_name):
    """Aggregate one survey variable from all buildings to census tracts (simple mean)."""

    @orca.column("census_tracts", var_name, cache=True, cache_scope="forever")
    def _col(buildings, parcels, census_tracts):
        p = parcels.to_frame(["census_bg_id", "county_id"])
        p["tract_id"] = (p["census_bg_id"] // 1000 + p["county_id"] * 10000).astype(np.int64)
        b = buildings.to_frame(["parcel_id", var_name])
        b = b.join(p["tract_id"], on="parcel_id")
        return b.groupby("tract_id")[var_name].mean().reindex(census_tracts.index).fillna(0)

    return _col


for _sv in _SURVEY_VARS + _SURVEY_PARCEL_VARS:
    _make_tract_survey_var(_sv)
