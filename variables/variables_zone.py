import numpy as np
import orca
import pandas as pd


#####################
# ZONES VARIABLES
#####################

@orca.column('zones', cache=True, cache_scope='iteration')
def popden(parcels, households):
    return households.persons.groupby(households.zone_id).sum() / parcels.acres.groupby(parcels.zone_id).sum()


@orca.column('zones', cache=True)
def jobs_within_30_min(jobs, travel_data):
    from urbansim.utils import misc
    j = pd.DataFrame({'zone_id': jobs.zone_id})
    td = travel_data.to_frame()
    zone_ids = np.unique(td.reset_index().to_zone_id)
    return misc.compute_range(td,
                              j.groupby('zone_id').size().reindex(index=zone_ids).fillna(0),
                              "am_auto_total_time",
                              30, agg=np.sum)


@orca.column('zones', cache=True, cache_scope='iteration')
def households(households):
    print type(households)
    return households.zone_id.groupby(households.zone_id).size()


@orca.column('zones', cache=True, cache_scope='iteration')
def population(households):
    return households.persons.groupby(households.zone_id).sum()


@orca.column('zones', cache=True, cache_scope='iteration')
def employment(jobs, travel_data):
    td = travel_data.to_frame()
    zone_ids = np.unique(td.reset_index().to_zone_id)
    j = pd.DataFrame({'zone_id': jobs.zone_id})
    return j.groupby('zone_id').size().reindex(index=zone_ids).fillna(0)


@orca.column('zones', cache=True, cache_scope='iteration')
def retail_jobs(jobs, travel_data):
    td = travel_data.to_frame()
    zone_ids = np.unique(td.reset_index().to_zone_id)
    j = pd.DataFrame({'zone_id': jobs.zone_id, 'sector_id': jobs.sector_id})
    return j.loc[j.sector_id == 5, :].groupby('zone_id').size().reindex(index=zone_ids).fillna(0)


@orca.column('zones', cache=True, cache_scope='iteration')
def empden(zones, parcels,):
    return zones.employment / parcels.acres.groupby(parcels.zone_id).sum()


def logsum_based_accessibility(travel_data, zones, name_attribute, spatial_var):
    td = travel_data.to_frame()
    zones = zones.to_frame(['population', 'employment'])

    td = td.reset_index()
    zones = zones.reset_index()
    unique_zone_ids = np.unique(zones.zone_id.values)

    zones.index = zones.index.values + 1
    zone_id_xref = dict(zip(zones.zone_id, zones.index.values))
    apply_xref = lambda x: zone_id_xref[x]

    td = td[td.from_zone_id.isin(unique_zone_ids)]
    td = td[td.to_zone_id.isin(unique_zone_ids)]

    td['from_zone_id2'] = td.from_zone_id.apply(apply_xref)
    td['to_zone_id2'] = td.to_zone_id.apply(apply_xref)

    rows = td['from_zone_id2']
    cols = td['to_zone_id2']

    logsums = 0 * np.ones((rows.max() + 1, cols.max() + 1), dtype=td[name_attribute].dtype)
    logsums.put(indices=rows * logsums.shape[1] + cols, values=td[name_attribute])

    population = zones[spatial_var].values
    population = population[np.newaxis, :]

    zone_ids = zones.index.values
    zone_matrix = population * np.exp(logsums[zone_ids, :][:, zone_ids])
    zone_matrix[np.isnan(zone_matrix)] = 0
    results = pd.Series(zone_matrix.sum(axis=1), index=zones.index.values)
    zones['logsum_var'] = results
    zones = zones.reset_index().set_index('zone_id')
    return zones.logsum_var


@orca.column('zones', cache=True, cache_scope='iteration')
def logsum_pop_high_income(zones, travel_data):
    name_attribute = 'am_work_highinc_logsum'
    spatial_var = 'population'
    return logsum_based_accessibility(travel_data, zones, name_attribute, spatial_var)


@orca.column('zones', cache=True, cache_scope='iteration')
def logsum_pop_low_income(zones, travel_data):
    name_attribute = 'am_work_lowinc_logsum'
    spatial_var = 'population'
    return logsum_based_accessibility(travel_data, zones, name_attribute, spatial_var)


@orca.column('zones', cache=True, cache_scope='iteration')
def logsum_job_high_income(zones, travel_data):
    name_attribute = 'am_work_highinc_logsum'
    spatial_var = 'employment'
    return logsum_based_accessibility(travel_data, zones, name_attribute, spatial_var)


@orca.column('zones', cache=True, cache_scope='iteration')
def logsum_job_low_income(zones, travel_data):
    name_attribute = 'am_work_lowinc_logsum'
    spatial_var = 'employment'
    return logsum_based_accessibility(travel_data, zones, name_attribute, spatial_var)


@orca.column('zones', cache=True, cache_scope='iteration')
def z_total_jobs(jobs):
    return jobs.zone_id.value_counts()


@orca.column('zones', cache=True, cache_scope='iteration')
def transit_jobs_50min(zones, travel_data):
    td = travel_data.to_frame(['am_transit_total_time']).reset_index()
    zemp = zones.to_frame(['employment'])
    temp = pd.merge(td,zemp, left_on = 'to_zone_id', right_index = True, how='left' )
    transit_jobs_50min = temp[temp.am_transit_total_time <=50].groupby('from_zone_id').employment.sum()
    return transit_jobs_50min


@orca.column('zones', cache=True, cache_scope='iteration')
def transit_jobs_30min(zones, travel_data):
    td = travel_data.to_frame(['am_transit_total_time']).reset_index()
    zemp = zones.to_frame(['employment'])
    temp = pd.merge(td,zemp, left_on = 'to_zone_id', right_index = True, how='left' )
    transit_jobs_30min = temp[temp.am_transit_total_time <=30].groupby('from_zone_id').employment.sum()
    return transit_jobs_30min


@orca.column('zones', cache=True, cache_scope='iteration')
def a_ln_emp_26min_drive_alone(zones, travel_data):
    drvtime = travel_data.to_frame(['am_auto_total_time']).reset_index()
    zemp = zones.to_frame(['employment'])
    temp = pd.merge(drvtime, zemp, left_on ='to_zone_id', right_index = True, how='left' )
    return np.log1p(temp[temp.am_auto_total_time <=26].groupby('from_zone_id').employment.sum().fillna(0))


@orca.column('zones', cache=True, cache_scope='iteration')
def a_ln_emp_50min_transit(zones, travel_data):
    transittime = travel_data.to_frame(['am_transit_total_time']).reset_index()
    zemp = zones.to_frame(['employment'])
    temp = pd.merge(transittime,zemp, left_on = 'to_zone_id', right_index = True, how='left' )
    return np.log1p(temp[temp.am_transit_total_time <=50].groupby('from_zone_id').employment.sum().fillna(0))


@orca.column('zones', cache=True, cache_scope='iteration')
def a_ln_retail_emp_15min_drive_alone(zones, travel_data):
    drvtime = travel_data.to_frame(['midday_auto_total_time']).reset_index()
    zemp = zones.to_frame(['employment'])
    temp = pd.merge(drvtime,zemp, left_on = 'to_zone_id', right_index = True, how='left' )
    return np.log1p(temp[temp.midday_auto_total_time <=15].groupby('from_zone_id').employment.sum().fillna(0))


@orca.column('zones', cache=True, cache_scope='iteration')
def percent_vacant_job_spaces(buildings):
    buildings = buildings.to_frame(buildings.local_columns + ['job_spaces', 'vacant_job_spaces', 'zone_id'])
    job_spaces = buildings.groupby('zone_id').job_spaces.sum()
    vacant_job_spaces = buildings.groupby('zone_id').vacant_job_spaces.sum()

    return (vacant_job_spaces*1.0 / job_spaces).replace([np.inf, -np.inf], np.nan).fillna(0)


@orca.column('zones', cache=True, cache_scope='iteration')
def percent_vacant_residential_units(buildings):
    buildings = buildings.to_frame(buildings.local_columns + ['vacant_residential_units', 'zone_id'])
    du = buildings.groupby('zone_id').residential_units.sum()
    vacant_du = buildings.groupby('zone_id').vacant_residential_units.sum()

    return (vacant_du*1.0 / du).replace([np.inf, -np.inf], np.nan).fillna(0)


def make_employment_density_variable(sector_id):
    """
    Generate zonal employment density variable. Registers with orca.
    """
    var_name = 'ln_empden_%s' % sector_id
    
    @orca.column('zones', var_name, cache=True, cache_scope='iteration')
    def func():
        zones = orca.get_table('zones')
        jobs = orca.get_table('jobs')
        total_acres = zones.acres
        jobs = jobs.to_frame(jobs.local_columns + ['zone_id'])
        jobs_sector = jobs[jobs.sector_id == sector_id].zone_id.value_counts()
        return np.log1p(jobs_sector / total_acres).fillna(0)

    return func


emp_sectors = np.arange(18) + 1
for sector in emp_sectors:
    make_employment_density_variable(sector)

@orca.column('zones', cache=True, cache_scope='iteration')
def mean_age_of_head(households):
    return households.age_of_head.groupby(households.zone_id).mean()

@orca.column('zones', cache=True, cache_scope='iteration')
def prop_race_1(zones, households):
    households = households.to_frame(['race_id', 'zone_id'])
    return households.query('race_id == 1').groupby('zone_id').size() / zones.households

@orca.column('zones', cache=True, cache_scope='iteration')
def prop_race_2(zones, households):
    households = households.to_frame(['race_id', 'zone_id'])
    return households.query('race_id == 2').groupby('zone_id').size() / zones.households

@orca.column('zones', cache=True, cache_scope='iteration')
def prop_race_3(zones, households):
    households = households.to_frame(['race_id', 'zone_id'])
    return households.query('race_id == 3').groupby('zone_id').size() / zones.households

@orca.column('zones', cache=True, cache_scope='iteration')
def prop_race_4(zones, households):
    households = households.to_frame(['race_id', 'zone_id'])
    return households.query('race_id == 4').groupby('zone_id').size() / zones.households

##########  Parcel vars to add for proforma calibration

from urbansim.utils import misc

def make_disagg_var(from_geog_name, to_geog_name, var_to_disaggregate, from_geog_id_name, name_based_on_geography=True):
    """
    Generator function for disaggregating variables. Registers with orca.
    """
    if name_based_on_geography:
        var_name = from_geog_name + '_' + var_to_disaggregate
    else:
        var_name = var_to_disaggregate

    @orca.column(to_geog_name, var_name, cache=True, cache_scope='iteration')
    def func():
        print 'Disaggregating {} to {} from {}'.format(var_to_disaggregate, to_geog_name, from_geog_name)

        from_geog = orca.get_table(from_geog_name)
        to_geog = orca.get_table(to_geog_name)
        return misc.reindex(from_geog[var_to_disaggregate], to_geog[from_geog_id_name]).fillna(0)

    return func


geographic_levels = [('zones', 'zone_id')]

for geography in geographic_levels:
    geography_name = geography[0]
    geography_id = geography[1]
    if geography_name != 'parcels':
        parcel_vars = orca.get_table('parcels').columns
        for var in orca.get_table(geography_name).columns:
            if var not in parcel_vars:
                make_disagg_var(geography_name, 'parcels', var, geography_id)

def standardize(series):
    return (series - series.mean()) / series.std()

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
    new_col_name = 'st_' + column_to_s
    @orca.column(table_name, new_col_name, cache=True, cache_scope='iteration')
    def column_func():
        return standardize(orca.get_table(table_name)[column_to_s])
    return column_func

for var in orca.get_table('parcels').columns:
    register_standardized_variable('parcels', var)
