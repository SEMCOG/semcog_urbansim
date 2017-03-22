import pandas as pd, numpy as np
from urbansim.utils import misc
import dataset

###
import orca
import utils
import pandana as pdna


#####################
# ZONES VARIABLES
#####################

@orca.column('zones', cache=True, cache_scope='iteration')
def popden(zones, parcels, households):
    return households.persons.groupby(households.zone_id).sum() / parcels.acres.groupby(parcels.zone_id).sum()


##@orca.column('zones', 'jobs_within_30_min', cache=True)
##def jobs_within_30_min(jobs, travel_data):
##    j = pd.DataFrame({'zone_id':jobs.zone_id})
##    td = travel_data.to_frame()
##    zone_ids = np.unique(td.reset_index().to_zone_id)
##    return misc.compute_range(td,
##                                  j.groupby('zone_id').size().reindex(index = zone_ids).fillna(0),
##                                  "am_single_vehicle_to_work_travel_time",
##                                  30, agg=np.sum)

@orca.column('zones', cache=True, cache_scope='iteration')
def households(zones, households):
    print type(households)
    return households.zone_id.groupby(households.zone_id).size()


@orca.column('zones', cache=True, cache_scope='iteration')
def population(zones, households):
    return households.persons.groupby(households.zone_id).sum()


@orca.column('zones', cache=True, cache_scope='iteration')
def employment(zones, jobs, travel_data):
    td = travel_data.to_frame()
    zone_ids = np.unique(td.reset_index().to_zone_id)
    j = pd.DataFrame({'zone_id': jobs.zone_id})
    return j.groupby('zone_id').size().reindex(index=zone_ids).fillna(0)


@orca.column('zones', cache=True, cache_scope='iteration')
def retail_jobs(zones, jobs, travel_data):
    td = travel_data.to_frame()
    zone_ids = np.unique(td.reset_index().to_zone_id)
    j = pd.DataFrame({'zone_id': jobs.zone_id})
    return j.loc[j.sector_id == 5, :].groupby('zone_id').size().reindex(index=zone_ids).fillna(0)


@orca.column('zones', cache=True, cache_scope='iteration')
def empden(zones, parcels, households):
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
