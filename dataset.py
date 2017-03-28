import warnings

import numpy as np
import orca
import pandas as pd

warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)


@orca.table()
def scheduled_development_events():
    return pd.read_csv("data/scheduled_development_events.csv")


@orca.table()
def buildings(store):
    df = store['buildings']
    df['sqft_price_nonres'] = df.improvement_value * 1.0 / 0.7 / df.non_residential_sqft
    df.sqft_price_nonres[df.sqft_price_nonres == np.inf] = 0
    df.loc[df.sqft_price_nonres == np.inf, 'sqft_price_nonres'] = 0
    df['sqft_price_res'] = df.improvement_value * 1.25 / 0.7 / (df.sqft_per_unit * df.residential_units)
    df.loc[df.sqft_price_res == np.inf, 'sqft_price_res'] = 0
    df.fillna(0, inplace=True)
    return df


@orca.table()
def households(store):
    df = store['households']
    df.loc[df.building_id == -1, 'building_id'] = np.random.choice(store.buildings.index.values,
                                                                   (df.building_id == -1).sum())
    idx_invalid_building_id = np.in1d(df.building_id, store.buildings.index.values) is False
    df.loc[idx_invalid_building_id, 'building_id'] = np.random.choice(store.buildings.index.values,
                                                                      idx_invalid_building_id.sum())
    return df


@orca.table()
def jobs(store):
    return store['jobs']


@orca.table()
def persons(store):
    return store['persons']


@orca.table()
def parcels(store):
    return store['parcels']


@orca.table()
def zones(store):
    return store['zones']


@orca.table()
def cities(store):
    return store['cities']


@orca.table()
def counties(store):
    return store['counties']


@orca.table()
def employment_sectors(store):
    return store['employment_sectors']


@orca.table()
def home_based_status(store):
    return store['home_based_status']


@orca.table()
def target_vacancies(store):
    return store['target_vacancies']


@orca.table()
def building_sqft_per_job(store):
    return store['building_sqft_per_job']


@orca.table()
def annual_relocation_rates_for_households(store):
    return store['annual_relocation_rates_for_households']


@orca.table()
def annual_relocation_rates_for_jobs(store):
    return store['annual_relocation_rates_for_jobs']


@orca.table()
def annual_household_control_totals(store):
    return store['annual_household_control_totals']


@orca.table()
def annual_employment_control_totals(store):
    return store['annual_employment_control_totals']


@orca.table()
def travel_data(store):
    return store['travel_data']


@orca.table()
def zoning(store):
    return store['zoning']


@orca.table()
def large_areas(store):
    return store['large_areas']


@orca.table()
def building_types(store):
    return store['building_types']


@orca.table()
def land_use_types(store):
    return store['land_use_types']


@orca.table()
def access_drive_minutes(store):
    return store['access_drive_minutes']


@orca.table()
def access_walk_feet(store):
    return store['access_walk_feet']


@orca.table()
def transit_stops(store):
    return store['transit_stops']


@orca.table()
def crime_rates(store):
    return store['crime_rates']


@orca.table()
def schools(store):
    return store['schools']


@orca.table()
def poi(store):
    return store['poi']


# these are dummy returns that last until accessibility runs
for gq_tbl in ['nodes', 'nodes_walk', 'nodes_drv']:
    empty_df = pd.DataFrame()
    orca.add_table(gq_tbl, empty_df)

# GQ placeholders
for gq_tbl in ['tazcounts2040gq', 'tazcounts2015gq', 'tazcounts2020gq', 'tazcounts2035gq', 'tazcounts2025gq',
               'tazcounts2030gq']:
    empty_df = pd.DataFrame()
    orca.add_table(gq_tbl, empty_df)

# this specifies the relationships between tables
orca.broadcast('nodes_walk', 'buildings', cast_index=True, onto_on='nodeid_walk')
orca.broadcast('nodes_walk', 'parcels', cast_index=True, onto_on='nodeid_walk')
orca.broadcast('nodes_drv', 'buildings', cast_index=True, onto_on='nodeid_drv')
orca.broadcast('nodes_drv', 'parcels', cast_index=True, onto_on='nodeid_drv')
# orca.broadcast('transit_to_jobs_am', 'parcels', cast_index=True, onto_on='nodeid_walk')
# orca.broadcast('transit_to_jobs_midday', 'parcels', cast_index=True, onto_on='nodeid_walk')
orca.broadcast('parcels', 'buildings', cast_index=True, onto_on='parcel_id')
orca.broadcast('buildings', 'households', cast_index=True, onto_on='building_id')
orca.broadcast('buildings', 'jobs', cast_index=True, onto_on='building_id')
orca.broadcast('households', 'persons', cast_index=True, onto_on='household_id')
orca.broadcast('building_types', 'buildings', cast_index=True, onto_on='building_type_id')
orca.broadcast('zones', 'parcels', cast_index=True, onto_on='zone_id')
orca.broadcast('schools', 'parcels', cast_on='parcel_id', onto_index=True)
