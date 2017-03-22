import numpy as np
import pandas as pd
import os
import assumptions
from urbansim.utils import misc

import warnings

##
import orca
import utils


warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

@orca.table()
def scheduled_development_events(store):
    df = pd.read_csv("data/scheduled_development_events.csv")
    return df

@orca.table('jobs')
def jobs(store):
    df = store['jobs']
    return df

@orca.table()
def buildings(store):
    buildings = store['buildings']
    buildings['sqft_price_nonres'] = buildings.improvement_value*1.0 / 0.7 / buildings.non_residential_sqft
    buildings.sqft_price_nonres[buildings.sqft_price_nonres==np.inf] = 0
    buildings['sqft_price_res'] = buildings.improvement_value*1.25 / 0.7 / (buildings.sqft_per_unit * buildings.residential_units)
    buildings.loc[buildings.sqft_price_res == np.inf, 'sqft_price_res'] = 0
    buildings.fillna(0, inplace=True)
    return buildings

@orca.table()
def households(store):
    df = store['households']
    df.loc[df.building_id == -1, 'building_id'] = np.random.choice(store.buildings.index.values,
                                                                   (df.building_id == -1).sum())
    idx_invalid_building_id = np.in1d(df.building_id, store.buildings.index.values) == False
    df.loc[idx_invalid_building_id, 'building_id'] = np.random.choice(store.buildings.index.values,
                                                                      idx_invalid_building_id.sum())
    return df
    
@orca.table()
def persons(store):
    df = store['persons']
    return df

@orca.table()
def parcels(store):
    df = store['parcels']
    return df

@orca.table()
def zones(store):
    df = store['zones']
    return df
    
@orca.table()
def cities(store):
    df = store['cities']
    return df


@orca.table()
def counties(store):
    df = store['counties']
    return df


@orca.table()
def employment_sectors(store):
    df = store['employment_sectors']
    return df


@orca.table()
def home_based_status(store):
    df = store['home_based_status']
    return df


@orca.table()
def target_vacancies(store):
    df = store['target_vacancies']
    return df


@orca.table()
def building_sqft_per_job(store):
    df = store['building_sqft_per_job']
    return df
    
@orca.table()
def annual_relocation_rates_for_households(store):
    df = store['annual_relocation_rates_for_households']
    return df
    
@orca.table()
def annual_relocation_rates_for_jobs(store):
    df = store['annual_relocation_rates_for_jobs']
    return df
    
@orca.table()
def annual_household_control_totals(store):
    df = store['annual_household_control_totals']
    return df

@orca.table()
def annual_employment_control_totals(store):
    df = store['annual_employment_control_totals']
    return df
    
@orca.table()
def travel_data(store):
    df = store['travel_data']
    return df


@orca.table()
def zoning(store):
    df = store['zoning']
    return df

@orca.table()
def large_areas(store):
    df = store['large_areas']
    return df

@orca.table()
def building_types(store):
    df = store['building_types']
    return df

@orca.table()
def land_use_types(store):
    df = store['land_use_types']
    return df


@orca.table()
def access_drive_minutes(store):
    df = store['access_drive_minutes']
    return df


@orca.table()
def access_walk_feet(store):
    df = store['access_walk_feet']
    return df


@orca.table()
def transit_stops(store):
    df = store['transit_stops']
    return df


# these are dummy returns that last until accessibility runs
@orca.table()
def nodes():
    return pd.DataFrame()


@orca.table()
def nodes_walk():
    return pd.DataFrame()


@orca.table()
def nodes_drv():
    return pd.DataFrame()


@orca.table()
def crime_rates(store):
    df = store['crime_rates']
    return df


@orca.table()
def schools(store):
    df = store['schools']   
    return df

@orca.table()
def poi(store):
    df = store['poi']   
    return df

# GQ placeholders
for gq_tbl in ['tazcounts2040gq', 'tazcounts2015gq', 'tazcounts2020gq', 'tazcounts2035gq', 'tazcounts2025gq', 'tazcounts2030gq']:
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


