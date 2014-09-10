import numpy as np
import pandas as pd
import os
import assumptions
from urbansim.utils import misc
import urbansim.sim.simulation as sim

import warnings
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

@sim.table_source('scheduled_development_events')
def scheduled_development_events(store):
    df = pd.read_csv("data/scheduled_development_events.csv")
    return df

@sim.table_source('jobs')
def jobs(store):
    df = store['jobs']
    return df

@sim.table_source('buildings')
def buildings(store):
    buildings = store['buildings']
    buildings['sqft_price_nonres'] = buildings.improvement_value*1.0 / buildings.non_residential_sqft
    buildings.sqft_price_nonres[buildings.sqft_price_nonres==np.inf] = 0
    buildings['sqft_price_res'] = buildings.improvement_value*1.25 / (buildings.sqft_per_unit * buildings.residential_units)
    buildings.sqft_price_res[buildings.sqft_price_res==np.inf] = 0
    return buildings

@sim.table_source('households')
def households(store):
    df = store['households']
    return df
    
@sim.table_source('persons')
def persons(store):
    df = store['persons']
    return df

@sim.table_source('parcels')
def parcels(store):
    df = store['parcels']
    return df

@sim.table_source('zones')
def zones(store):
    df = store['zones']
    return df
    
@sim.table_source('cities')
def cities(store):
    df = store['cities']
    return df
    
@sim.table_source('building_sqft_per_job')
def building_sqft_per_job(store):
    df = store['building_sqft_per_job']
    return df
    
@sim.table_source('annual_relocation_rates_for_households')
def annual_relocation_rates_for_households(store):
    df = store['annual_relocation_rates_for_households']
    return df
    
@sim.table_source('annual_relocation_rates_for_jobs')
def annual_relocation_rates_for_jobs(store):
    df = store['annual_relocation_rates_for_jobs']
    return df
    
@sim.table_source('annual_household_control_totals')
def annual_household_control_totals(store):
    df = store['annual_household_control_totals']
    return df

@sim.table_source('annual_employment_control_totals')
def annual_employment_control_totals(store):
    df = store['annual_employment_control_totals']
    return df
    
@sim.table_source('travel_data')
def travel_data(store):
    df = store['travel_data']
    return df
    
# these are dummy returns that last until accessibility runs
@sim.table("nodes")
def nodes():
    return pd.DataFrame()

@sim.table("nodes_prices")
def nodes_prices():
    return pd.DataFrame()
    
# GQ placeholders
for gq_tbl in ['tazcounts2040gq', 'tazcounts2015gq', 'tazcounts2020gq', 'tazcounts2035gq', 'tazcounts2025gq', 'tazcounts2030gq']:
    empty_df = pd.DataFrame()
    sim.add_table(gq_tbl,empty_df)

# this specifies the relationships between tables
sim.broadcast('nodes', 'buildings', cast_index=True, onto_on='_node_id')
sim.broadcast('nodes', 'parcels', cast_index=True, onto_on='_node_id')
sim.broadcast('nodes_prices', 'buildings', cast_index=True, onto_on='_node_id')
sim.broadcast('parcels', 'buildings', cast_index=True, onto_on='parcel_id')
sim.broadcast('buildings', 'households', cast_index=True,
              onto_on='building_id')
sim.broadcast('buildings', 'jobs', cast_index=True, onto_on='building_id')
sim.broadcast('households', 'persons', cast_index=True, onto_on='household_id')
