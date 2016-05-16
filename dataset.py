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
    buildings['sqft_price_nonres'] = buildings.improvement_value*1.0 / buildings.non_residential_sqft
    buildings.sqft_price_nonres[buildings.sqft_price_nonres==np.inf] = 0
    buildings['sqft_price_res'] = buildings.improvement_value*1.25 / (buildings.sqft_per_unit * buildings.residential_units)
    buildings.sqft_price_res[buildings.sqft_price_res==np.inf] = 0
    buildings.fillna(0, inplace=True)
    return buildings

@orca.table()
def households(store):
    df = store['households']
    df.building_id[df.building_id==-1] = np.random.choice(store.buildings.index.values,(df.building_id==-1).sum())
    idx_invalid_building_id = np.in1d(df.building_id,store.buildings.index.values)==False
    df.building_id[idx_invalid_building_id] = np.random.choice(store.buildings.index.values,idx_invalid_building_id.sum())
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

#???sim.table to orca.table, correct?
# these are dummy returns that last until accessibility runs
@orca.table()
def nodes():
    return pd.DataFrame()

#???sim.table to orca.table, correct?
@orca.table()
def nodes_prices():
    return pd.DataFrame()
    
# GQ placeholders
for gq_tbl in ['tazcounts2040gq', 'tazcounts2015gq', 'tazcounts2020gq', 'tazcounts2035gq', 'tazcounts2025gq', 'tazcounts2030gq']:
    empty_df = pd.DataFrame()
    orca.add_table(gq_tbl, empty_df)

# this specifies the relationships between tables
orca.broadcast('nodes', 'buildings', cast_index=True, onto_on='_node_id')
orca.broadcast('nodes', 'parcels', cast_index=True, onto_on='_node_id')
orca.broadcast('nodes_prices', 'buildings', cast_index=True, onto_on='_node_id')
orca.broadcast('parcels', 'buildings', cast_index=True, onto_on='parcel_id')
orca.broadcast('buildings', 'households', cast_index=True,
              onto_on='building_id')
orca.broadcast('buildings', 'jobs', cast_index=True, onto_on='building_id')
orca.broadcast('households', 'persons', cast_index=True, onto_on='household_id')
