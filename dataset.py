import warnings

import numpy as np
import orca
import pandas as pd

import assumptions

warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)


@orca.table(cache=True)
def buildings(store):
    df = store['buildings']
    # Todo: combine two sqft prices into one and set non use sqft price to 0
    df.loc[df.improvement_value < 0, 'improvement_value'] = 0
    df['sqft_price_nonres'] = df.improvement_value * 1.0 / 0.7 / df.non_residential_sqft
    df.loc[df.sqft_price_nonres > 1000, 'sqft_price_nonres'] = 0
    df.loc[df.sqft_price_nonres < 0, 'sqft_price_nonres'] = 0
    df['sqft_price_res'] = df.improvement_value * 1.25 / 0.7 / (df.sqft_per_unit.astype(int) * df.residential_units)
    df.loc[df.sqft_price_res > 1000, 'sqft_price_res'] = 0
    df.loc[df.sqft_price_res < 0, 'sqft_price_res'] = 0
    df.fillna(0, inplace=True)
    orca.add_injectable("max_building_id", 10000000)
    return df


@orca.table(cache=True)
def households(store):
    df = store['households']
    df.loc[df.building_id == -1, 'building_id'] = np.random.choice(store.buildings.index.values,
                                                                   (df.building_id == -1).sum())
    idx_invalid_building_id = np.in1d(df.building_id, store.buildings.index.values) == False
    df.loc[idx_invalid_building_id, 'building_id'] = np.random.choice(store.buildings.index.values,
                                                                      idx_invalid_building_id.sum())
    return df


for name in ['jobs', 'persons', 'parcels', 'zones', 'semmcds', 'counties', 'employment_sectors',
             'building_sqft_per_job',
             'annual_relocation_rates_for_households',
             'annual_relocation_rates_for_jobs', 'annual_employment_control_totals',
             'travel_data', 'zoning', 'large_areas', 'building_types', 'land_use_types',
             'workers_labor_participation_rates', 'workers_employment_rates_by_large_area_age',
             'workers_employment_rates_by_large_area',
             'transit_stops', 'crime_rates', 'schools', 'poi',
             'annual_household_control_totals',
             'events_addition', 'events_deletion', 'refiner_events', 'income_growth_rates']:
    store = orca.get_injectable("store")
    orca.add_table(name, store[name])

orca.add_table("remi_pop_total", pd.read_csv("data/remi_hhpop_bylarge.csv", index_col='large_area_id'))
orca.add_table('target_vacancies', pd.read_csv("data/target_vacancies.csv"))


@orca.injectable(cache=True)
def base_job_space(buildings):
    df = buildings.jobs_non_home_based
    return df[df > 0]


# these are dummy returns that last until accessibility runs
for node_tbl in ['nodes', 'nodes_walk', 'nodes_drv']:
    empty_df = pd.DataFrame()
    orca.add_table(node_tbl, empty_df)

# GQ placeholders
# for gq_tbl in ['tazcounts2040gq', 'tazcounts2015gq', 'tazcounts2020gq', 'tazcounts2035gq', 'tazcounts2025gq',
#                'tazcounts2030gq']:
#     empty_df = pd.DataFrame()
#     orca.add_table(gq_tbl, empty_df)

# this specifies the relationships between tables
orca.broadcast('nodes_walk', 'buildings', cast_index=True, onto_on='nodeid_walk')
orca.broadcast('nodes_walk', 'parcels', cast_index=True, onto_on='nodeid_walk')
orca.broadcast('nodes_drv', 'buildings', cast_index=True, onto_on='nodeid_drv')
orca.broadcast('nodes_drv', 'parcels', cast_index=True, onto_on='nodeid_drv')
orca.broadcast('parcels', 'buildings', cast_index=True, onto_on='parcel_id')
orca.broadcast('buildings', 'households', cast_index=True, onto_on='building_id')
orca.broadcast('buildings', 'jobs', cast_index=True, onto_on='building_id')
orca.broadcast('households', 'persons', cast_index=True, onto_on='household_id')
orca.broadcast('building_types', 'buildings', cast_index=True, onto_on='building_type_id')
orca.broadcast('zones', 'parcels', cast_index=True, onto_on='zone_id')
orca.broadcast('schools', 'parcels', cast_on='parcel_id', onto_index=True)
