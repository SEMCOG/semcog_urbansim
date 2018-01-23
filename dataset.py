import warnings

import numpy as np
import orca
import pandas as pd
from urbansim.utils import misc

import assumptions

warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)


for name in ['persons', 'parcels', 'zones', 'semmcds', 'counties', 'employment_sectors',
             'building_sqft_per_job',
             'annual_relocation_rates_for_households',
             'annual_relocation_rates_for_jobs', 'annual_employment_control_totals',
             'travel_data', 'zoning', 'large_areas', 'building_types', 'land_use_types',
             'workers_labor_participation_rates', 'workers_employment_rates_by_large_area_age',
             'workers_employment_rates_by_large_area',
             'transit_stops', 'crime_rates', 'schools', 'poi',
             'group_quarters', 'group_quarters_control_totals',
             'annual_household_control_totals',
             'events_addition', 'events_deletion', 'refiner_events', 'income_growth_rates']:
    store = orca.get_injectable("store")
    orca.add_table(name, store[name])

orca.add_table("remi_pop_total", pd.read_csv("data/remi_hhpop_bylarge.csv", index_col='large_area_id'))
orca.add_table('target_vacancies', pd.read_csv("data/target_vacancies.csv"))
orca.add_table('demolition_rates', pd.read_csv("data/DEMOLITION_RATES.csv", index_col='city_id'))


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

    df['hu_filter'] = 0
    cites = [3130, 6020, 6040]
    sample = df.b_city_id.isin(cites)
    sample = sample[sample.residential_units > 0]
    sample = sample[~(sample.index.isin(store['households'].building_id))]
    for c in cites:
        df.hu_filter.loc[sample[sample.b_city_id == c].sample(frac=0.9, replace=False).index.values] = 1

    return df


@orca.table(cache=True)
def households(store, buildings):
    df = store['households']
    b = buildings.to_frame(['large_area_id'])
    b = b[b.large_area_id.isin({161.0, 3.0, 5.0, 125.0, 99.0, 115.0, 147.0, 93.0})]
    df.loc[df.building_id == -1, 'building_id'] = np.random.choice(b.index.values,
                                                                   (df.building_id == -1).sum())
    idx_invalid_building_id = np.in1d(df.building_id, b.index.values) == False
    df.loc[idx_invalid_building_id, 'building_id'] = np.random.choice(b.index.values,
                                                                      idx_invalid_building_id.sum())
    df['large_area_id'] = misc.reindex(b.large_area_id, df.building_id)
    return df


@orca.table(cache=True)
def jobs(store, buildings):
    df = store['jobs']
    b = buildings.to_frame(['large_area_id'])
    b = b[b.large_area_id.isin({161.0, 3.0, 5.0, 125.0, 99.0, 115.0, 147.0, 93.0})]
    df.loc[df.building_id == -1, 'building_id'] = np.random.choice(b.index.values,
                                                                   (df.building_id == -1).sum())
    idx_invalid_building_id = np.in1d(df.building_id, b.index.values) == False
    df.loc[idx_invalid_building_id, 'building_id'] = np.random.choice(b.index.values,
                                                                      idx_invalid_building_id.sum())
    df['large_area_id'] = misc.reindex(b.large_area_id, df.building_id)
    return df


@orca.table(cache=True)
def base_job_space(buildings):
    return buildings.jobs_non_home_based.to_frame("base_job_space")


# these are dummy returns that last until accessibility runs
for node_tbl in ['nodes', 'nodes_walk', 'nodes_drv']:
    empty_df = pd.DataFrame()
    orca.add_table(node_tbl, empty_df)


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
