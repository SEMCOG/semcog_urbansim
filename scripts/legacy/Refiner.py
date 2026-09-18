import orca
import shutil

import os

import pandas as pd

import models, utils
from urbansim.utils import misc, networks

orca.add_table('refiner_events', pd.read_csv("data/add_pop_11032017.csv"))

data_out = utils.get_run_filename()
print(data_out)

orca.run([
    "refiner",
],
    iter_vars=list(range(2015, 2015 + 1)),
    data_out=data_out,
    out_base_tables=['jobs', 'base_job_space', 'employment_sectors', 'annual_relocation_rates_for_jobs',
                     'households', 'persons', 'annual_relocation_rates_for_households',
                     'buildings', 'parcels', 'zones', 'semmcds', 'counties',
                     'target_vacancies', 'building_sqft_per_job',
                     'annual_employment_control_totals',
                     'travel_data', 'zoning', 'large_areas', 'building_types', 'land_use_types',
                     'workers_labor_participation_rates', 'workers_employment_rates_by_large_area_age',
                     'workers_employment_rates_by_large_area',
                     'transit_stops', 'crime_rates', 'schools', 'poi',
                     'group_quarters', 'group_quarters_control_totals',
                     'annual_household_control_totals',
                     'events_addition', 'events_deletion', 'refiner_events'],
    out_run_tables=['buildings', 'jobs', 'base_job_space', 'parcels', 'households', 'persons', 'group_quarters'],
    out_interval=1,
    compress=True)
