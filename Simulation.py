import orca
import models, utils
from urbansim.utils import misc, networks

orca.run(['build_networks'])

base_tables = ['jobs', 'employment_sectors','annual_relocation_rates_for_jobs',
               'households','persons', 'annual_relocation_rates_for_households',
               'buildings', 'parcels', 'zones', 'cities', 'counties',
                 'target_vacancies', 'building_sqft_per_job',
                 'annual_employment_control_totals',
                 'travel_data', 'zoning', 'large_areas', 'building_types', 'land_use_types',
                 'workers_labor_participation_rates', 'workers_employment_rates_by_large_area_age',
                 'workers_employment_rates_by_large_area',
                 'transit_stops', 'crime_rates', 'schools', 'poi',
                 'annual_household_control_totals', 'scheduled_development_events', 'scheduled_demolition_events']

out_tables = ['buildings', 'jobs', 'parcels', 'households', 'persons']

orca.run([
    "neighborhood_vars",  # worked
    "households_transition",  # worked
    "fix_lpr",  # worked
    "households_relocation",  # worked
    "jobs_transition",  # worked
    "jobs_relocation",  # worked
    "scheduled_development_events",  # worked
    "scheduled_demolition_events",  # worked
    "price_vars",  # worked # compute average price measures
    "feasibility",  # fix in other branch
    "residential_developer",  # fix in other branch
    "non_residential_developer",  # fix in other branch
    "nrh_simulate",  # worked # non-residential rent hedonic
    "rsh_simulate",  # worked # residential sales hedonic
    "hlcm_simulate",  # worked # households location choice
    "elcm_simulate",  # worked # employment location choice
    "government_jobs_scaling_model",  # worked
    "refiner",
    # "gq_model", Fixme: we have new data so need new approach
    # "travel_model", Fixme: on hold
    # "housing_value_update", Fixme: maybe we don't need
], iter_vars=range(2016, 2045 + 1),
    data_out=utils.get_run_filename(),
    out_base_tables=base_tables, #TODO: check pass None to get all tables for base  year
    out_run_tables=out_tables,
    out_interval=1)
# TODO: remove `liv_index`, `price_adj` coll
# TODO: check target_vacancies used or not and how
