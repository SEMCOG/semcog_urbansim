import orca
import models, utils
from urbansim.utils import misc, networks

orca.run(['build_networks'])

orca.run([
    "neighborhood_vars",
    "households_transition",
    "fix_lpr",
    "households_relocation",
    "jobs_transition",
    "jobs_relocation",
    "scheduled_demolition_events",
    "scheduled_development_events",
    "feasibility",
    "residential_developer",
    "non_residential_developer",
    "nrh_simulate",  # non-residential rent hedonic
    "rsh_simulate",  # residential sales hedonic
    "hlcm_simulate",  # households location choice
    "elcm_simulate",  # employment location choice
    "government_jobs_scaling_model",
    "refiner",
    # "gq_model", Fixme: we have new data so need new approach
    # "travel_model", Fixme: on hold
    # "housing_value_update", Fixme: maybe we don't need
], iter_vars=range(2016, 2045 + 1),
    data_out=utils.get_run_filename(),
    out_base_tables=['jobs', 'employment_sectors', 'annual_relocation_rates_for_jobs',
                     'households', 'persons', 'annual_relocation_rates_for_households',
                     'buildings', 'parcels', 'zones', 'semmcds', 'counties',
                     'target_vacancies', 'building_sqft_per_job',
                     'annual_employment_control_totals',
                     'travel_data', 'zoning', 'large_areas', 'building_types', 'land_use_types',
                     'workers_labor_participation_rates', 'workers_employment_rates_by_large_area_age',
                     'workers_employment_rates_by_large_area',
                     'transit_stops', 'crime_rates', 'schools', 'poi',
                     'annual_household_control_totals',
                     'events_addition', 'events_deletion', 'refiner_events'],
    out_run_tables=['buildings', 'jobs', 'parcels', 'households', 'persons', 'dropped_buildings'],
    out_interval=1)
# TODO: check target_vacancies used or not and how
