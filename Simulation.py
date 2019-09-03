import orca
import shutil

import os

import models, utils
from urbansim.utils import misc, networks
import output_indicators

data_out = utils.get_run_filename()
print data_out

import sys,statvfs

f = os.statvfs("/home")
freespace=f[statvfs.F_BAVAIL] * f[statvfs.F_BSIZE] / (1048576 * 1024.0)
if freespace < 10:
    print freespace, 'GB available. Disk space is too small, stop running'
    sys.exit()


orca.run(["refiner",
          'build_networks',
          "neighborhood_vars"] +
          orca.get_injectable('repm_step_names') + # In place of ['nrh_simulate', 'rsh_simulate']
          ["increase_property_values"])  # Hack to make more feasibility

orca.run([
    "neighborhood_vars",
    "scheduled_demolition_events",
    "random_demolition_events",
    "scheduled_development_events",
    "refiner",       
    "households_transition",
    "fix_lpr",
    "households_relocation",
    "jobs_transition",
    "jobs_relocation",
    "feasibility",
    "residential_developer",
    "non_residential_developer"] +
    orca.get_injectable('repm_step_names') +  # In place of ['nrh_simulate', 'rsh_simulate']
    ["increase_property_values"] +  # Hack to make more feasibility
    orca.get_injectable('hlcm_step_names_choosermover') +
    orca.get_injectable('elcm_step_names_lacontrol_unitlevel') +
    ["elcm_home_based",
    "jobs_scaling_model",
    "gq_pop_scaling_model",
    # "travel_model", Fixme: on hold
],
    iter_vars=range(2016, 2045 + 1),
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
    out_run_tables=['buildings', 'jobs', 'base_job_space', 'parcels', 'households', 'persons', 'group_quarters', 'dropped_buildings'],
    out_interval=1,
    compress=True)

output_indicators.main(data_out)

dir_out = data_out.replace('.h5', '')
shutil.copytree(dir_out, '/mnt/hgfs/U/RDF2045/model_runs/' + os.path.basename(os.path.normpath(dir_out)))
shutil.copy(data_out, '/mnt/hgfs/J')
