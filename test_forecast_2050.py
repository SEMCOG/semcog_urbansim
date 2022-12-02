import orca
import shutil

import os
import models, utils
from urbansim.utils import misc, networks
import time
import output_indicators

data_out = utils.get_run_filename()
print(data_out)

import sys

f = os.statvfs("/home")
freespace = f.f_bavail * f.f_bsize / (1048576 * 1024.0)
print("freespace:", freespace)
if freespace < 10:
    print(freespace, "GB available. Disk space is too small, stop running")
    sys.exit()

start_time = time.ctime()

# orca.run(["households_transition"], iter_vars=list(range(2020, 2025)))
orca.run(
    ["refiner", "build_networks_2050", "neighborhood_vars"]
    + orca.get_injectable("repm_step_names")
    + ["increase_property_values"]  # In place of ['nrh_simulate', 'rsh_simulate']
)  # increase feasibility based on projected income

orca.run(
    [
        "build_networks_2050",
        "neighborhood_vars",
        "scheduled_demolition_events",
        "random_demolition_events",
        "scheduled_development_events",
        "refiner",
        "households_transition",
        "fix_lpr", # await data
        "households_relocation_2050",
        "jobs_transition",
        "jobs_relocation_2050",
        "feasibility",
        "residential_developer",
        "non_residential_developer",
        "update_sp_filter",
    ]
    + orca.get_injectable("repm_step_names")
    + ["increase_property_values"]  # In place of ['nrh_simulate', 'rsh_simulate']
    + ["mcd_hu_sampling"]  # Hack to make more feasibility
    + orca.get_injectable("hlcm_step_names") # disable for now, wait until new estimation
    + orca.get_injectable("elcm_step_names")
    + [
        "elcm_home_based",
        "jobs_scaling_model",
        "gq_pop_scaling_model",
        # "travel_model", #Fixme: on hold
        "update_bg_hh_increase",
    ],
    iter_vars=list(range(2020, 2051)),
    data_out=data_out,
    out_base_tables=[
        "jobs",
        "base_job_space",
        "employment_sectors",
        "annual_relocation_rates_for_jobs",
        "households",
        "persons",
        "annual_relocation_rates_for_households",
        "buildings",
        "pseudo_building_2020",
        "parcels",
        "zones",
        "semmcds",
        "counties",
        "target_vacancies_mcd",
        "target_vacancies",
        "building_sqft_per_job",
        "annual_employment_control_totals",
        "travel_data",
        "travel_data_2030",
        "zoning",
        "large_areas",
        "building_types",
        "land_use_types",
        # "workers_labor_participation_rates",
        "employed_workers_rate",
        "transit_stops",
        "crime_rates",
        "schools",
        "poi",
        "group_quarters",
        "group_quarters_control_totals",
        "annual_household_control_totals",
        "remi_pop_total",
        "events_addition",
        "events_deletion",
        "refiner_events",
    ],
    out_run_tables=[
        "buildings",
        "jobs",
        "base_job_space",
        "parcels",
        "households",
        "persons",
        "group_quarters",
        "dropped_buildings",
    ],
    out_interval=1,
    compress=True,
)

output_indicators.main(data_out)
print("Simulation started at %s, finished at %s. " % (start_time, time.ctime()))

# dir_out = data_out.replace('.h5', '')
# shutil.copytree(dir_out, '/mnt/hgfs/U/RDF2045/model_runs/' + os.path.basename(os.path.normpath(dir_out)))
# shutil.copy(data_out, '/mnt/hgfs/J')
