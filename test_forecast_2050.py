import orca
import shutil
import sys
import os
import utils

# get run number and set up log file
data_out = utils.get_run_filename()
orca.add_injectable("data_out_dir", data_out.replace(".h5", ""))
print(data_out)

import models
from urbansim.utils import misc, networks
import time
import output_indicators
import logging

# set up run
base_year = 2020
final_year = 2050
indicator_spacing = 5
upload_to_carto = True
run_debug = False
add_2019 = True

# check disk space, need at least 16GB
# total, used, free = [round(s / (2 ** 30), 1) for s in shutil.disk_usage(".")]
# print(f"Disk space: {total} GB;   Used: {used} GB;   Free: {free} GB")
# if free < 17:
#     print(f"Free space is too small. Only {free} GB available. Stop running")
#     sys.exit()

start_time = time.time()

run_info = f"""data_out: {data_out} \
            \nRun number: {os.path.basename(data_out.replace('.h5', ''))} \
            \nStart time: {time.ctime(start_time)}"""
utils.run_log(run_info)

if run_debug is True:
    utils.debug_log()

orca.run(
    [
        "build_networks_2050",
        "neighborhood_vars",
        "scheduled_demolition_events",
        "random_demolition_events",
        "scheduled_development_events",
        "refiner",
        "households_transition",
        "fix_lpr",
        "households_relocation_2050",
        "jobs_transition",
        "jobs_relocation_2050",
        "drop_pseudo_buildings",
        "feasibility",
        "residential_developer",
        "non_residential_developer",
        "update_sp_filter",
    ]
    + orca.get_injectable("repm_step_names")
    # + ["increase_property_values"]  # on hold
    + ["refine_housing_units"]
    + ["mcd_hu_sampling"]
    + orca.get_injectable("hlcm_step_names")
    + orca.get_injectable("elcm_step_names")
    + [
        "elcm_home_based",
        "jobs_scaling_model",
        "gq_pop_scaling_model",
        # "travel_model", #Fixme: on hold
        "update_bg_hh_increase",
    ],
    iter_vars=list(range(base_year + 1, final_year + 1)),
    data_out=data_out,
    out_base_tables=[
        "jobs",
        "jobs_2019",
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
        "employed_workers_rate",
        "transit_stops",
        "crime_rates",
        "schools",
        "poi",
        "group_quarters",
        "group_quarters_households",
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
        "hu_sampling_bg_summary",
    ],
    out_interval=1,
    compress=True,
)

output_indicators.main(
    data_out,
    base_year,
    final_year,
    spacing=indicator_spacing,
    upload_to_carto=upload_to_carto,
    add_2019=add_2019,
)

utils.run_log(
    f"Total run time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}"
)

print("Simulation started at %s, finished at %s. " % (start_time, time.ctime()))

# dir_out = data_out.replace('.h5', '')
# shutil.copytree(dir_out, '/mnt/hgfs/U/RDF2045/model_runs/' + os.path.basename(os.path.normpath(dir_out)))
# shutil.copy(data_out, '/mnt/hgfs/J')
