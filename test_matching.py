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
final_year = 2022
indicator_spacing = 5
upload_to_carto = False
run_debug = False

# check disk space, need at least 16GB
total, used, free = [round(s / (2 ** 30), 1) for s in shutil.disk_usage(".")]
print(f"Disk space: {total} GB;   Used: {used} GB;   Free: {free} GB")
if free < 17:
    print(f"Free space is too small. Only {free} GB available. Stop running")
    sys.exit()

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
        "households_transition",
        "households_relocation_2050",
    ]
    # + ["increase_property_values"]  # on hold
    + ["mcd_hu_sampling"]
    + orca.get_injectable("hlcm_step_names")
    + [
    ],
    iter_vars=list(range(base_year + 1, final_year + 1)),
    data_out=data_out,
    out_base_tables=[
        "jobs",
        "households",
        "persons",
        "buildings",
        "parcels",
        "zones",
    ],
    out_run_tables=[
        "buildings",
        "jobs",
        "parcels",
        "households",
        "persons",
    ],
    out_interval=1,
    compress=True,
)
