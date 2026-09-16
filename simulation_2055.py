import orca
import shutil
import sys
import os
import pandas as pd
import utils
import input_paths

# ===============
## Config
# ===============
# get run number and set up log file
data_out = utils.get_run_filename()
orca.add_injectable("data_out_dir", data_out.replace(".h5", ""))
print(data_out)
# run config
RUN_OUTPUT_INDICATORS = True
base_year = 2025
final_year = 2055
indicator_spacing = 5
upload_to_carto = False
run_debug = False
# model paths (all external inputs are centralized in input_paths.py)
orca.add_injectable('hlcm_model_path', input_paths.HLCM_MODEL_DIR)
orca.add_injectable('elcm_model_path', input_paths.ELCM_MODEL_DIR)
orca.add_injectable('xgb_repm_dir', input_paths.REPM_MODEL_DIR)
orca.add_injectable('yaml_configs', 'yaml_configs_elcm_hlcm.yaml')
# base/final years
orca.add_injectable('base_year', base_year)
orca.add_injectable('final_year', final_year)
# scenario controls
orca.add_injectable('ENABLE_SCENARIO', False)
orca.add_injectable('scenario_hh_control_path', input_paths.SCENARIO_HH_CONTROL_CSV)
orca.add_injectable('scenario_remi_total_pop', input_paths.SCENARIO_REMI_POP_CSV)
orca.add_injectable('scenario_emp_control_path', input_paths.SCENARIO_EMP_CONTROL_CSV)
# household-pop target: inputs provide remi_hh_pop, so no fallback to the legacy
# total-pop table (households_transition raises if remi_hh_pop is absent)
orca.add_injectable('allow_total_pop_fallback', False)
# households matching no control category: warn (False) or raise (True)
orca.add_injectable('require_full_control_coverage', False)
# run-level random seed; None draws a fresh one (logged in run_config.yaml)
RANDOM_SEED = 271828
if RANDOM_SEED is None:
    import numpy as _np
    RANDOM_SEED = int(_np.random.SeedSequence().entropy & 0xFFFFFFFF)
orca.add_injectable('random_seed', RANDOM_SEED)
print('using random_seed', RANDOM_SEED)
# Checkpoint config
# run starting from last checkpoint year
orca.add_injectable('use_checkpoint', False)
orca.add_injectable('runnum_to_resume', 'run1365.h5')
# run_config.yaml and copies of key model configs in run folder
utils.write_run_metadata(
    data_out,
    input_paths.BASE_HDF,
    {
        "RUN_OUTPUT_INDICATORS": RUN_OUTPUT_INDICATORS,
        "indicator_spacing": indicator_spacing,
        "upload_to_carto": upload_to_carto,
        "run_debug": run_debug,
    },
)

# =================
## Initialize Model
# =================
import models
from urbansim.utils import misc, networks
import time
import logging
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    _eastern = ZoneInfo("America/Detroit")
    def _eastern_now():
        return datetime.now(_eastern).strftime("%Y-%m-%d %H:%M:%S %Z")
except ImportError:
    import pytz
    _eastern = pytz.timezone("America/Detroit")
    def _eastern_now():
        return datetime.now(_eastern).strftime("%Y-%m-%d %H:%M:%S %Z")
start_time = time.time()
run_info = f"""data_out: {data_out} \
            \nRun number: {os.path.basename(data_out.replace('.h5', ''))} \
            \nStart time: {_eastern_now()}"""
utils.run_log(run_info)
if run_debug is True:
    utils.debug_log()
run_start = base_year if not orca.get_injectable('use_checkpoint') else orca.get_injectable('checkpoint_year')

# =============
## Simulation
# =============
# run init_taz_hlcm_trend_by_year on baseyear
orca.run([
    'init_taz_hlcm_trend_by_year',
])
# Start iteration
orca.run(
    [
        "clear_iteration_cache",  # drop last year's memoized derived cols
        "build_networks",
        "neighborhood_vars",
        "update_taz_hlcm_trend",
        "log_memory",  # after networks + accessibility
        "cache_hh_seeds", # only run on first year
        "scheduled_demolition_events",
        "scored_demolition_events",
        "scheduled_development_events",
        "refiner",
        "households_transition",
        "workers_adjustment_model",
        "households_relocation",
        "jobs_transition",
        # "jobs_relocation",
        "log_memory",  # after transition/relocation
        "feasibility",
        "residential_developer",
        "non_residential_developer",
        "update_sp_filter",
        "log_memory",  # after developer
    ]
    + orca.get_injectable("repm_step_names")
    + ["real_estate_adjustment"]
    # + ["refine_housing_units"]
    # + ["mcd_hu_sampling"]
    + ["log_memory"]  # after REPM + housing-unit refine
    + orca.get_injectable("hlcm_step_names")
    + orca.get_injectable("elcm_step_names")
    + [
        # "elcm_home_based", # disable elcm_home_based due the the new NN based elcm
        "log_memory",  # after HLCM + ELCM
        "jobs_scaling_model",
        "seed_new_gq_buildings",  # must run immediately before gq_pop_scaling_model
        "gq_pop_scaling_model",
        # "travel_model", #Fixme: on hold
        "update_bg_hh_increase",
    ],
    iter_vars=list(range(run_start + 1, final_year + 1)),
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
        "parcels",
        "zones",
        "semmcds",
        "counties",
        "target_vacancies_mcd",
        "target_vacancies",
        "building_sqft_per_job",
        "annual_employment_control_totals",
        "travel_data",
        "zoning",
        "large_areas",
        "building_types",
        "land_use_types",
        "employed_workers_rate",
        "transit_stops",
        "crime_rates",
        "schools",
        "points_of_interest_by_category",
        "group_quarters",
        "group_quarters_households",
        "group_quarters_control_totals",
        "annual_household_control_totals",
        "events_addition",
        "events_deletion",
        "refiner_events",
    ]
    # snapshot whichever HH-population target table is present (remi_pop_total
    # only exists in scenario runs)
    + [t for t in ("remi_hh_pop", "remi_pop_total") if orca.is_table(t)],
    out_run_tables=[
        "buildings",
        "jobs",
        "base_job_space",
        "parcels",
        "households",
        "persons",
        "group_quarters",
        "dropped_buildings",
        "bg_hh_increase",
    ],
    out_interval=1,
    compress=True,
)

# if use checkpoint to resume run, add result from previous year back
if orca.get_injectable('use_checkpoint'):
    store_la = pd.HDFStore(data_out, mode="a")
    run_path = "/mnt/semcog_urbansim/runs"
    hdf_path = os.path.join(run_path, orca.get_injectable('runnum_to_resume'))
    old_result = pd.HDFStore(hdf_path, "r")
    for k in old_result:
        if '/base/' in k or k in store_la.keys():
            continue
        print('adding %s to output hdf from checkpoint...' % k)
        store_la[k] = old_result[k]
    old_result.close()
    store_la.close()

# load late because of introduce of new vars
import output_indicators
if RUN_OUTPUT_INDICATORS:
    # set up run
    import output_indicators
    output_indicators.main(
        data_out,
        base_year,
        final_year,
        spacing=indicator_spacing,
        upload_to_carto=upload_to_carto,
    )

utils.run_log(
    f"Total run time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}"
)

print("Simulation finished at %s. Total run time: %s" % (_eastern_now(), time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))))

# dir_out = data_out.replace('.h5', '')
# shutil.copytree(dir_out, '/mnt/hgfs/U/RDF2045/model_runs/' + os.path.basename(os.path.normpath(dir_out)))
# shutil.copy(data_out, '/mnt/hgfs/J')
