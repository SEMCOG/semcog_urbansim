#!/usr/bin/env python

import orca
import shutil
import os

import pandas as pd
import models, utils
from urbansim.utils import misc, networks
import variables

data_out = os.path.join(misc.runs_dir(), "test_city_model.h5") 

orca.run([
    # "neighborhood_vars",
    # "scheduled_demolition_events",
    # "random_demolition_events",
    # "scheduled_development_events",
    # "refiner",       
    "households_transition",
    # "fix_lpr",
    "households_relocation",
    # "jobs_transition",
    # "jobs_relocation",
    # "feasibility",
    # "residential_developer",
    # "non_residential_developer"
    ] + 
    # orca.get_injectable('repm_step_names') +  # In place of ['nrh_simulate', 'rsh_simulate']
    # ["increase_property_values"] +  # Hack to make more feasibility
    ['mcd_hu_sampling'] +
    orca.get_injectable('hlcm_step_names') +
    # orca.get_injectable('elcm_step_names') + 
    [
        # "elcm_home_based",
        # "jobs_scaling_model",
        # "gq_pop_scaling_model",
        'update_bg_hh_increase',
    ],
    iter_vars=list(range(2020, 2025)),
    data_out=data_out,
    out_base_tables=['households', 'semmcds'],
    out_run_tables=['households'],
    out_interval=1,
    compress=True)