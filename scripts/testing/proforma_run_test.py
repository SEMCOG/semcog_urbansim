import orca
import pandas as pd
from urbansim.utils import misc, networks

import dataset
import variables
import models
import utils

orca.run(['build_networks'])

orca.run([
    "neighborhood_vars", ] +  # neighborhood variables
    orca.get_injectable('repm_step_names') +  # In place of ['nrh_simulate', 'rsh_simulate']
    ["increase_property_values",  # Hack to make more feasibility
    "feasibility",  # compute development feasibility
    "residential_developer",  # build actual buildings
    "non_residential_developer"
], iter_vars=list(range(2016, 2017)),
    out_interval=1)
