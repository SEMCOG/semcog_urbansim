import orca
import pandas as pd
from urbansim.utils import misc, networks

import dataset
import variables
import models
import utils

orca.run(['build_networks'])

orca.run([
    "neighborhood_vars",  # neighborhood variables
    "price_vars",  # compute average price measures
    # "feasibility",  # compute development feasibility
    "new_feasibility",
    # "residential_developer",  # build actual buildings
    "new_res_developer"
], iter_vars=range(2016, 2017),
    data_out=utils.get_run_filename(),
    out_interval=1)
