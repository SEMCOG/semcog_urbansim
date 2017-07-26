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
    "nrh_simulate",  # non-residential rent hedonic
    "rsh_simulate",  # residential sales hedonic
    "feasibility",  # compute development feasibility
    "residential_developer",  # build actual buildings
    "non_residential_developer"
], iter_vars=range(2016, 2017),
    out_interval=1)
