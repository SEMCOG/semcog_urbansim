from .variables_parcel import *
from .variables_access import *
from .variables_demographic import *
from .variables_employment import *
from .variables_zone import *
from .variables_building import *
from .variables_tract import *
# Travel survey behavioral variables now come from the base HDF table
# `travel_survey_bg_vars` (loaded in dataset.py). The former travel_survey_vars
# module recomputed them live from the raw survey CSVs and, being a registered
# orca table, SHADOWED the HDF one while providing only 10 of its 11 columns
# (avg_hh_income was silently lost). Removed -- see git history to regenerate.
