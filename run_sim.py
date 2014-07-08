import models
import dataset

dset = dataset.SemcogDataset("data/semcog_data.h5")
models.build_networks(dset)

YEARS = range(2012, 2021)
MODEL_LIST = [
    "clear_cache",            # recompute variables every year
    "neighborhood_vars",      # neighborhood variables
    "households_transition",  # households transition
    "households_relocation",  # households relocation model
    "jobs_transition",        # jobs transition
    "jobs_relocation",        # jobs relocation model
#     "cache_variables",        # this is the variable computation time
    "scheduled_development_events",
    "nrh_simulate",           # non-residential rent hedonic
    "rsh_simulate",           # residential sales hedonic
    "hlcm_simulate",          # households location choice
    "elcm_simulate",          # employment location choice
    "price_vars",             # compute average price measures
    "feasibility",            # compute development feasibility
    "residential_developer",   # build actual buildings
    "non_residential_developer",   # build actual buildings
    "government_jobs_scaling_model",
    "refiner",
    "aging_model",
    "income_inflation_model",
]
#dset.debug = True
models._run_models(dset, MODEL_LIST, YEARS)