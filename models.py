import pandas as pd
import time
from urbansim.models import transition
from urbansim.models.yamlmodelrunner import *
from variables import var_calc


# residential sales hedonic
def rsh_estimate(dset):
    return hedonic_estimate(dset.buildings_computed, "rsh.yaml")


def rsh_simulate(dset):
    return hedonic_simulate(dset.buildings_computed, "rsh.yaml", dset.buildings, "unit_price_res")


# non-residential hedonic
def nrh_estimate(dset):
    return hedonic_estimate(dset.buildings_computed, "nrh.yaml")


def nrh_simulate(dset):
    return hedonic_simulate(dset.buildings_computed, "nrh.yaml", dset.buildings, "unit_price_nonres")


# household location choice
def hlcm_estimate(dset):
    return lcm_estimate(dset.households, "building_id", dset.buildings_computed, "hlcm.yaml")


def hlcm_simulate(dset):
    units = get_vacant_units(dset.households, "building_id", dset.buildings_computed, "residential_units")
    return lcm_simulate(dset.households, units, "hlcm.yaml", dset.households, "building_id")


# employment location choice
def elcm_estimate(dset):
    return lcm_estimate(dset.jobs, "building_id", dset.buildings_computed, "elcm.yaml")


def elcm_simulate(dset):
    units = get_vacant_units(dset.jobs, "building_id", dset.buildings_computed, "job_spaces")
    units = units.loc[np.random.choice(units.index, size=200000, replace=False)]
    return lcm_simulate(dset.jobs, units, "elcm.yaml", dset.jobs, "building_id")


def households_relocation(dset):
    return simple_relocation(dset.households, .01)


def jobs_relocation(dset):
    return simple_relocation(dset.jobs, .02)


def households_transition(dset):
    ct = dset.fetch('annual_household_control_totals')
    totals_field = ct.reset_index().groupby('year').total_number_of_households.sum()
    ct = pd.DataFrame({'total_number_of_households': totals_field})
    tran = transition.TabularTotalsTransition(ct, 'total_number_of_households')
    model = transition.TransitionModel(tran)
    new, added_hh_idx, new_linked = \
        model.transition(dset.households, dset.year,
                         linked_tables={'linked': (dset.persons, 'household_id')})
    new.loc[added_hh_idx, "building_id"] = np.nan
    dset.save_tmptbl("households", new)
    dset.save_tmptbl("persons", new_linked['linked'])


def jobs_transition(dset):
    ct_emp = dset.fetch('annual_employment_control_totals')
    totals_field = ct_emp.reset_index().groupby('year').total_number_of_jobs.sum()
    ct_emp = pd.DataFrame({'total_number_of_jobs': totals_field})
    tran = transition.TabularTotalsTransition(ct_emp, 'total_number_of_jobs')
    model = transition.TransitionModel(tran)
    new, added_jobs_idx, new_linked = model.transition(dset.jobs, dset.year)
    new.loc[added_jobs_idx, "building_id"] = np.nan
    dset.save_tmptbl("jobs", new)


def _run_models(dset, model_list, years):
    for year in years:

        dset.year = year

        t1 = time.time()

        var_calc.calculate(dset)

        for model in model_list:
            t2 = time.time()
            print "\n" + model + "\n"
            globals()[model](dset)
            print "Model %s executed in %.3fs" % (model, time.time()-t2)
        print "Year %d completed in %.3fs" % (year, time.time()-t1)