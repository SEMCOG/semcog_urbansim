import pandas as pd
import numpy as np
import time
from urbansim.models import transition
import urbansim.models.yamlmodelrunner as ymr
from variables import var_calc
from urbansim.developer import sqftproforma, developer
from urbansim.utils import misc


# residential sales hedonic
def rsh_estimate(dset):
    return ymr.hedonic_estimate(dset.buildings_computed, "rsh.yaml")


def rsh_simulate(dset):
    return ymr.hedonic_simulate(dset.buildings_computed, "rsh.yaml", dset.buildings, "unit_price_res")


# non-residential hedonic
def nrh_estimate(dset):
    return ymr.hedonic_estimate(dset.buildings_computed, "nrh.yaml")


def nrh_simulate(dset):
    return ymr.hedonic_simulate(dset.buildings_computed, "nrh.yaml", dset.buildings, "unit_price_nonres")


# household location choice
def hlcm_estimate(dset):
    return ymr.lcm_estimate(dset.households, "building_id", dset.buildings_computed, "hlcm.yaml")


def hlcm_simulate(dset):
    units = ymr.get_vacant_units(dset.households, "building_id", dset.buildings_computed, "residential_units")
    return ymr.lcm_simulate(dset.households, units, "hlcm.yaml", dset.households, "building_id")


# employment location choice
def elcm_estimate(dset):
    return ymr.lcm_estimate(dset.jobs, "building_id", dset.buildings_computed, "elcm.yaml")


def elcm_simulate(dset):
    units = ymr.get_vacant_units(dset.jobs, "building_id", dset.buildings_computed, "job_spaces")
    units = units.loc[np.random.choice(units.index, size=200000, replace=False)]
    return ymr.lcm_simulate(dset.jobs, units, "elcm.yaml", dset.jobs, "building_id")


def households_relocation(dset):
    return ymr.simple_relocation(dset.households, .01)


def jobs_relocation(dset):
    return ymr.simple_relocation(dset.jobs, .02)


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


def feasibility(dset):
    pf = sqftproforma.SqFtProForma()

    buildings = dset.buildings_computed
    average_residential_rent = buildings.query("residential_units > 0")\
        .groupby('zone_id').unit_price_res.quantile() * pf.config.cap_rate
    average_non_residential_rent = buildings.query("non_residential_sqft > 0")\
        .groupby('zone_id').unit_price_nonres.quantile() * pf.config.cap_rate

    parcels = dset.parcels
    df = pd.DataFrame(index=parcels.index)
    df["residential"] = misc.reindex(average_residential_rent, parcels.zone_id)
    df["retail"] = df["office"] = df["industrial"] = \
        misc.reindex(average_non_residential_rent, parcels.zone_id)

    df['total_sqft'] = buildings.groupby('parcel_id').building_sqft.sum()
    df['land_cost'] = (df.total_sqft * df.residential).fillna(0) / pf.config.cap_rate
    df['parcel_size'] = parcels.parcel_sqft
    df["max_far"] = 1.8
    df["max_height"] = 20
    df = df.query("total_sqft < 1500")
    print df.describe()

    d = {}
    for form in pf.config.forms:
        print "Computing feasibility for form %s" % form
        d[form] = pf.lookup(form, df)

    far_predictions = pd.concat(d.values(), keys=d.keys(), axis=1)

    dset.save_tmptbl("feasibility", far_predictions)


def residential_developer(dset):
    dev = developer.Developer(dset.feasibility)

    res_target_vacancy = .1
    res_target_units = \
        dev.compute_units_to_build(len(dset.households.index),
                                   dset.buildings_computed.residential_units.sum(),
                                   res_target_vacancy)

    min_new_unit_size = 650
    ave_unit_size = dset.buildings_computed.groupby('zone_id').sqft_per_unit.quantile()
    ave_unit_size[ave_unit_size < min_new_unit_size] = min_new_unit_size
    ave_unit_size = misc.reindex(ave_unit_size, dset.parcels.index)

    new_buildings = dev.pick_build(res_target_units,
                                   dset.parcels.parcel_sqft,
                                   ave_unit_size,
                                   dset.buildings_computed.groupby('parcel_id').residential_units.sum())
    print new_buildings.head()
    print new_buildings.columns
    print new_buildings.describe()
    """



    'building_id_old',
'building_type_id', \
'improvement_value', \
'land_area', \
'non_residential_sqft', \
'parcel_id', \
'residential_units', \
'sqft_per_unit', \
'stories', \
'tax_exempt', \
'year_built'
    """


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