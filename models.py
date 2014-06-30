import pandas as pd
import numpy as np
import time
import os
from urbansim.models import transition
import urbansim.models.yamlmodelrunner as ymr
from urbansim.developer import sqftproforma, developer
from urbansim.utils import misc, networks


def buildings_df(dset, addnodes=True):
    buildings = dset.view("buildings").build_df().fillna(0)
    if addnodes:
        buildings = dset.merge_nodes(buildings)
    return buildings


def households_df(dset):
    return dset.view("households").build_df()


def jobs_df(dset):
    return dset.view("jobs").build_df()


def clear_cache(dset):
    dset.clear_views()


def cache_variables(dset):
    buildings_df(dset, addnodes=False)
    households_df(dset)
    jobs_df(dset)


# residential sales hedonic
def rsh_estimate(dset):
    return ymr.hedonic_estimate(buildings_df(dset), "rsh.yaml")


def rsh_simulate(dset):
    return ymr.hedonic_simulate(buildings_df(dset), "rsh.yaml", dset.buildings, "sqft_price_res")


# non-residential hedonic
def nrh_estimate(dset):
    return ymr.hedonic_estimate(buildings_df(dset), "nrh.yaml")


def nrh_simulate(dset):
    return ymr.hedonic_simulate(buildings_df(dset), "nrh.yaml", dset.buildings, "sqft_price_nonres")


# household location choice
def hlcm_estimate(dset):
    return ymr.lcm_estimate(households_df(dset), "building_id", buildings_df(dset),
                            "hlcm.yaml")


def hlcm_simulate(dset):
    units = ymr.get_vacant_units(households_df(dset), "building_id", buildings_df(dset),
                                 "residential_units")
    return ymr.lcm_simulate(households_df(dset), units, "hlcm.yaml", dset.households, "building_id")


# employment location choice
def elcm_estimate(dset):
    return ymr.lcm_estimate(jobs_df(dset), "building_id", buildings_df(dset),
                            "elcm.yaml")


def elcm_simulate(dset):
    units = ymr.get_vacant_units(jobs_df(dset), "building_id", buildings_df(dset),
                                 "job_spaces")
    units = units.loc[np.random.choice(units.index, size=200000, replace=False)]
    return ymr.lcm_simulate(jobs_df(dset), units, "elcm.yaml", dset.jobs, "building_id")


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


def government_jobs_scaling_model(dset):
    government_sectors = [18, 19, 20]
    def random_choice(chooser_ids, alternative_ids, probabilities):
        choices = pd.Series([np.nan] * len(chooser_ids), index=chooser_ids)
        chosen = np.random.choice(
            alternative_ids, size=len(chooser_ids), replace=True, p=probabilities)
        choices[chooser_ids] = chosen
        return choices
    jobs_to_place = dset.jobs[dset.jobs.building_id.isnull().values]
    segments = jobs_to_place.groupby(['large_area_id','sector_id'])
    for name,segment in segments:
        large_area_id = int(name[0])
        sector = int(name[1])
        if sector in government_sectors:
            jobs_to_place = segment.index.values
            counts_by_bid = dset.jobs[(dset.jobs.sector_id == sector).values*(dset.jobs.large_area_id==large_area_id)].groupby(['building_id']).size()
            prop_by_bid = counts_by_bid/counts_by_bid.sum()
            choices = random_choice(jobs_to_place, prop_by_bid.index.values, prop_by_bid.values)
            dset.jobs.loc[choices.index, 'building_id'] = choices.values
        

def refiner(dset):
    refinements1 = pd.read_csv("data/refinements.csv")
    refinements2 = pd.read_csv("data/employment_events.csv")
    refinements = pd.concat([refinements1,refinements2])
    refinements = refinements[refinements.year == dset.year]
    if len(refinements) > 0:
        def relocate_agents(agents, agent_type, filter_expression, location_type, location_id, number_of_agents):
            agents = dset.view(agent_type).query(filter_expression)
            if location_type == 'zone':
                new_building_id = dset.view("buildings").zone_id[dset.view("buildings").zone_id == location_id].index.values[0]
                agents['zone_id'] = dset.view(agent_type).zone_id
                agent_pool = agents[agents.zone_id != location_id]
            if location_type == 'parcel':
                # buildings_ids = dset.buildings[dset.view("buildings").parcel_id == location_id].index.values
                # if len(building_ids)> 0
                try:
                    new_building_id = dset.view("buildings").parcel_id[dset.view("buildings").parcel_id == location_id].index.values[0]
                except:
                    print 'No building in %s %s.' % (location_type,location_id)
                    return
                agents['parcel_id'] = dset.view(agent_type).parcel_id
                agent_pool = agents[agents.parcel_id != location_id]
            shuffled_ids = agent_pool.index.values
            np.random.shuffle(shuffled_ids)
            agents_to_relocate = shuffled_ids[:number_of_agents]
            if agent_type == 'households':
                idx_agents_to_relocate = np.in1d(dset.households.index.values, agents_to_relocate)
                dset.households.building_id[idx_agents_to_relocate] = new_building_id
            if agent_type == 'jobs':
                idx_agents_to_relocate = np.in1d(dset.jobs.index.values, agents_to_relocate)
                dset.jobs.building_id[idx_agents_to_relocate] = new_building_id

        def unplace_agents(agents, agent_type, filter_expression, location_type, location_id, number_of_agents):
            agents = dset.view(agent_type).query(filter_expression)
            if location_type == 'zone':
                agents['zone_id'] = dset.view(agent_type).zone_id
                agent_pool = agents[agents.zone_id == location_id]
            if location_type == 'parcel':
                agents['parcel_id'] = dset.view(agent_type).parcel_id
                agent_pool = agents[agents.parcel_id == location_id]
            if len(agent_pool) >= number_of_agents:
                shuffled_ids = agent_pool.index.values
                np.random.shuffle(shuffled_ids)
                agents_to_relocate = shuffled_ids[:number_of_agents]
                if agent_type == 'households':
                    idx_agents_to_relocate = np.in1d(dset.households.index.values, agents_to_relocate)
                    dset.households.building_id[idx_agents_to_relocate] = -1
                if agent_type == 'jobs':
                    idx_agents_to_relocate = np.in1d(dset.jobs.index.values, agents_to_relocate)
                    dset.jobs.building_id[idx_agents_to_relocate] = -1
        for idx in refinements.index.values:
            record = refinements[refinements.index.values == idx]
            action = record.action.values[0]
            agent_dataset = record.agent_dataset.values[0]
            filter_expression = record.filter_expression.values[0]
            amount = record.amount.values[0]
            location_id = record.location_id.values[0]
            location_type = record.location_type.values[0]
            if action == 'add':
                if agent_dataset == 'job':
                    relocate_agents(dset.jobs, 'jobs', filter_expression, location_type, location_id, amount)
                if agent_dataset == 'household':
                    relocate_agents(dset.households, 'households', filter_expression, location_type, location_id, amount)
            if action in ['delete', 'subtract']:
                if agent_dataset == 'job':
                    unplace_agents(dset.jobs, 'jobs', filter_expression, location_type, location_id, amount)
                if agent_dataset == 'household':
                    unplace_agents(dset.households, 'households', filter_expression, location_type, location_id, amount)
                    
def aging_model(dset):
    print dset.persons.age.describe()
    dset.persons.age = dset.persons.age + 1
    print dset.persons.age.describe()
    
def income_inflation_model(dset):
    print dset.households.income.describe()
    dset.households.income = dset.households.income*1.01
    print dset.households.income.describe()
    
def scheduled_development_events(dset):
    sched_dev = pd.read_csv("data/scheduled_development_events.csv")
    sched_dev[sched_dev.year_built==dset.year]
    sched_dev['building_sqft'] = 0
    sched_dev["sqft_price_res"] = 0
    sched_dev["sqft_price_nonres"] = 0
    if len(sched_dev) > 0:
        max_bid = dset.buildings.index.values.max()
        idx = np.arange(max_bid + 1,max_bid+len(sched_dev)+1)
        sched_dev['building_id'] = idx
        sched_dev = sched_dev.set_index('building_id')
        from urbansim.developer.developer import Developer
        merge = Developer(pd.DataFrame({})).merge
        all_buildings = merge(dset.buildings,sched_dev[dset.buildings.columns])

def price_vars(dset):
    nodes = networks.from_yaml(dset, "networks2.yaml")
    dset.save_tmptbl("nodes_prices", nodes)


def feasibility(dset):
    pf = sqftproforma.SqFtProForma()

    parcels = dset.view("parcels")
    df = parcels.build_df()
    df = df.ix[np.random.choice(df.index, 500000,replace=False)]
    # add prices for each use
    for use in pf.config.uses:
        df[use] = parcels.price(use)

    # convert from cost to yearly rent
    df[pf.config.uses] *= pf.config.cap_rate
    print df[pf.config.uses].describe()

    d = {}
    for form in pf.config.forms:
        print "Computing feasibility for form %s" % form
        d[form] = pf.lookup(form, df[parcels.allowed(form)])

    far_predictions = pd.concat(d.values(), keys=d.keys(), axis=1)

    dset.save_tmptbl("feasibility", far_predictions)


def residential_developer(dset):
    residential_target_vacancy = .20
    dev = developer.Developer(dset.feasibility)

    target_units = dev.compute_units_to_build(len(dset.households),
                                              dset.buildings.residential_units.sum(),
                                              residential_target_vacancy)

    parcels = dset.view("parcels")
    new_buildings = dev.pick("residential",
                             target_units,
                             parcels.parcel_size,
                             parcels.ave_unit_sqft,
                             parcels.total_units,
                             max_parcel_size=200000,
                             drop_after_build=True)

    new_buildings["year_built"] = dset.year
    new_buildings["form"] = "residential"
    new_buildings["building_type_id"] = new_buildings["form"].apply(dset.random_type)
    new_buildings["stories"] = new_buildings.stories.apply(np.ceil)
    new_buildings['building_id_old'] = 0
    new_buildings['improvement_value'] = 0
    new_buildings['land_area'] = 0
    new_buildings['sqft_per_unit'] = 1500
    new_buildings['tax_exempt'] = 0
    new_buildings['building_sqft'] = 0
    for col in ["sqft_price_res",  "sqft_price_nonres"]:
        new_buildings[col] = np.nan

    #print "NEW BUILDINGS"
    #print new_buildings[dset.buildings.columns].describe()

    print "Adding {} buildings with {:,} residential units".format(len(new_buildings),
                                                                   new_buildings.residential_units.sum())

    all_buildings = dev.merge(dset.buildings, new_buildings[dset.buildings.columns])
    dset.save_tmptbl("buildings", all_buildings)


def non_residential_developer(dset):
    non_residential_target_vacancy = .6
    dev = developer.Developer(dset.feasibility)

    target_units = dev.compute_units_to_build(len(dset.jobs),
                                              dset.view("buildings").non_residential_units.sum(),
                                              non_residential_target_vacancy)

    parcels = dset.view("parcels")
    new_buildings = dev.pick(["office", "retail", "industrial"],
                             target_units,
                             parcels.parcel_size,
                             # This is hard-coding 500 as the average sqft per job
                             # which isn't right but it doesn't affect outcomes much
                             # developer will build enough units assuming 500 sqft
                             # per job but then it just returns the result as square
                             # footage and the actual building_sqft_per_job will be
                             # used to compute non_residential_units.  In other words,
                             # we can over- or under- build the number of units here
                             # but we should still get roughly the right amount of
                             # development out of this and the final numbers are precise.
                             # just move this up and down if dev is over- or under-
                             # buildings things
                             pd.Series(500, index=parcels.index),
                             dset.view("parcels").total_nonres_units,
                             max_parcel_size=200000,
                             drop_after_build=True,
                             residential=False)

    new_buildings["year_built"] = dset.year
    new_buildings["building_type_id"] = new_buildings["form"].apply(dset.random_type)
    new_buildings["residential_units"] = 0
    new_buildings["stories"] = new_buildings.stories.apply(np.ceil)
    new_buildings['building_id_old'] = 0
    new_buildings['improvement_value'] = 0
    new_buildings['land_area'] = 0
    new_buildings['sqft_per_unit'] = 1500
    new_buildings['tax_exempt'] = 0
    new_buildings['building_sqft'] = 0
    for col in ["sqft_price_res",  "sqft_price_nonres"]:
        new_buildings[col] = np.nan

    #print "NEW BUILDINGS"
    #print new_buildings[dset.buildings.columns].describe()

    print "Adding {} buildings with {:,} non-residential sqft".format(len(new_buildings),
                                                                      new_buildings.non_residential_sqft.sum())

    all_buildings = dev.merge(dset.buildings, new_buildings[dset.buildings.columns])
    dset.save_tmptbl("buildings", all_buildings)

def build_networks(dset):
    if networks.NETWORKS is None:
        networks.NETWORKS = networks.Networks(
            [os.path.join(misc.data_dir(), x) for x in ['osm_semcog.pkl']],
            factors=[1.0],
            maxdistances=[2000],
            twoway=[1],
            impedances=None)

    parcels = dset.parcels
    parcels['x'] = parcels.centroid_x
    parcels['y'] = parcels.centroid_y
    parcels = networks.NETWORKS.addnodeid(parcels)
    dset.save_tmptbl("parcels", parcels)


def neighborhood_vars(dset):
    nodes = networks.from_yaml(dset, "networks.yaml")
    dset.save_tmptbl("nodes", nodes)


def _run_models(dset, model_list, years):

    for year in years:

        dset.year = year

        t1 = time.time()

        for model in model_list:
            t2 = time.time()
            print "\n" + model + "\n"
            globals()[model](dset)
            print "Model %s executed in %.3fs" % (model, time.time()-t2)
        print "Year %d completed in %.3fs" % (year, time.time()-t1)
