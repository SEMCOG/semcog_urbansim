import os
import random
import operator
from multiprocessing import Pool

import numpy as np
import orca
import pandana as pdna
import pandas as pd
from urbansim.developer import sqftproforma
from urbansim.models import transition, relocation
from urbansim.utils import misc, networks
from urbansim_parcels import utils as parcel_utils

import utils
import variables


@orca.step()
def diagnostic(parcels, buildings, jobs, households, nodes, iter_var):
    parcels = parcels.to_frame()
    buildings = buildings.to_frame()
    jobs = jobs.to_frame()
    households = households.to_frame()
    nodes = nodes.to_frame()
    import pdb;
    pdb.set_trace()


@orca.step()
def rsh_estimate(buildings, nodes_walk):
    return utils.hedonic_estimate("rsh.yaml", buildings, nodes_walk)


@orca.step()
def rsh_simulate(buildings, nodes_walk):
    return utils.hedonic_simulate("rsh.yaml", buildings, nodes_walk,
                                  "sqft_price_res")


@orca.step()
def nrh_estimate(buildings, nodes_walk):
    return utils.hedonic_estimate("nrh.yaml", buildings, nodes_walk)


@orca.step()
def nrh_simulate(buildings, nodes_walk):
    return utils.hedonic_simulate("nrh.yaml", buildings, nodes_walk,
                                  "sqft_price_nonres")


@orca.step()
def hlcm_estimate(households, buildings, nodes_walk):
    return utils.lcm_estimate("hlcm.yaml", households, "building_id",
                              buildings, nodes_walk)


@orca.step()
def hlcm_simulate(households, buildings, nodes_walk):
    h = households.building_id
    idx_invalid_building_id = np.in1d(h, buildings.index.values) == False
    if idx_invalid_building_id.sum() > 0:
        print("we have households with bad building id's there are #", idx_invalid_building_id.sum())
        h.loc[idx_invalid_building_id] = -1
        households.update_col_from_series('building_id', h)

    return utils.lcm_simulate("hlcm.yaml", households, buildings, nodes_walk,
                              "building_id", "residential_units",
                              "vacant_residential_units")


@orca.step()
def elcm_estimate(jobs, buildings, nodes_drv):
    utils.lcm_estimate("elcm3.yaml", jobs, "building_id", buildings, nodes_drv)
    utils.lcm_estimate("elcm5.yaml", jobs, "building_id", buildings, nodes_drv)
    utils.lcm_estimate("elcm93.yaml", jobs, "building_id", buildings, nodes_drv)
    utils.lcm_estimate("elcm99.yaml", jobs, "building_id", buildings, nodes_drv)
    utils.lcm_estimate("elcm115.yaml", jobs, "building_id", buildings, nodes_drv)
    utils.lcm_estimate("elcm125.yaml", jobs, "building_id", buildings, nodes_drv)
    utils.lcm_estimate("elcm147.yaml", jobs, "building_id", buildings, nodes_drv)
    utils.lcm_estimate("elcm161.yaml", jobs, "building_id", buildings, nodes_drv)


@orca.step()
def elcm_simulate(jobs, buildings, nodes_drv):
    jobs_df = jobs.to_frame()
    buildings = buildings.to_frame()

    idx_invalid_building_id = np.in1d(jobs_df.building_id, buildings.index.values) == False
    if idx_invalid_building_id.sum() > 0:
        print("we have jobs with bad building id's there are #", idx_invalid_building_id.sum())
        jobs_df.loc[idx_invalid_building_id, 'building_id'] = -1

    def register_broadcast_simulate_segment(jobs_df_name, jobs_in_la, buildings_df_name, buildings_in_la, yaml_name):
        orca.add_table(jobs_df_name, jobs_in_la)
        orca.add_table(buildings_df_name, buildings_in_la)
        orca.broadcast('nodes_drv', buildings_df_name, cast_index=True, onto_on='nodeid_drv')
        orca.broadcast('parcels', buildings_df_name, cast_index=True, onto_on='parcel_id')
        orca.broadcast(buildings_df_name, jobs_df_name, cast_index=True, onto_on='building_id')
        jobs_in_la = orca.get_table(jobs_df_name)
        buildings_in_la = orca.get_table(buildings_df_name)
        utils.lcm_simulate(yaml_name, jobs_in_la, buildings_in_la, nodes_drv,
                           "building_id", "job_spaces",
                           "vacant_job_spaces")
        jobs.update_col_from_series('building_id', jobs_in_la.building_id, cast=True)

    for large_area_id in [3, 5, 93, 99, 115, 125, 147, 161]:
        str_id = str(large_area_id)
        register_broadcast_simulate_segment(jobs_df_name='jobs' + str_id,
                                            jobs_in_la=jobs_df[jobs_df.lid == large_area_id],
                                            buildings_df_name='buildings' + str_id,
                                            buildings_in_la=buildings[buildings.large_area_id == large_area_id],
                                            yaml_name='elcm' + str_id + '.yaml')


@orca.step()
def households_relocation(households, annual_relocation_rates_for_households):
    relocation_rates = annual_relocation_rates_for_households.to_frame()
    relocation_rates = relocation_rates.rename(columns={'age_max': 'age_of_head_max', 'age_min': 'age_of_head_min'})
    relocation_rates.probability_of_relocating *= .05
    reloc = relocation.RelocationModel(relocation_rates, 'probability_of_relocating')
    hh = households.to_frame(households.local_columns)
    idx_reloc = reloc.find_movers(hh)
    households.update_col_from_series('building_id',
                                      pd.Series(-1, index=idx_reloc),
                                      cast=True)
    _print_number_unplaced(households, 'building_id')


@orca.step()
def jobs_relocation(jobs, annual_relocation_rates_for_jobs):
    relocation_rates = annual_relocation_rates_for_jobs.to_frame()
    relocation_rates.job_relocation_probability *= .05
    reloc = relocation.RelocationModel(relocation_rates, 'job_relocation_probability')
    j = jobs.to_frame(jobs.local_columns)
    idx_reloc = reloc.find_movers(j)
    j.loc[idx_reloc, "building_id"] = -1
    orca.add_table("jobs", j)
    _print_number_unplaced(jobs, 'building_id')


def presses_trans((ct, hh, p, target, iter_var)):
    ct_finite = ct[ct.persons_max <= 100]
    ct_inf = ct[ct.persons_max > 100]
    tran = transition.TabularTotalsTransition(ct_finite, 'total_number_of_households')
    model = transition.TransitionModel(tran)
    new, added_hh_idx, new_linked = \
        model.transition(hh, iter_var,
                         linked_tables={'linked': (p, 'household_id')})
    new.loc[added_hh_idx, "building_id"] = -1
    pers = new_linked['linked']
    pers = pers[pers.household_id.isin(new.index)]
    out = [[new, pers]]
    target -= len(pers)
    best_qal = np.inf
    best = []
    for _ in range(3):
        tran = transition.TabularTotalsTransition(ct_inf, 'total_number_of_households')
        model = transition.TransitionModel(tran)
        new, added_hh_idx, new_linked = \
            model.transition(hh, iter_var,
                             linked_tables={'linked': (p, 'household_id')})
        new.loc[added_hh_idx, "building_id"] = -1
        pers = new_linked['linked']
        pers = pers[pers.household_id.isin(new.index)]
        qal = abs(target - len(pers))
        if qal < best_qal:
            best = (new, pers)
            best_qal = qal
    out.append(best)
    return out


@orca.step()
def households_transition(households, persons, annual_household_control_totals, remi_pop_total, iter_var):
    region_ct = annual_household_control_totals.to_frame()
    max_cols = region_ct.columns[region_ct.columns.str.endswith('_max') & (region_ct == -1).any(axis=0)]
    region_ct[max_cols] = region_ct[max_cols].replace(-1, np.inf)
    region_ct[max_cols] += 1
    region_hh = households.to_frame(households.local_columns + ['large_area_id'])
    region_p = persons.to_frame(persons.local_columns)
    region_target = remi_pop_total.to_frame()

    def cut_to_la((large_area_id, hh)):
        p = region_p[region_p.household_id.isin(hh.index)]
        target = int(region_target.loc[large_area_id, str(iter_var)])
        ct = region_ct[region_ct.large_area_id == large_area_id]
        del ct["large_area_id"]
        return ct, hh, p, target, iter_var

    arg_per_la = map(cut_to_la, region_hh.groupby('large_area_id'))
    # cunks_per_la = map(presses_trans, arg_per_la)
    pool = Pool(8)
    cunks_per_la = pool.map(presses_trans, arg_per_la)
    pool.close()
    pool.join()
    out = reduce(operator.concat, cunks_per_la)

    # fix indexes
    out_hh_fixed = []
    out_p_fixed = []
    hhidmax = region_hh.index.values.max() + 1
    pidmax = region_p.index.values.max() + 1
    hh_la_lookup = orca.get_injectable("households_large_area_lookup")
    for hh, p in out:
        hh = hh.reset_index()
        hh['household_id_old'] = hh['household_id']
        new_hh = (hh.building_id == -1).sum()
        hh.loc[hh.building_id == -1, 'household_id'] = range(hhidmax, hhidmax + new_hh)
        hhidmax += new_hh
        hhid_map = hh[['household_id_old', 'household_id']].set_index('household_id_old')
        p = pd.merge(p.reset_index(), hhid_map, left_on='household_id', right_index=True)
        new_p = (p.household_id_x != p.household_id_y).sum()
        p.loc[p.household_id_x != p.household_id_y, 'person_id'] = range(pidmax, pidmax + new_p)
        pidmax += new_p
        p['household_id'] = p['household_id_y']

        new_hh_df = hh.set_index('household_id')
        hh_la_lookup = hh_la_lookup.append(new_hh_df[new_hh_df.building_id == -1].large_area_id,
                                           verify_integrity=True)
        out_hh_fixed.append(new_hh_df[households.local_columns])
        out_p_fixed.append(p.set_index('person_id')[persons.local_columns])

    orca.add_injectable("households_large_area_lookup",
                        hh_la_lookup,
                        autocall=False, cache=True)
    orca.add_table("households", pd.concat(out_hh_fixed, verify_integrity=True))
    orca.add_table("persons", pd.concat(out_p_fixed, verify_integrity=True))


@orca.step()
def fix_lpr(households, persons, iter_var, workers_employment_rates_by_large_area):
    from numpy.random import choice
    hh = households.to_frame(households.local_columns + ['large_area_id'])
    hh["target_workers"] = 0
    p = persons.to_frame(persons.local_columns + ['large_area_id'])
    lpr = workers_employment_rates_by_large_area.to_frame(['age_min', 'age_max', str(iter_var)])
    employed = p.worker == True
    p["weight"] = 1.0 / np.sqrt(p.join(hh["workers"], "household_id").workers + 1.0)

    colls = ['persons', 'race_id', 'workers', 'children', 'large_area_id'] # , 'age_of_head'
    same = {tuple(idx): df[['income', 'cars']] for idx, df in hh.groupby(colls)}

    new_employ = []
    new_unemploy = []

    for large_area_id, row in lpr.iterrows():
        select = (p.large_area_id == large_area_id) & (p.age >= row.age_min) & (p.age <= row.age_max)
        lpr = row[str(iter_var)]
        lpr_workers = int(select.sum() * lpr)
        num_workers = (select & employed).sum()

        if lpr_workers > num_workers:
            # employ some persons
            new_employ.append(choice(p[select & (~employed)].index, int(lpr_workers - num_workers), False))
        else:
            # unemploy some persons
            prob = p[select & employed].weight
            prob /= prob.sum()
            new_unemploy.append(choice(p[select & employed].index, int(num_workers - lpr_workers), False, p=prob))

        # print large_area_id, row.age_min, row.age_max, select.sum(), num_workers, lpr_workers, lpr

    if len(new_employ) > 0:
        p.loc[np.concatenate(new_employ), "worker"] = True
    if len(new_unemploy):
        p.loc[np.concatenate(new_unemploy), "worker"] = False

    hh["old_workers"] = hh.workers
    hh.workers = p[p.worker == True].groupby("household_id").size()
    hh.workers = hh.workers.fillna(0)
    changed = (hh.workers != hh.old_workers)

    for match_colls, chh in hh[changed].groupby(colls):
        try:
            match = same[tuple(match_colls)]
            new_workers = choice(match.index, len(chh), True)
            hh.loc[chh.index, ['income', 'cars']] = match.loc[new_workers, ['income', 'cars']].values
        except KeyError:
            pass
            # todo: something better!?

    orca.add_table("households", hh[households.local_columns])
    orca.add_table("persons", p[persons.local_columns])


@orca.step()
def jobs_transition(jobs, annual_employment_control_totals, iter_var):
    ct_emp = annual_employment_control_totals.to_frame()
    ct_emp = ct_emp.reset_index().set_index('year')
    tran = transition.TabularTotalsTransition(ct_emp, 'total_number_of_jobs')
    model = transition.TransitionModel(tran)
    j = jobs.to_frame(jobs.local_columns + ['large_area_id'])
    new, added_jobs_idx, _ = model.transition(j, iter_var)
    orca.add_injectable("jobs_large_area_lookup",
                        new.large_area_id,
                        autocall=False, cache=True)
    new.loc[added_jobs_idx, "building_id"] = -1
    orca.add_table("jobs", new[jobs.local_columns])


@orca.step()
def government_jobs_scaling_model(jobs):
    jobs = jobs.to_frame(jobs.local_columns+['large_area_id'])
    government_sectors = [18]

    def random_choice(chooser_ids, alternative_ids, probabilities):
        return pd.Series(np.random.choice(
            alternative_ids, size=len(chooser_ids), replace=True, p=probabilities), index=chooser_ids)

    jobs_to_place = jobs[jobs.building_id.isnull()]

    if len(jobs_to_place) > 0:
        segments = jobs_to_place.groupby(['large_area_id', 'sector_id'])
        for name, segment in segments:
            large_area_id = int(name[0])
            sector = int(name[1])
            if sector in government_sectors:
                jobs_to_place = segment.index.values
                counts_by_bid = jobs[(jobs.sector_id == sector) & (jobs.large_area_id == large_area_id)].groupby(
                    ['building_id']).size()
                prop_by_bid = counts_by_bid / counts_by_bid.sum()
                choices = random_choice(jobs_to_place, prop_by_bid.index.values, prop_by_bid.values)
                jobs.loc[choices.index, 'building_id'] = choices.values
        orca.add_table("jobs", jobs)


@orca.step()
def gq_pop_scaling_model(jobs):
    gqpop = gqpop.to_frame(gqpop.local_columns+['large_area_id']) # add gq pop table to database
    target_gq = gq_pop_forecast.to_frame()

    diff_gq = target_gq - exist_gq#by large area and age group
    gqpop_lab = gqpop.grouby(['large_area_id', 'age_group', 'building_id']).size()

    for id, row in diff_gq.iterrows():
        local_gqpop =gqpop.loc[(
            gqpop.large_area_id==row.large_area_id) & (gqpop.age_group==row.age_group)]
        if row.diff >=0:
            newgq = local_gqpop.sample(gqpop_lab.building_id, row.diff) #add some updates
        else:
            removegq = local_gqpop.sample(gqpop_lab.building_id, abs(row.diff))
            gqpop.drop(removegq.index, inplace = True)




@orca.step()
def refiner(jobs, households, buildings, iter_var):
    # need clean up
    jobs_columns = jobs.local_columns + ['zone_id', 'large_area_id']
    jobs = jobs.to_frame(jobs_columns)
    households_columns = households.local_columns + ['zone_id', 'large_area_id']
    households = households.to_frame(households_columns)
    buildings = buildings.to_frame(buildings.local_columns + ['zone_id', 'large_area_id'])
    dic_agent = {'jobs': jobs, 'households': households}

  #  refinements1 = pd.read_csv("data/refinements.csv")
    refinements = pd.read_csv("data/employment_events.csv")
    #refinements = pd.concat([refinements1, refinements2])
    refinements = refinements[refinements.year == iter_var]

    def rec_values(record):
        action = record.action
        agents = record.agents
        agents_expression = record.agent_expression
        amount = record.amount
        location_expression = record.location_expression
        return action, agents, agents_expression, amount, location_expression

    if len(refinements) > 0:
        def relocate_agents(agents, agents_pool, agent_expression, location_expression, number_of_agents):
            bselect = buildings.query(location_expression)
            if len(bselect) <= 0:
                print("We can't fined a building to place these agents")
                return agents, agents_pool
            new_building_ids = bselect.sample(number_of_agents, replace=True).index.values
            # maybe use job reallocation instead of random

            if len(agents_pool) > 0:
                agents_sub_pool = agents_pool.query(agent_expression)
                if len(agents_sub_pool) >= number_of_agents:
                    agents_sample = agents_sub_pool.sample(number_of_agents, replace=False)
                else:
                    agents_sample = agents_sub_pool.sample(number_of_agents, replace=True)
                agents_sample.building_id = new_building_ids
                agents_pool.drop(agents_sample.index, inplace=True)
            else:
                agents_sample = agents.query(agent_expression).sample(number_of_agents, replace=True)
                agents_sample.index = agents.index.values.max() + 1 + np.array(range(number_of_agents))
                agents_sample.building_id = new_building_ids
            agents = pd.concat([agents, agents_sample])
            return agents, agents_pool

        def unplace_agents(agents, agents_pool, agent_expression, location_expression, number_of_agents):
            available_agents = agents.query(agent_expression)
            bselect = buildings.query(location_expression)
            local_agents = available_agents.loc[available_agents.building_id.isin(bselect.index.values)]
            if len(local_agents) > 0:
                selected_agents = local_agents.sample(min(len(local_agents), number_of_agents))

                agents_pool = pd.concat([agents_pool, selected_agents])
                agents.drop(selected_agents.index, inplace=True)
            return agents, agents_pool

        def target_agents(agents, agent_expression, location_expression, number_of_agents):
            #use for employment event model
            exist_agents = agents.query(agent_expression)
            bselect = buildings.query(location_expression)
            local_agents = exist_agents.loc[exist_agents.building_id.isin(bselect.index.values)]

            diff = len(local_agents) - number_of_agents
            if diff > 0:
                select_agents = local_agents.sample(diff)
                agents.drop(select_agents.index, inplace=True)
            else:
                if len(local_agents) == 0:
                    local_agents = exist_agents.loc[exist_agents.large_area_id == bselect.large_area_id[0]]
                select_agents = local_agents.sample(abs(diff), replace=True)
                select_agents.index = agents.index.values.max() + 1 + np.array(range(abs(diff)))
                select_agents.building_id = bselect.sample(abs(diff), replace=True).index.values
                agents = pd.concat([agents, select_agents])
            return agents

        for tid, trecords in refinements.groupby("transaction_id"):
            dic_agent['jobs_pool'] = pd.DataFrame(data=None, columns=jobs.columns)
            dic_agent['households_pool'] = pd.DataFrame(data=None, columns=households.columns)
            print '** processing transcaction ', tid

            records = trecords[trecords.action == 'subtract']
            for _, record in records.iterrows():
                print record
                action, agents, agents_expression, amount, location_expression = rec_values(record)
                dic_agent[agents], dic_agent[agents + '_pool'] = unplace_agents(dic_agent[agents],
                                                             dic_agent[agents+'_pool'],
                                                             agents_expression,
                                                            location_expression,
                                                            amount)

            records = trecords[trecords.action == 'add']
            for _, record in records.iterrows():
                print record
                action, agents, agents_expression, amount, location_expression = rec_values(record)
                dic_agent[agents], dic_agent[agents + '_pool'] = relocate_agents(dic_agent[agents],
                                                                                 dic_agent[agents+'_pool'],
                                                                                 agents_expression,
                                                                                 location_expression, amount)

            records = trecords[trecords.action == 'target']
            for _, record in records.iterrows():
                action, agents, agents_expression, amount, location_expression = rec_values(record)
                dic_agent[agents] = target_agents(dic_agent[agents],
                                                  agents_expression,
                                                location_expression,
                                                  amount)

        orca.add_table('jobs', dic_agent['jobs'][jobs_columns])
        orca.add_table('households', dic_agent['households'][households_columns])


@orca.step()
def scheduled_development_events(buildings, iter_var, scheduled_development_events):
    sched_dev = scheduled_development_events.to_frame()
    sched_dev = sched_dev[sched_dev.year_built == iter_var].reset_index(drop=True)
    if len(sched_dev) > 0:
        sched_dev["sqft_price_res"] = 0
        sched_dev["sqft_price_nonres"] = 0
        sched_dev = add_extra_columns_res(sched_dev)
        b = buildings.to_frame(buildings.local_columns)
        max_id = orca.get_injectable("max_building_id")
        all_buildings = parcel_utils.merge_buildings(b, sched_dev[b.columns], False, max_id)
        orca.add_injectable("max_building_id", max(all_buildings.index.max(), max_id))
        orca.add_table("buildings", all_buildings)

        # Todo: maybe we need to impute some columns
        # Todo: parcel use need to be updated
        # Todo: record dev_id -> building_id


@orca.step()
def scheduled_demolition_events(buildings, households, jobs, iter_var, scheduled_demolition_events):
    sched_dev = scheduled_demolition_events.to_frame()
    sched_dev = sched_dev[sched_dev.year_built == iter_var].reset_index(drop=True)
    if len(sched_dev) > 0:
        buildings = buildings.to_frame(buildings.local_columns)
        buildings_idx = buildings[buildings.index.isin(sched_dev.building_id)].index
        orca.add_table("buildings", buildings.drop(buildings_idx))

        # unplace HH
        households = households.to_frame(households.local_columns)
        households.loc[households.building_id.isin(sched_dev.building_id), "building_id"] = -1
        orca.add_table("households", households)

        # unplace jobs
        jobs = jobs.to_frame(jobs.local_columns)
        jobs.loc[jobs.building_id.isin(sched_dev.building_id), "building_id"] = -1
        orca.add_table("jobs", jobs)
        # Todo: parcel use need to be updated


@orca.step()
def price_vars(net_walk):
    nodes = networks.from_yaml(net_walk, "networks_walk.yaml")
    # nodes.residential=nodes.residential*1.2
    print nodes.describe()
    print pd.Series(nodes.index).describe()
    orca.add_table("nodes_prices", nodes)


def parcel_average_price(use):
    # Copied from variables.py
    parcels_wrapper = orca.get_table('parcels')
    if len(orca.get_table('nodes_walk')) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels_wrapper.index)
    return misc.reindex(orca.get_table('nodes_walk')[use],
                        orca.get_table('parcels').nodeid_walk)


@orca.step('feasibility')
def feasibility(parcels):
    parcel_utils.run_feasibility(parcels,
                                 parcel_average_price,
                                 variables.parcel_is_allowed,
                                 cfg='proforma.yaml')


def add_extra_columns_res(df):
    for col in ['improvement_value', 'land_area', 'tax_exempt', 'sqft_price_nonres',
                'sqft_price_res']:
        df[col] = 0
    df['sqft_per_unit'] = 1500
    df = df.fillna(0)
    df['year_built'] = orca.get_injectable('year')
    return df


def add_extra_columns_nonres(df):
    for col in ['improvement_value', 'land_area', 'tax_exempt', 'sqft_price_nonres',
                'sqft_price_res']:
        df[col] = 0
    df['sqft_per_unit'] = 0
    df['year_built'] = orca.get_injectable('year')
    df = df.fillna(0)
    return df


def random_type(row):
    form = row['form']
    form_to_btype = orca.get_injectable("form_to_btype")
    return random.choice(form_to_btype[form])


@orca.step('residential_developer')
def residential_developer(feasibility, households, buildings, parcels, iter_var):
    parcel_utils.run_developer(
        "residential",
        households,
        buildings,
        'residential_units',
        feasibility,
        parcels.parcel_size,
        parcels.ave_unit_size,
        parcels.total_units,
        'res_developer.yaml',
        year=iter_var,
        target_vacancy=.20,
        form_to_btype_callback=random_type,
        add_more_columns_callback=add_extra_columns_res)


@orca.step()
def non_residential_developer(feasibility, jobs, buildings, parcels, iter_var):
    parcel_utils.run_developer(
        ["office", "retail", "industrial", "medical"],
        jobs,
        buildings,
        "job_spaces",
        feasibility,
        parcels.parcel_size,
        parcels.ave_unit_size,
        parcels.total_job_spaces,
        'nonres_developer.yaml',
        year=iter_var,
        target_vacancy=0.60,
        form_to_btype_callback=random_type,
        add_more_columns_callback=add_extra_columns_nonres)


@orca.step()
def build_networks(parcels):
    pdna.network.reserve_num_graphs(2)

    # networks in semcog_networks.h5
    # Todo add injectable that reads from yaml
    dic_net = {'mgf14_drive':
                   {'cost1': 'feet',
                    'cost2': 'minutes',
                    'edges': 'edges_mgf14_drive_full',
                    'local_edges': 'edges_mgf14_drive_local',
                    'local_nodes': 'nodes_mgf14_drive_local',
                    'nodes': 'nodes_mgf14_drive_full'
                    },
               'mgf14_ext_drive':
                   {'cost1': 'feet',
                    'cost2': 'minutes',
                    'edges': 'edges_mgf14_ext_drive_full',
                    'local_edges': 'edges_mgf14_ext_drive_local',
                    'local_nodes': 'nodes_mgf14_ext_drive_local',
                    'nodes': 'nodes_mgf14_ext_drive_full'
                    },
               'mgf14_ext_walk':
                   {'cost1': 'feet',
                    'cost2': 'meters',
                    'edges': 'edges_mgf14_ext_walk',
                    'nodes': 'nodes_mgf14_ext_walk'
                    },
               'mgf14_walk':
                   {'cost1': 'meters',
                    'edges': 'edges_mgf14_walk',
                    'nodes': 'nodes_mgf14_walk'
                    },
               'tdm':
                   {'cost1': 'peak_mins',
                    'cost2': 'nonpk_mins',
                    'edges': 'edges_tdm_full',
                    'local_edges': 'edges_tdm_local',
                    'local_nodes': 'nodes_tdm_local',
                    'nodes': 'nodes_tdm_full'
                    },
               'tdm_ext':
                   {'cost1': 'peak_mins',
                    'cost2': 'nonpk_mins',
                    'edges': 'edges_tdm_ext_full',
                    'local_edges': 'edges_tdm_ext_local',
                    'local_nodes': 'nodes_tdm_ext_local',
                    'nodes': 'nodes_tdm_ext_full'
                    }
               }

    st = pd.HDFStore(os.path.join(misc.data_dir(), "semcog_networks.h5"), "r")

    lstnet = [
        {
            'name': 'mgf14_ext_walk',
            'cost': 'cost1',
            'prev': 10560,  # 2 miles
            'net': 'net_walk'
        },
        {
            'name': 'tdm_ext',
            'cost': 'cost1',
            'prev': 60,  # 60 minutes
            'net': 'net_drv'
        },
    ]

    for n in lstnet:
        nodes, edges = st[dic_net[n['name']]['nodes']], st[dic_net[n['name']]['edges']]
        net = pdna.Network(nodes["x"], nodes["y"], edges["from"], edges["to"],
                           edges[[dic_net[n['name']][n['cost']]]])
        net.precompute(n['prev'])
        net.init_pois(num_categories=10, max_dist=n['prev'], max_pois=3)

        orca.add_injectable(n['net'], net)

    # spatially join node ids to parcels
    p = parcels.local
    p['nodeid_walk'] = orca.get_injectable('net_walk').get_node_ids(p['centroid_x'], p['centroid_y'])
    p['nodeid_drv'] = orca.get_injectable('net_drv').get_node_ids(p['centroid_x'], p['centroid_y'])
    orca.add_table("parcels", p)


@orca.step()
def neighborhood_vars(jobs, households, buildings):
    b = buildings.to_frame(['large_area_id'])
    j = jobs.to_frame()
    # j = jobs.to_frame(jobs.local_columns)
    h = households.to_frame(households.local_columns)
    idx_invalid_building_id = np.in1d(j.building_id, b.index.values) == False
    # print j.large_area_id.head()
    # print b.large_area_id.head()

    if idx_invalid_building_id.sum() > 0:
        print("we have jobs with bad building id's there are #", idx_invalid_building_id.sum())
        j.loc[idx_invalid_building_id, 'building_id'] = np.random.choice(
            b.index.values,
            idx_invalid_building_id.sum())
        orca.add_table("jobs", j)
    idx_invalid_building_id = np.in1d(h.building_id, b.index.values) == False
    if idx_invalid_building_id.sum() > 0:
        print("we have households with bad building id's there are #", idx_invalid_building_id.sum())
        h.loc[idx_invalid_building_id, 'building_id'] = np.random.choice(
            b.index.values,
            idx_invalid_building_id.sum())
        orca.add_table("households", h)

    nodes = networks.from_yaml(orca.get_injectable('net_walk'), "networks_walk.yaml")
    # print nodes.describe()
    # print pd.Series(nodes.index).describe()
    orca.add_table("nodes_walk", nodes)

    nodes = networks.from_yaml(orca.get_injectable('net_drv'), "networks_drv.yaml")
    # print nodes.describe()
    # print pd.Series(nodes.index).describe()
    orca.add_table("nodes_drv", nodes)

    post_access_variables()


@orca.step('gq_model')  # group quarters
def gq_model(iter_var, tazcounts2015gq, tazcounts2020gq, tazcounts2025gq, tazcounts2030gq, tazcounts2035gq,
             tazcounts2040gq):
    # Configuration
    max_iterations = 500
    convergence_criteria = .000001
    first_year_to_run = 2015

    # Function used to allocate aggregate GQ from large area controls to TAZ (to form the row marginal)
    def random_choice(chooser_ids, alternative_ids, probabilities):
        choices = pd.Series([np.nan] * len(chooser_ids), index=chooser_ids)
        chosen = np.random.choice(
            alternative_ids, size=len(chooser_ids), replace=True, p=probabilities)
        choices[chooser_ids] = chosen
        return choices

    tazcounts = pd.read_csv('gq/tazcounts.csv').set_index('tazce10')
    gqcontrols = pd.read_csv('gq/largearea_gq_controls.csv')

    # Determine years to run from the control totals
    years = []
    for col in gqcontrols.columns:
        try:
            if int(col) >= first_year_to_run:
                years.append(col)
        except:
            pass

    if iter_var == np.array([int(yr) for yr in years]).min():
        for year in years:
            print year
            for lid in np.unique(gqcontrols.largearea_id):
                print 'Large area id %s' % lid
                gq_lid = gqcontrols[['age_grp', year]][gqcontrols.largearea_id == lid]
                gq_lid = gq_lid.set_index('age_grp')
                tazcounts_lid = tazcounts[['gq04', 'gq517', 'gq1834', 'gq3564', 'gq65plus']][
                    tazcounts.largearea_id == lid]
                taz_sum = tazcounts_lid.gq04 + tazcounts_lid.gq517 + tazcounts_lid.gq1834 + tazcounts_lid.gq3564 + tazcounts_lid.gq65plus
                diff = gq_lid.sum().values[0] - taz_sum.sum()
                # print 'GQ change is %s' % diff
                if diff != 0:
                    ##Allocation of GQ total to TAZ to prepare the row marginal
                    if diff > 0:
                        taz_probs = taz_sum / taz_sum.sum()
                        chosen_taz = random_choice(np.arange(diff), taz_probs.index.values, taz_probs.values)
                        for taz in chosen_taz:
                            taz_sum[taz_sum.index.values == int(taz)] += 1
                    if diff < 0:
                        taz_probs = taz_sum / taz_sum.sum()
                        chosen_taz = random_choice(np.arange(abs(diff)), taz_probs.index.values, taz_probs.values)
                        for taz in chosen_taz:
                            taz_sum[taz_sum.index.values == int(taz)] -= 1
                    ##IPF procedure
                    marginal1 = taz_sum[taz_sum > 0]
                    marginal2 = gq_lid[year]
                    tazes_to_fit = marginal1.index.values
                    seed = tazcounts_lid[np.in1d(tazcounts_lid.index.values, tazes_to_fit)]
                    i = 0
                    while 1:
                        i += 1
                        axis1_adj = marginal1 / seed.sum(axis=1)
                        axis1_adj_mult = np.reshape(np.repeat(axis1_adj.values, seed.shape[1]), seed.shape)
                        seed = seed * axis1_adj_mult
                        axis2_adj = marginal2 / seed.sum()
                        axis2_adj_mult = np.tile(axis2_adj.values, len(seed)).reshape(seed.shape)
                        seed = seed * axis2_adj_mult
                        if ((np.abs(axis1_adj - 1).max() < convergence_criteria) and (
                            np.abs(axis2_adj - 1).max() < convergence_criteria)) or (i >= max_iterations):
                            rounded = np.round(seed)
                            rounding_error = marginal2.sum() - rounded.sum().sum()
                            for col in rounded.columns:
                                tazcounts.loc[rounded.index, col] = rounded[col].values
                            break
            tazcounts.to_csv('gq/tazcounts%s.csv' % year)
            orca.add_table('tazcounts%sgq' % year, tazcounts.copy())


@orca.step()
def travel_model(iter_var, travel_data, buildings, parcels, households, persons, jobs):
    if iter_var in [2015, 2020, 2025, 2030, 2035, 2040]:
        datatable = 'TAZ Data Table'
        joinfield = 'ZoneID'

        input_dir = './/runs//'  ##Where TM expects input
        input_file = input_dir + 'tm_input.tab'
        output_dir = './/data//'  ##Where TM outputs
        output_file = 'tm_output.txt'

        def delete_dcc_file(dcc_file):
            if os.path.exists(dcc_file):
                os.remove(dcc_file)

        delete_dcc_file(os.path.splitext(input_file)[0] + '.dcc')

        parcels = parcels.to_frame()  # (['zone_id','parcel_sqft'])
        hh = households.to_frame()
        persons = persons.to_frame()
        jobs = jobs.to_frame()

        zonal_indicators = pd.DataFrame(index=np.unique(parcels.zone_id.values))
        zonal_indicators['AcresTotal'] = parcels.groupby('zone_id').parcel_sqft.sum() / 43560.0
        zonal_indicators['Households'] = hh.groupby('zone_id').size()
        zonal_indicators['HHPop'] = hh.groupby('zone_id').persons.sum()
        zonal_indicators['EmpPrinc'] = jobs.groupby('zone_id').size()
        # zonal_indicators['workers'] = hh.groupby('zone_id').workers.sum()
        zonal_indicators['Agegrp1'] = persons[persons.age <= 4].groupby('zone_id').size()  # ???
        zonal_indicators['Agegrp2'] = persons[(persons.age >= 5) * (persons.age <= 17)].groupby('zone_id').size()  # ???
        zonal_indicators['Agegrp3'] = persons[(persons.age >= 18) * (persons.age <= 34)].groupby('zone_id').size()  # ???
        zonal_indicators['Age_18to34'] = persons[(persons.age >= 18) * (persons.age <= 34)].groupby('zone_id').size()
        zonal_indicators['Agegrp4'] = persons[(persons.age >= 35) * (persons.age <= 64)].groupby('zone_id').size()  # ???
        zonal_indicators['Agegrp5'] = persons[persons.age >= 65].groupby('zone_id').size()  # ???
        enroll_ratios = pd.read_csv("data/schdic_taz10.csv")
        school_age_by_district = pd.DataFrame({'children': persons[(persons.age >= 5) * (persons.age <= 17)].groupby('school_district_id').size()})
        enroll_ratios = pd.merge(enroll_ratios, school_age_by_district, left_on='school_district_id', right_index=True)
        enroll_ratios['enrolled'] = enroll_ratios.enroll_ratio * enroll_ratios.children
        enrolled = enroll_ratios.groupby('zone_id').enrolled.sum()
        zonal_indicators['K12Enroll'] = np.round(enrolled)
        zonal_indicators['PopDens'] = zonal_indicators.HHPop / (parcels.groupby('zone_id').parcel_sqft.sum() / 43560)
        zonal_indicators['EmpDens'] = zonal_indicators.EmpPrinc / (parcels.groupby('zone_id').parcel_sqft.sum() / 43560)
        # zonal_indicators['EmpBasic'] = jobs[jobs.sector_id.isin([1,3])].groupby('zone_id').size()
        # zonal_indicators['EmpNonBas'] = jobs[~jobs.sector_id.isin([1,3])].groupby('zone_id').size()
        zonal_indicators['Natural_Resource_and_Mining'] = jobs[jobs.sector_id == 1].groupby('zone_id').size()
        # zonal_indicators['sector2'] = jobs[jobs.sector_id==2].groupby('zone_id').size()
        zonal_indicators['Manufacturing'] = jobs[jobs.sector_id == 3].groupby('zone_id').size()
        zonal_indicators['Wholesale_Trade'] = jobs[jobs.sector_id == 4].groupby('zone_id').size()
        zonal_indicators['Retail_Trade'] = jobs[jobs.sector_id == 5].groupby('zone_id').size()
        zonal_indicators['Transportation_and_Warehousing'] = jobs[jobs.sector_id == 6].groupby('zone_id').size()
        zonal_indicators['Utilities'] = jobs[jobs.sector_id == 7].groupby('zone_id').size()
        zonal_indicators['Information'] = jobs[jobs.sector_id == 8].groupby('zone_id').size()
        zonal_indicators['Financial_Service'] = jobs[jobs.sector_id == 9].groupby('zone_id').size()
        zonal_indicators['Professional_Science_Tec'] = jobs[jobs.sector_id == 10].groupby('zone_id').size()
        zonal_indicators['Management_of_CompEnt'] = jobs[jobs.sector_id == 11].groupby('zone_id').size()
        zonal_indicators['Administrative_Support_and_WM'] = jobs[jobs.sector_id == 12].groupby('zone_id').size()
        zonal_indicators['Education_Services'] = jobs[jobs.sector_id == 13].groupby('zone_id').size()
        # zonal_indicators['sector14'] = jobs[jobs.sector_id==14].groupby('zone_id').size()
        # zonal_indicators['sector15'] = jobs[jobs.sector_id==15].groupby('zone_id').size()
        zonal_indicators['Health_Care_and_SocialSer'] = jobs[np.in1d(jobs.sector_id, [14, 15, 19])].groupby('zone_id').size()
        zonal_indicators['Leisure_and_Hospitality'] = jobs[jobs.sector_id == 16].groupby('zone_id').size()
        zonal_indicators['Other_Services'] = jobs[jobs.sector_id == 17].groupby('zone_id').size()
        zonal_indicators['sector18'] = jobs[jobs.sector_id == 18].groupby('zone_id').size()
        zonal_indicators['sector19'] = jobs[jobs.sector_id == 19].groupby('zone_id').size()
        zonal_indicators['Public_Administration'] = jobs[jobs.sector_id == 20].groupby('zone_id').size()

        hh['schoolkids'] = persons[(persons.age >= 5) * (persons.age <= 17)].groupby('household_id').size()
        hh.schoolkids = hh.schoolkids.fillna(0)
        zonal_indicators['PrCh21'] = hh[(hh.persons == 2) * (hh.schoolkids == 1)].groupby('zone_id').size()
        zonal_indicators['PrCh31'] = hh[(hh.persons == 3) * (hh.schoolkids == 1)].groupby('zone_id').size()
        zonal_indicators['PrCh32'] = hh[(hh.persons == 3) * (hh.schoolkids == 2)].groupby('zone_id').size()
        zonal_indicators['PrCh41'] = hh[(hh.persons == 4) * (hh.schoolkids == 1)].groupby('zone_id').size()
        zonal_indicators['PrCh42'] = hh[(hh.persons == 4) * (hh.schoolkids == 2)].groupby('zone_id').size()
        zonal_indicators['PrCh43'] = hh[(hh.persons == 4) * (hh.schoolkids >= 3)].groupby('zone_id').size()
        zonal_indicators['PrCh51'] = hh[(hh.persons == 5) * (hh.schoolkids == 1)].groupby('zone_id').size()
        zonal_indicators['PrCh52'] = hh[(hh.persons == 5) * (hh.schoolkids == 2)].groupby('zone_id').size()
        zonal_indicators['PrCh53'] = hh[(hh.persons == 5) * (hh.schoolkids >= 3)].groupby('zone_id').size()
        hh['quartile'] = pd.Series(pd.qcut(hh.income, 4, labels=False), index=hh.index) + 1
        zonal_indicators['Inc1HHsze1'] = hh[(hh.persons == 1) * (hh.quartile == 1)].groupby('zone_id').size()
        zonal_indicators['Inc2HHsze1'] = hh[(hh.persons == 1) * (hh.quartile == 2)].groupby('zone_id').size()
        zonal_indicators['Inc3HHsze1'] = hh[(hh.persons == 1) * (hh.quartile == 3)].groupby('zone_id').size()
        zonal_indicators['Inc4HHsze1'] = hh[(hh.persons == 1) * (hh.quartile == 4)].groupby('zone_id').size()
        zonal_indicators['Inc1HHsze2'] = hh[(hh.persons == 2) * (hh.quartile == 1)].groupby('zone_id').size()
        zonal_indicators['Inc2HHsze2'] = hh[(hh.persons == 2) * (hh.quartile == 2)].groupby('zone_id').size()
        zonal_indicators['Inc3HHsze2'] = hh[(hh.persons == 2) * (hh.quartile == 3)].groupby('zone_id').size()
        zonal_indicators['Inc4HHsze2'] = hh[(hh.persons == 2) * (hh.quartile == 4)].groupby('zone_id').size()
        zonal_indicators['Inc1HHsze3'] = hh[(hh.persons == 3) * (hh.quartile == 1)].groupby('zone_id').size()
        zonal_indicators['Inc2HHsze3'] = hh[(hh.persons == 3) * (hh.quartile == 2)].groupby('zone_id').size()
        zonal_indicators['Inc3HHsze3'] = hh[(hh.persons == 3) * (hh.quartile == 3)].groupby('zone_id').size()
        zonal_indicators['Inc4HHsze3'] = hh[(hh.persons == 3) * (hh.quartile == 4)].groupby('zone_id').size()
        zonal_indicators['Inc1HHsze4'] = hh[(hh.persons == 4) * (hh.quartile == 1)].groupby('zone_id').size()
        zonal_indicators['Inc2HHsze4'] = hh[(hh.persons == 4) * (hh.quartile == 2)].groupby('zone_id').size()
        zonal_indicators['Inc3HHsze4'] = hh[(hh.persons == 4) * (hh.quartile == 3)].groupby('zone_id').size()
        zonal_indicators['Inc4HHsze4'] = hh[(hh.persons == 4) * (hh.quartile == 4)].groupby('zone_id').size()
        zonal_indicators['Inc1HHsze5p'] = hh[(hh.persons == 5) * (hh.quartile == 1)].groupby('zone_id').size()
        zonal_indicators['Inc2HHsze5p'] = hh[(hh.persons == 5) * (hh.quartile == 2)].groupby('zone_id').size()
        zonal_indicators['Inc3HHsze5p'] = hh[(hh.persons == 5) * (hh.quartile == 3)].groupby('zone_id').size()
        zonal_indicators['Inc4HHsze5p'] = hh[(hh.persons == 5) * (hh.quartile == 4)].groupby('zone_id').size()
        zonal_indicators['WkAu10'] = hh[(hh.workers == 1) * (hh.cars == 0)].groupby('zone_id').size()
        zonal_indicators['WkAu11'] = hh[(hh.workers == 1) * (hh.cars == 1)].groupby('zone_id').size()
        zonal_indicators['WkAu12'] = hh[(hh.workers == 1) * (hh.cars == 2)].groupby('zone_id').size()
        zonal_indicators['WkAu13'] = hh[(hh.workers == 1) * (hh.cars >= 3)].groupby('zone_id').size()
        zonal_indicators['WkAu20'] = hh[(hh.workers == 2) * (hh.cars == 0)].groupby('zone_id').size()
        zonal_indicators['WkAu21'] = hh[(hh.workers == 2) * (hh.cars == 1)].groupby('zone_id').size()
        zonal_indicators['WkAu22'] = hh[(hh.workers == 2) * (hh.cars == 2)].groupby('zone_id').size()
        zonal_indicators['WkAu23'] = hh[(hh.workers == 2) * (hh.cars >= 3)].groupby('zone_id').size()
        zonal_indicators['WkAu30'] = hh[(hh.workers >= 3) * (hh.cars == 0)].groupby('zone_id').size()
        zonal_indicators['WkAu31'] = hh[(hh.workers >= 3) * (hh.cars == 1)].groupby('zone_id').size()
        zonal_indicators['WkAu32'] = hh[(hh.workers >= 3) * (hh.cars == 2)].groupby('zone_id').size()
        zonal_indicators['WkAu33'] = hh[(hh.workers >= 3) * (hh.cars >= 3)].groupby('zone_id').size()

        zonal_indicators['PrAu10'] = hh[(hh.persons == 1) * (hh.cars == 0)].groupby('zone_id').size()
        zonal_indicators['PrAu11'] = hh[(hh.persons == 1) * (hh.cars == 1)].groupby('zone_id').size()
        zonal_indicators['PrAu12'] = hh[(hh.persons == 1) * (hh.cars == 2)].groupby('zone_id').size()
        zonal_indicators['PrAu13'] = hh[(hh.persons == 1) * (hh.cars >= 3)].groupby('zone_id').size()
        zonal_indicators['PrAu20'] = hh[(hh.persons == 2) * (hh.cars == 0)].groupby('zone_id').size()
        zonal_indicators['PrAu21'] = hh[(hh.persons == 2) * (hh.cars == 1)].groupby('zone_id').size()
        zonal_indicators['PrAu22'] = hh[(hh.persons == 2) * (hh.cars == 2)].groupby('zone_id').size()
        zonal_indicators['PrAu23'] = hh[(hh.persons == 2) * (hh.cars >= 3)].groupby('zone_id').size()
        zonal_indicators['PrAu30'] = hh[(hh.persons == 3) * (hh.cars == 0)].groupby('zone_id').size()
        zonal_indicators['PrAu31'] = hh[(hh.persons == 3) * (hh.cars == 1)].groupby('zone_id').size()
        zonal_indicators['PrAu32'] = hh[(hh.persons == 3) * (hh.cars == 2)].groupby('zone_id').size()
        zonal_indicators['PrAu33'] = hh[(hh.persons == 3) * (hh.cars >= 3)].groupby('zone_id').size()
        zonal_indicators['PrAu40'] = hh[(hh.persons == 4) * (hh.cars == 0)].groupby('zone_id').size()
        zonal_indicators['PrAu41'] = hh[(hh.persons == 4) * (hh.cars == 1)].groupby('zone_id').size()
        zonal_indicators['PrAu42'] = hh[(hh.persons == 4) * (hh.cars == 2)].groupby('zone_id').size()
        zonal_indicators['PrAu43'] = hh[(hh.persons == 4) * (hh.cars >= 3)].groupby('zone_id').size()
        zonal_indicators['PrAu50'] = hh[(hh.persons == 5) * (hh.cars == 0)].groupby('zone_id').size()
        zonal_indicators['PrAu51'] = hh[(hh.persons == 5) * (hh.cars == 1)].groupby('zone_id').size()
        zonal_indicators['PrAu52'] = hh[(hh.persons == 5) * (hh.cars == 2)].groupby('zone_id').size()
        zonal_indicators['PrAu53'] = hh[(hh.persons == 5) * (hh.cars >= 3)].groupby('zone_id').size()

        # zonal_indicators['inc1nc'] = hh[(hh.quartile==1)*(hh.children==0)].groupby('zone_id').size()
        # zonal_indicators['inc1wc'] = hh[(hh.quartile==1)*(hh.children>0)].groupby('zone_id').size()
        # zonal_indicators['inc2nc'] = hh[(hh.quartile==2)*(hh.children==0)].groupby('zone_id').size()
        # zonal_indicators['inc2wc'] = hh[(hh.quartile==2)*(hh.children>0)].groupby('zone_id').size()
        # zonal_indicators['inc3nc'] = hh[(hh.quartile==3)*(hh.children==0)].groupby('zone_id').size()
        # zonal_indicators['inc3wc'] = hh[(hh.quartile==3)*(hh.children>0)].groupby('zone_id').size()
        # zonal_indicators['inc4nc'] = hh[(hh.quartile==4)*(hh.children==0)].groupby('zone_id').size()
        # zonal_indicators['inc4wc'] = hh[(hh.quartile==4)*(hh.children>0)].groupby('zone_id').size()

        zonal_indicators['Inc1w0'] = hh[(hh.quartile == 1) * (hh.workers == 0)].groupby('zone_id').size()
        zonal_indicators['Inc1w1'] = hh[(hh.quartile == 1) * (hh.workers == 1)].groupby('zone_id').size()
        zonal_indicators['Inc1w2'] = hh[(hh.quartile == 1) * (hh.workers == 2)].groupby('zone_id').size()
        zonal_indicators['Inc1w3p'] = hh[(hh.quartile == 1) * (hh.workers >= 3)].groupby('zone_id').size()
        zonal_indicators['Inc2w0'] = hh[(hh.quartile == 2) * (hh.workers == 0)].groupby('zone_id').size()
        zonal_indicators['Inc2w1'] = hh[(hh.quartile == 2) * (hh.workers == 1)].groupby('zone_id').size()
        zonal_indicators['Inc2w2'] = hh[(hh.quartile == 2) * (hh.workers == 2)].groupby('zone_id').size()
        zonal_indicators['Inc2w3p'] = hh[(hh.quartile == 2) * (hh.workers >= 3)].groupby('zone_id').size()
        zonal_indicators['Inc3w0'] = hh[(hh.quartile == 3) * (hh.workers == 0)].groupby('zone_id').size()
        zonal_indicators['Inc3w1'] = hh[(hh.quartile == 3) * (hh.workers == 1)].groupby('zone_id').size()
        zonal_indicators['Inc3w2'] = hh[(hh.quartile == 3) * (hh.workers == 2)].groupby('zone_id').size()
        zonal_indicators['Inc3w3p'] = hh[(hh.quartile == 3) * (hh.workers >= 3)].groupby('zone_id').size()
        zonal_indicators['Inc4w0'] = hh[(hh.quartile == 4) * (hh.workers == 0)].groupby('zone_id').size()
        zonal_indicators['Inc4w1'] = hh[(hh.quartile == 4) * (hh.workers == 1)].groupby('zone_id').size()
        zonal_indicators['Inc4w2'] = hh[(hh.quartile == 4) * (hh.workers == 2)].groupby('zone_id').size()
        zonal_indicators['Inc4w3p'] = hh[(hh.quartile == 4) * (hh.workers >= 3)].groupby('zone_id').size()

        zonal_indicators['Workers4HH_IncomeGroup1'] = hh[hh.quartile == 1].groupby('zone_id').workers.sum()
        zonal_indicators['Workers4HH_IncomeGroup2'] = hh[hh.quartile == 2].groupby('zone_id').workers.sum()
        zonal_indicators['Workers4HH_IncomeGroup3'] = hh[hh.quartile == 3].groupby('zone_id').workers.sum()
        zonal_indicators['Workers4HH_IncomeGroup4'] = hh[hh.quartile == 4].groupby('zone_id').workers.sum()

        if os.path.exists('gq/tazcounts%s.csv' % iter_var):
            gq = pd.read_csv('gq/tazcounts%s.csv' % iter_var).set_index('tazce10')
            gq['GrPop'] = gq.gq04 + gq.gq517 + gq.gq1834 + gq.gq3564 + gq.gq65plus
            zonal_indicators['GrPop'] = gq['GrPop']
            zonal_indicators['Population'] = zonal_indicators['GrPop'] + zonal_indicators['HHPop']

        ##Update parcel land_use_type_id
        buildings = buildings.to_frame(['parcel_id', 'building_type_id', 'year_built'])
        new_construction = buildings[buildings.year_built == iter_var].groupby('parcel_id').building_type_id.median()
        if len(new_construction) > 0:
            parcels.loc[new_construction.index, 'land_use_type_id'] = new_construction.values
            orca.add_table("parcels", parcels)

        emp_btypes = orca.get_injectable('emp_btypes')
        emp_parcels = buildings[np.in1d(buildings.building_type_id,emp_btypes)].groupby('parcel_id').size().index.values
        parcels['emp'] = 0
        parcels.emp[np.in1d(parcels.index.values, emp_parcels)] = 1
        parcels['emp_acreage'] = parcels.emp * parcels.parcel_sqft / 43560.0
        zonal_indicators['AcresEmp'] = parcels.groupby('zone_id').emp_acreage.sum()

        zonal_indicators['TAZCE10_N'] = zonal_indicators.index.values
        # zonal_indicators = zonal_indicators.fillna(0).reset_index().rename({'ZoneID':'TAZCE10_N'})

        taz_table = pd.read_csv("data/taz_table.csv")

        merged = pd.merge(taz_table, zonal_indicators, left_on='TAZCE10_N', right_on='TAZCE10_N', how='left')

        merged.to_csv(input_file, sep='\t', index=False)

        #######################################################################
        ####    TRANSCAD INTERACTIONS #########################################
        #######################################################################
        if orca.get_injectable("transcad_available") == True:
            transcad.transcad_interaction(merged, taz_table)


# @orca.step()
# def housing_value_update(iter_var):
#     "update sev with income growth to attempt to fix profarma"
#     income_forecast = special_forecast.to_frame().loc['year'==iter_var][['large_area_id','income_growh_rate']]
#     parcels = parcels.to_frame()
#     parcels = pd.merge(parcels, income_growth_rate, left_on='large_area_id', left_on='large_area_id', how = 'left')
#     parcels['sev_value'] =  parcels['sev_value'] * parcels['income_growth_rate']
#     parcels.drop('income_growth_rate',inplace=True)
#     orca.add_table("parcels", parcels)


def _print_number_unplaced(df, fieldname="building_id"):
    """
    Just an internal function to use to compute and print info on the number
    of unplaced agents.
    """
    counts = (df[fieldname] == -1).sum()
    print "Total currently unplaced: %d" % counts


def post_access_variables():
    """
    Disaggregate nodal variables to building.
    """

    geographic_levels = [('nodes_walk', 'nodeid_walk'),
                         ('nodes_drv', 'nodeid_drv')]

    for geography in geographic_levels:
        geography_name = geography[0]
        geography_id = geography[1]
        if geography_name != 'buildings':
            building_vars = orca.get_table('buildings').columns
            for var in orca.get_table(geography_name).columns:
                if var not in building_vars:
                    variables.make_disagg_var(geography_name, 'buildings', var, geography_id)
