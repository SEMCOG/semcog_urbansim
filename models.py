import os
import yaml
import operator
from multiprocessing import Pool

import numpy as np
import orca
import pandana as pdna
import pandas as pd
from urbansim.models import transition, relocation
from urbansim.utils import misc, networks
from urbansim_parcels import utils as parcel_utils

import utils
import lcm_utils
import variables
from functools import reduce

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.statespace import sarimax

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# from neuralprophet import NeuralProphet, utils
import neuralprophet

neuralprophet.utils.set_random_seed(0)

# Set up location choice model objects.
# Register as injectable to be used throughout simulation
location_choice_models = {}
hlcm_step_names = []
elcm_step_names = []
model_configs = lcm_utils.get_model_category_configs()
for model_category_name, model_category_attributes in model_configs.items():
    if model_category_attributes["model_type"] == "location_choice":
        model_config_files = model_category_attributes["config_filenames"]

        for model_config in model_config_files:
            model = lcm_utils.create_lcm_from_config(
                model_config, model_category_attributes
            )
            location_choice_models[model.name] = model

            if model_category_name == "hlcm":
                hlcm_step_names.append(model.name)

            if model_category_name == "elcm":
                elcm_step_names.append(model.name)

orca.add_injectable("location_choice_models", location_choice_models)
orca.add_injectable("hlcm_step_names", sorted(hlcm_step_names, reverse=True))
orca.add_injectable("elcm_step_names", sorted(elcm_step_names, reverse=True))

for name, model in list(location_choice_models.items()):
    lcm_utils.register_choice_model_step(
        model.name, model.choosers, choice_function=lcm_utils.unit_choices
    )


@orca.step()
def elcm_home_based(jobs, households):
    wrap_jobs = jobs
    _print_number_unplaced(wrap_jobs, "building_id")
    jobs = wrap_jobs.to_frame(["building_id", "home_based_status", "large_area_id"])
    jobs = jobs[(jobs.home_based_status >= 1) & (jobs.building_id == -1)]
    hh = households.to_frame(["building_id", "large_area_id"])
    hh = hh[hh.building_id > 0]

    for la, la_job in jobs.groupby("large_area_id"):
        la_hh = hh[hh.large_area_id == la]
        la_job["building_id"] = la_hh.sample(
            len(la_job), replace=True
        ).building_id.values
        wrap_jobs.update_col_from_series(
            "building_id", la_job["building_id"], cast=True
        )

    _print_number_unplaced(wrap_jobs, "building_id")


@orca.step()
def mcd_hu_sampling(mcd_total, buildings):
    """
    Apply the mcd total forecast to Limit and calculate the pool of housing 
    units to match the distribution of the mcd_total growth table for the large_area
    Parameters
    ----------
    mcd_total : pandas.DataFrame
        MCD total table
    buildings : pandas.DataFrame
        Buildings table 
    Returns
    -------
    new_units : pandas.Series
        Index of alternatives which have been picked as the candidates
    """
    year = orca.get_injectable("year")
    # get housing unit table from buildings
    vacant_variable = "vacant_residential_units"
    blds = buildings.to_frame(
        [
            "building_id",
            "city_id",
            vacant_variable,
            "building_age",
            "census_bg_id",
            "mcd_model_quota",
        ]
    )
    vacant_units = blds[vacant_variable]
    vacant_units = vacant_units[vacant_units.index.values >= 0]
    vacant_units = vacant_units[vacant_units > 0]
    indexes = np.repeat(vacant_units.index.values, vacant_units.values.astype("int"))
    housing_units = blds.loc[indexes]
    # quota
    # init output df
    new_units = pd.Series()

    # the mcd_total for year and year-1
    mcd_total = mcd_total.to_frame([str(year - 1), str(year)])
    # get the growth by subtract the previous year
    mcd_growth = mcd_total[str(year)] - mcd_total[str(year - 1)]
    ####
    # generating pseudo bg trend table
    unique_bg_id = blds[blds.census_bg_id.notna()].census_bg_id.unique()
    bg_trend = pd.Series(
        # random mcd total growth dataframe
        np.random.randint(-100, 100, len(unique_bg_id)),
        index=unique_bg_id.astype("int"),
        name="bg_trend",
    )
    bg_trend.index.name = "census_bg_id"
    bg_trend_norm_by_bg = (bg_trend - bg_trend.mean()) / bg_trend.std()

    # only selecting growth > 0
    mcd_growth = mcd_growth[mcd_growth > 0]
    for city in mcd_growth.index:
        # for each city, make n_units = n_choosers
        # sorted by year built
        city_units = housing_units[housing_units.city_id == city]
        # building_age normalized
        building_age = city_units.building_age
        building_age_norm = (building_age - building_age.mean()) / building_age.std()
        # bg trend normalized
        bg_trend_norm = (
            city_units[["census_bg_id"]]
            .join(bg_trend_norm_by_bg, how="left", on="census_bg_id")
            .bg_trend
        )
        # sum of normalized score
        normalized_score = (-building_age_norm) + bg_trend_norm
        # sorted by the score from high to low
        normalized_score = normalized_score.sort_values(
            ascending=False, ignore_index=False
        )
        # apply sorted index back to city_units
        city_units = city_units.loc[normalized_score.index]
        # .sort_values(by='building_age', ascending=True)
        # pick the top k units
        growth = mcd_growth.loc[city]
        selected_units = city_units.iloc[:growth]
        new_units = pd.concat([new_units, selected_units])
    # add mcd model quota to building table
    quota = new_units.index.value_counts()
    mcd_model_quota = pd.Series(0, index=blds.index)
    mcd_model_quota.loc[quota.index] = quota.values
    buildings.update_col_from_series("mcd_model_quota", mcd_model_quota, cast=True)


@orca.step()
def diagnostic(parcels, buildings, jobs, households, nodes, iter_var):
    parcels = parcels.to_frame()
    buildings = buildings.to_frame()
    jobs = jobs.to_frame()
    households = households.to_frame()
    nodes = nodes.to_frame()
    import pdb

    pdb.set_trace()


def make_repm_func(model_name, yaml_file, dep_var):
    """
    Generator function for single-model REPMs.
    """

    @orca.step(model_name)
    def func():
        buildings = orca.get_table("buildings")
        nodes_walk = orca.get_table("nodes_walk")
        print(yaml_file)
        return utils.hedonic_simulate(yaml_file, buildings, nodes_walk, dep_var)

    return func


repm_step_names = []
for repm_config in os.listdir("./configs/repm"):
    model_name = repm_config.split(".")[0]

    if repm_config.startswith("res"):
        dep_var = "sqft_price_res"
    elif repm_config.startswith("nonres"):
        dep_var = "sqft_price_nonres"

    make_repm_func(model_name, "repm/" + repm_config, dep_var)
    repm_step_names.append(model_name)
orca.add_injectable("repm_step_names", repm_step_names)


@orca.step()
def increase_property_values(buildings, income_growth_rates):
    # Hack to make more feasibility
    # dfinc = pd.read_csv("income_growth_rates.csv", index_col=['year'])
    dfinc = income_growth_rates.to_frame().loc[: orca.get_injectable("year")]
    dfrates = (
        dfinc.prod().to_frame(name="cumu_rates").fillna(1.0)
    )  # get cumulative increase from base to current year
    dfrates.index = dfrates.index.astype(float)
    bd = buildings.to_frame(["large_area_id", "sqft_price_res", "sqft_price_nonres"])
    bd = pd.merge(bd, dfrates, left_on="large_area_id", right_index=True, how="left")
    buildings.update_col_from_series(
        "sqft_price_res", bd.sqft_price_res * bd.cumu_rates
    )
    buildings.update_col_from_series(
        "sqft_price_nonres", bd.sqft_price_nonres * bd.cumu_rates
    )


@orca.step()
def households_relocation(households, annual_relocation_rates_for_households):
    relocation_rates = annual_relocation_rates_for_households.to_frame()
    relocation_rates = relocation_rates.rename(
        columns={"age_max": "age_of_head_max", "age_min": "age_of_head_min"}
    )
    relocation_rates.probability_of_relocating *= 0.2
    reloc = relocation.RelocationModel(relocation_rates, "probability_of_relocating")
    _print_number_unplaced(households, "building_id")
    print("un-placing")
    hh = households.to_frame(households.local_columns)
    idx_reloc = reloc.find_movers(hh)
    households.update_col_from_series(
        "building_id", pd.Series(-1, index=idx_reloc), cast=True
    )
    _print_number_unplaced(households, "building_id")


@orca.step()
def jobs_relocation(jobs, annual_relocation_rates_for_jobs):
    relocation_rates = annual_relocation_rates_for_jobs.to_frame().reset_index()
    reloc = relocation.RelocationModel(relocation_rates, "job_relocation_probability")
    _print_number_unplaced(jobs, "building_id")
    print("un-placing")
    j = jobs.to_frame(jobs.local_columns)
    idx_reloc = reloc.find_movers(j[j.home_based_status <= 0])
    jobs.update_col_from_series(
        "building_id", pd.Series(-1, index=idx_reloc), cast=True
    )
    _print_number_unplaced(jobs, "building_id")


def presses_trans(xxx_todo_changeme1):
    (ct, hh, p, target, iter_var) = xxx_todo_changeme1
    ct_finite = ct[ct.persons_max <= 100]
    ct_inf = ct[ct.persons_max > 100]
    tran = transition.TabularTotalsTransition(ct_finite, "total_number_of_households")
    model = transition.TransitionModel(tran)
    new, added_hh_idx, new_linked = model.transition(
        hh, iter_var, linked_tables={"linked": (p, "household_id")}
    )
    new.loc[added_hh_idx, "building_id"] = -1
    pers = new_linked["linked"]
    pers = pers[pers.household_id.isin(new.index)]
    out = [[new, pers]]
    target -= len(pers)
    best_qal = np.inf
    best = []
    for _ in range(3):
        tran = transition.TabularTotalsTransition(ct_inf, "total_number_of_households")
        model = transition.TransitionModel(tran)
        new, added_hh_idx, new_linked = model.transition(
            hh, iter_var, linked_tables={"linked": (p, "household_id")}
        )
        new.loc[added_hh_idx, "building_id"] = -1
        pers = new_linked["linked"]
        pers = pers[pers.household_id.isin(new.index)]
        qal = abs(target - len(pers))
        if qal < best_qal:
            best = (new, pers)
            best_qal = qal
    out.append(best)
    return out


@orca.step()
def households_transition(
    households, persons, annual_household_control_totals, remi_pop_total, iter_var
):
    region_ct = annual_household_control_totals.to_frame()
    max_cols = region_ct.columns[
        region_ct.columns.str.endswith("_max") & (region_ct == -1).any(axis=0)
    ]
    region_ct[max_cols] = region_ct[max_cols].replace(-1, np.inf)
    region_ct[max_cols] += 1
    region_hh = households.to_frame(households.local_columns + ["large_area_id"])
    region_hh.index = region_hh.index.astype(int)
    region_p = persons.to_frame(persons.local_columns)
    region_p.index = region_p.index.astype(int)
    region_target = remi_pop_total.to_frame()

    def cut_to_la(xxx_todo_changeme):
        (large_area_id, hh) = xxx_todo_changeme
        p = region_p[region_p.household_id.isin(hh.index)]
        target = int(region_target.loc[large_area_id, str(iter_var)])
        ct = region_ct[region_ct.large_area_id == large_area_id]
        del ct["large_area_id"]
        return ct, hh, p, target, iter_var

    arg_per_la = list(map(cut_to_la, region_hh.groupby("large_area_id")))
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
    for hh, p in out:
        hh.index.name = "household_id"
        hh = hh.reset_index()
        hh["household_id_old"] = hh["household_id"]
        new_hh = (hh.building_id == -1).sum()
        hh.loc[hh.building_id == -1, "household_id"] = list(
            range(hhidmax, hhidmax + new_hh)
        )
        hhidmax += new_hh
        hhid_map = hh[["household_id_old", "household_id"]].set_index(
            "household_id_old"
        )
        p.index.name = "person_id"
        p = pd.merge(
            p.reset_index(), hhid_map, left_on="household_id", right_index=True
        )
        new_p = (p.household_id_x != p.household_id_y).sum()
        p.loc[p.household_id_x != p.household_id_y, "person_id"] = list(
            range(pidmax, pidmax + new_p)
        )
        pidmax += new_p
        p["household_id"] = p["household_id_y"]

        new_hh_df = hh.set_index("household_id")
        out_hh_fixed.append(new_hh_df[households.local_columns])
        out_p_fixed.append(p.set_index("person_id")[persons.local_columns])

    orca.add_table("households", pd.concat(out_hh_fixed, verify_integrity=True))
    orca.add_table("persons", pd.concat(out_p_fixed, verify_integrity=True))


@orca.step()
def fix_lpr(households, persons, iter_var, workers_employment_rates_by_large_area):
    from numpy.random import choice

    hh = households.to_frame(households.local_columns + ["large_area_id"])
    hh["target_workers"] = 0
    p = persons.to_frame(persons.local_columns + ["large_area_id"])
    lpr = workers_employment_rates_by_large_area.to_frame(
        ["age_min", "age_max", str(iter_var)]
    )
    employed = p.worker == True
    p["weight"] = 1.0 / np.sqrt(p.join(hh["workers"], "household_id").workers + 1.0)

    colls = [
        "persons",
        "race_id",
        "workers",
        "children",
        "large_area_id",
    ]  # , 'age_of_head'
    same = {tuple(idx): df[["income", "cars"]] for idx, df in hh.groupby(colls)}

    new_employ = []
    new_unemploy = []

    for large_area_id, row in lpr.iterrows():
        select = (
            (p.large_area_id == large_area_id)
            & (p.age >= row.age_min)
            & (p.age <= row.age_max)
        )
        lpr = row[str(iter_var)]
        lpr_workers = int(select.sum() * lpr)
        num_workers = (select & employed).sum()

        if lpr_workers > num_workers:
            # employ some persons
            new_employ.append(
                choice(
                    p[select & (~employed)].index, int(lpr_workers - num_workers), False
                )
            )
        else:
            # unemploy some persons
            prob = p[select & employed].weight
            prob /= prob.sum()
            new_unemploy.append(
                choice(
                    p[select & employed].index,
                    int(num_workers - lpr_workers),
                    False,
                    p=prob,
                )
            )

        # print large_area_id, row.age_min, row.age_max, select.sum(), num_workers, lpr_workers, lpr

    if len(new_employ) > 0:
        p.loc[np.concatenate(new_employ), "worker"] = 1
    if len(new_unemploy):
        p.loc[np.concatenate(new_unemploy), "worker"] = 0

    hh["old_workers"] = hh.workers
    hh.workers = p.groupby("household_id").worker.sum()
    hh.workers = hh.workers.fillna(0)
    changed = hh.workers != hh.old_workers

    for match_colls, chh in hh[changed].groupby(colls):
        try:
            match = same[tuple(match_colls)]
            new_workers = choice(match.index, len(chh), True)
            hh.loc[chh.index, ["income", "cars"]] = match.loc[
                new_workers, ["income", "cars"]
            ].values
        except KeyError:
            pass
            # todo: something better!?

    orca.add_table("households", hh[households.local_columns])
    orca.add_table("persons", p[persons.local_columns])


@orca.step()
def jobs_transition(jobs, annual_employment_control_totals, iter_var):
    ct_emp = annual_employment_control_totals.to_frame()
    ct_emp = ct_emp.reset_index().set_index("year")
    tran = transition.TabularTotalsTransition(ct_emp, "total_number_of_jobs")
    model = transition.TransitionModel(tran)
    j = jobs.to_frame(jobs.local_columns + ["large_area_id"])
    new, added_jobs_idx, _ = model.transition(j, iter_var)
    orca.add_injectable(
        "jobs_large_area_lookup", new.large_area_id, autocall=False, cache=True
    )
    new.loc[added_jobs_idx, "building_id"] = -1
    orca.add_table("jobs", new[jobs.local_columns])


@orca.step()
def jobs_scaling_model(jobs):
    wrap_jobs = jobs
    jobs = jobs.to_frame(jobs.local_columns + ["large_area_id"])
    regional_sectors = {1, 7, 12, 13, 15, 18}
    la_sectors = []

    def random_choice(chooser_ids, alternative_ids, probabilities):
        return pd.Series(
            np.random.choice(
                alternative_ids, size=len(chooser_ids), replace=True, p=probabilities
            ),
            index=chooser_ids,
        )

    jobs_to_place = jobs[jobs.building_id.isnull() | (jobs.building_id == -1)]
    selected = jobs_to_place.sector_id.isin(regional_sectors)
    for (sec, la) in la_sectors:
        selected |= (jobs_to_place.sector_id == sec) & (
            jobs_to_place.large_area_id == la
        )
    jobs_to_place = jobs_to_place[selected]

    if len(jobs_to_place) > 0:
        for (large_area_id, sector), segment in jobs_to_place.groupby(
            ["large_area_id", "sector_id"]
        ):
            counts_by_bid = (
                jobs[(jobs.sector_id == sector) & (jobs.large_area_id == large_area_id)]
                .groupby(["building_id"])
                .size()
            )
            prop_by_bid = counts_by_bid / counts_by_bid.sum()
            choices = random_choice(
                segment.index.values, prop_by_bid.index.values, prop_by_bid.values
            )
            wrap_jobs.update_col_from_series("building_id", choices, cast=True)


@orca.step()
def gq_pop_scaling_model(group_quarters, group_quarters_control_totals, year):
    gqpop = group_quarters.to_frame(group_quarters.local_columns + ["city_id"])
    target_gq = group_quarters_control_totals.to_frame()
    target_gq = target_gq[target_gq.year == year].set_index("city_id")

    for city_id, local_gqpop in gqpop.groupby("city_id"):
        diff = target_gq.loc[city_id]["count"] - len(local_gqpop)
        protected = (
            ((local_gqpop.gq_code > 100) & (local_gqpop.gq_code < 200))
            | ((local_gqpop.gq_code > 500) & (local_gqpop.gq_code < 600))
            | (local_gqpop.gq_code == 701)
        )
        local_gqpop = local_gqpop[~protected]
        if diff > 0:
            diff = int(min(len(local_gqpop), abs(diff)))
            if diff > 0:
                newgq = local_gqpop.sample(diff, replace=True)
                newgq.index = gqpop.index.values.max() + 1 + np.arange(len(newgq))
                gqpop = gqpop.append(newgq)

        elif diff < 0:
            diff = min(len(local_gqpop), abs(diff))
            if diff > 0:
                removegq = local_gqpop.sample()
                gqpop.drop(removegq.index, inplace=True)

    orca.add_table("group_quarters", gqpop[group_quarters.local_columns])


@orca.step()
def refiner(jobs, households, buildings, persons, year, refiner_events, group_quarters):
    location_ids = ["b_zone_id", "zone_id", "b_city_id", "city_id", "large_area_id"]
    jobs_columns = jobs.local_columns
    jobs = jobs.to_frame(jobs_columns + location_ids)
    group_quarters_columns = group_quarters.local_columns
    group_quarters = group_quarters.to_frame(group_quarters_columns + location_ids)
    households_columns = households.local_columns
    households = households.to_frame(households_columns + location_ids)
    households["household_id_old"] = households.index.values
    buildings = buildings.to_frame(
        buildings.local_columns + location_ids + ["gq_building"]
    )
    dic_agent = {"jobs": jobs, "households": households, "gq": group_quarters}

    refinements = refiner_events.to_frame()
    refinements = refinements[refinements.year == year]
    assert refinements.action.isin(
        {"clone", "subtract_pop", "subtract", "add_pop", "add", "target_pop", "target"}
    ).all(), "Unknown action"
    assert refinements.agents.isin({"jobs", "households", "gq"}).all(), "Unknown agents"

    def add_agents(
        agents, agents_pool, agent_expression, location_expression, number_of_agents
    ):
        """Move from pool to data"""
        bselect = buildings.query(location_expression)
        if len(bselect) <= 0:
            print("We can't find a building to place these agents")
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
            agents_sample.index = (
                agents.index.values.max() + 1 + np.arange(len(agents_sample))
            )
        else:
            agents_sample = agents.query(agent_expression)
            if len(agents_sample) > 0:
                agents_sample = agents_sample.sample(number_of_agents, replace=True)
                agents_sample.index = (
                    agents.index.values.max() + 1 + np.arange(len(agents_sample))
                )
                agents_sample.building_id = new_building_ids

        agents = agents.append(agents_sample)
        return agents, agents_pool

    def add_pop_agents(
        agents, agents_pool, agent_expression, location_expression, number_of_agents
    ):
        """Move from pool to data"""
        bselect = buildings.query(location_expression)
        if len(bselect) <= 0:
            print("We can't fined a building to place these agents")
            return agents, agents_pool

        if len(agents_pool) > 0:
            available_agents = agents_pool.query(agent_expression)
            available_agents = available_agents[
                available_agents.persons <= number_of_agents
            ]
            while len(available_agents) > 0 and number_of_agents > 0:
                available_agents = available_agents[
                    available_agents.persons <= number_of_agents
                ]

                if len(available_agents) >= number_of_agents:
                    agents_sample = available_agents.sample(
                        number_of_agents, replace=False
                    )
                else:
                    agents_sample = available_agents.sample(
                        number_of_agents, replace=True
                    )

                agents_sample = agents_sample[
                    agents_sample.persons.cumsum() <= number_of_agents
                ]
                agents_sample.index = (
                    agents.index.values.max() + 1 + np.arange(len(agents_sample))
                )
                agents_sample.building_id = bselect.sample(
                    len(agents_sample), replace=True
                ).index.values
                agents = agents.append(agents_sample)
                number_of_agents -= agents_sample.persons.sum()
        else:
            available_agents = agents.query(agent_expression)
            available_agents = available_agents[
                available_agents.persons <= number_of_agents
            ]
            while len(available_agents) > 0 and number_of_agents > 0:
                available_agents = available_agents[
                    available_agents.persons <= number_of_agents
                ]
                agents_sample = available_agents.sample(number_of_agents, replace=True)
                agents_sample = agents_sample[
                    agents_sample.persons.cumsum() <= number_of_agents
                ]
                agents_sample.index = (
                    agents.index.values.max() + 1 + np.arange(len(agents_sample))
                )
                agents_sample.building_id = bselect.sample(
                    len(agents_sample), replace=True
                ).index.values
                agents = agents.append(agents_sample)
                number_of_agents -= agents_sample.persons.sum()
        return agents, agents_pool

    def subtract_agents(
        agents, agents_pool, agent_expression, location_expression, number_of_agents
    ):
        """Move from data to pool"""
        # TODO: Share code with clone_agents then call drop
        available_agents = agents.query(agent_expression)
        bselect = buildings.query(location_expression)
        local_agents = available_agents.loc[
            available_agents.building_id.isin(bselect.index.values)
        ]
        if len(local_agents) > 0 and number_of_agents > 0:
            selected_agents = local_agents.sample(
                min(len(local_agents), number_of_agents)
            )
            agents_pool = agents_pool.append(selected_agents, ignore_index=True)
            agents.drop(selected_agents.index, inplace=True)
        return agents, agents_pool

    def subtract_pop_agents(
        agents, agents_pool, agent_expression, location_expression, number_of_agents
    ):
        """Move from data to pool"""
        available_agents = agents.query(agent_expression)
        bselect = buildings.query(location_expression)
        local_agents = available_agents.loc[
            available_agents.building_id.isin(bselect.index.values)
        ]
        local_agents = local_agents[local_agents.persons <= number_of_agents]
        while len(local_agents) > 0 and number_of_agents > 0:
            local_agents = local_agents[local_agents.persons <= number_of_agents]
            selected_agents = local_agents.sample(
                min(len(local_agents), number_of_agents)
            )
            selected_agents = selected_agents[
                selected_agents.persons.cumsum() <= number_of_agents
            ]
            number_of_agents -= selected_agents.persons.sum()
            agents_pool = agents_pool.append(selected_agents, ignore_index=True)
            local_agents.drop(selected_agents.index, inplace=True)
            agents.drop(selected_agents.index, inplace=True)
        return agents, agents_pool

    def clone_agents(
        agents, agents_pool, agent_expression, location_expression, number_of_agents
    ):
        """Copy from data to pool. Don't remove from data!"""
        available_agents = agents.query(agent_expression)
        bselect = buildings.query(location_expression)
        local_agents = available_agents.loc[
            available_agents.building_id.isin(bselect.index.values)
        ]
        if len(local_agents) > 0:
            selected_agents = local_agents.sample(
                min(len(local_agents), number_of_agents)
            )
            agents_pool = agents_pool.append(selected_agents, ignore_index=True)
        return agents, agents_pool

    def target_agents(
        agents, agent_expression, location_expression, number_of_agents, by_pop=False
    ):
        """Determine whether to add or subtract based on data"""
        #  use for employment event model
        exist_agents = agents.query(agent_expression)
        bselect = buildings.query(location_expression)
        local_agents = exist_agents.loc[
            exist_agents.building_id.isin(bselect.index.values)
        ]

        if by_pop:
            return local_agents.persons.sum() - number_of_agents
        return len(local_agents) - number_of_agents

    for tid, trecords in refinements.groupby("transaction_id"):
        print("** processing transcaction ", tid)
        agent_types = trecords.agents.drop_duplicates()
        assert len(agent_types) == 1, "different agents in same transaction_id"
        agent_type = agent_types.iloc[0]
        agents = dic_agent[agent_type]
        pool = pd.DataFrame(data=None, columns=agents.columns)

        for _, record in trecords[trecords.action == "clone"].iterrows():
            print(record)
            agents, pool = clone_agents(
                agents,
                pool,
                record.agent_expression,
                record.location_expression,
                record.amount,
            )

        for _, record in trecords[trecords.action == "subtract_pop"].iterrows():
            print(record)
            assert agent_type == "households"
            agents, pool = subtract_pop_agents(
                agents,
                pool,
                record.agent_expression,
                record.location_expression,
                record.amount,
            )

        for _, record in trecords[trecords.action == "subtract"].iterrows():
            print(record)
            agents, pool = subtract_agents(
                agents,
                pool,
                record.agent_expression,
                record.location_expression,
                record.amount,
            )

        for _, record in trecords[trecords.action == "add_pop"].iterrows():
            print(record)
            assert agent_type == "households"
            agents, pool = add_pop_agents(
                agents,
                pool,
                record.agent_expression,
                record.location_expression,
                record.amount,
            )

        for _, record in trecords[trecords.action == "add"].iterrows():
            print(record)
            agents, pool = add_agents(
                agents,
                pool,
                record.agent_expression,
                record.location_expression,
                record.amount,
            )

        for _, record in trecords[trecords.action == "target_pop"].iterrows():
            print(record)
            assert agent_type == "households"
            diff = target_agents(
                dic_agent[record.agents],
                record.agent_expression,
                record.location_expression,
                record.amount,
                by_pop=True,
            )
            if diff < 0:
                agents, pool = add_pop_agents(
                    agents,
                    pool,
                    record.agents_expression,
                    record.location_expression,
                    abs(diff),
                )
            elif diff > 0:
                agents, pool = subtract_pop_agents(
                    agents,
                    pool,
                    record.agent_expression,
                    record.location_expression,
                    diff,
                )

        for _, record in trecords[trecords.action == "target"].iterrows():
            print(record)
            diff = target_agents(
                dic_agent[record.agents],
                record.agent_expression,
                record.location_expression,
                record.amount,
            )
            if diff < 0:
                print("add: ", abs(diff))
                agents, pool = add_agents(
                    agents,
                    pool,
                    record.agent_expression,
                    record.location_expression,
                    abs(diff),
                )
            elif diff > 0:
                print("subtract: ", abs(diff))
                agents, pool = subtract_agents(
                    agents,
                    pool,
                    record.agent_expression,
                    record.location_expression,
                    diff,
                )
        dic_agent[agent_type] = agents

    if refinements.agents.isin({"jobs"}).sum() > 0:
        jobs = dic_agent["jobs"]
        assert jobs.index.duplicated().sum() == 0, "duplicated index in jobs"
        jobs["large_area_id"] = misc.reindex(buildings.large_area_id, jobs.building_id)
        orca.add_table("jobs", jobs[jobs_columns])

    if refinements.agents.isin({"gq"}).sum() > 0:
        group_quarters = dic_agent["gq"]
        assert (
            group_quarters.index.duplicated().sum() == 0
        ), "duplicated index in group_quarters"
        orca.add_table("group_quarters", group_quarters[group_quarters_columns])

    if refinements.agents.isin({"households"}).sum() > 0:
        households = dic_agent["households"]
        assert (
            households.index.duplicated().sum() == 0
        ), "duplicated index in households"
        households["large_area_id"] = misc.reindex(
            buildings.large_area_id, households.building_id
        )
        orca.add_table("households", households[households_columns])

        persons_columns = persons.local_columns
        persons = persons.to_frame(persons_columns)
        pidmax = persons.index.values.max() + 1

        hh_index_lookup = (
            households[["household_id_old"]].reset_index().set_index("household_id_old")
        )
        hh_index_lookup.columns = ["household_id"]
        p = pd.merge(
            persons.reset_index(),
            hh_index_lookup,
            left_on="household_id",
            right_index=True,
        )
        new_p = (p.household_id_x != p.household_id_y).sum()
        p.loc[p.household_id_x != p.household_id_y, "person_id"] = list(
            range(pidmax, pidmax + new_p)
        )
        p["household_id"] = p["household_id_y"]
        persons = p.set_index("person_id")

        assert persons.household_id.isin(
            households.index
        ).all(), "persons.household_id not in households"
        assert len(persons.groupby("household_id").size()) == len(
            households.persons
        ), "households with no persons"
        assert persons.index.duplicated().sum() == 0, "duplicated index in persons"
        orca.add_table("persons", persons[persons_columns])


@orca.step()
def scheduled_development_events(buildings, iter_var, events_addition):
    sched_dev = events_addition.to_frame()
    sched_dev = sched_dev[sched_dev.year_built == iter_var].reset_index(drop=True)
    if len(sched_dev) > 0:
        sched_dev["stories"] = 0
        zone = (
            sched_dev.b_zone_id
        )  # save buildings based zone and city ids for later updates. model could update columns using parcel zone and city ids.
        city = sched_dev.b_city_id
        ebid = sched_dev.event_bid.copy()  # save event_bid to be used later
        sched_dev = add_extra_columns_res(sched_dev)
        sched_dev["b_zone_id"] = zone
        sched_dev["b_city_id"] = city
        sched_dev["event_bid"] = ebid  # add back event_bid
        b = buildings.to_frame(buildings.local_columns)

        all_buildings = parcel_utils.merge_buildings(b, sched_dev[b.columns], False)

        orca.add_table("buildings", all_buildings)

        # Todo: maybe we need to impute some columns
        # Todo: parcel use need to be updated
        # Todo: record dev_id -> building_id


@orca.step()
def scheduled_demolition_events(buildings, households, jobs, iter_var, events_deletion):
    sched_dev = events_deletion.to_frame()
    sched_dev = sched_dev[sched_dev.year_built == iter_var].reset_index(drop=True)
    if len(sched_dev) > 0:
        buildings = buildings.to_frame(buildings.local_columns)
        drop_buildings = buildings[buildings.index.isin(sched_dev.building_id)].copy()
        buildings_idx = drop_buildings.index
        drop_buildings["year_demo"] = iter_var

        if orca.is_table("dropped_buildings"):
            prev_drops = orca.get_table("dropped_buildings").to_frame()
            orca.add_table("dropped_buildings", pd.concat([drop_buildings, prev_drops]))
        else:
            orca.add_table("dropped_buildings", drop_buildings)

        orca.add_table("buildings", buildings.drop(buildings_idx))

        # unplace HH
        # todo: use orca.update_col_from_series
        households = households.to_frame(households.local_columns)
        households.loc[
            households.building_id.isin(sched_dev.building_id), "building_id"
        ] = -1
        orca.add_table("households", households)

        # unplace jobs
        # todo: use orca.update_col_from_series
        jobs = jobs.to_frame(jobs.local_columns)
        jobs.loc[jobs.building_id.isin(sched_dev.building_id), "building_id"] = -1
        orca.add_table("jobs", jobs)
        # Todo: parcel use need to be updated


@orca.step()
def random_demolition_events(buildings, households, jobs, year, demolition_rates):
    demolition_rates = demolition_rates.to_frame()
    demolition_rates *= 0.1 + (1.0 - 0.1) * (2045 - year) / (2045 - 2015)
    buildings_columns = buildings.local_columns
    buildings = buildings.to_frame(
        buildings.local_columns + ["b_total_jobs", "b_total_households"]
    )
    b = buildings.copy()
    allowed = variables.parcel_is_allowed()
    b = b[b.parcel_id.isin(allowed[allowed].index)]
    buildings_idx = []

    def sample(targets, type_b, accounting, weights):
        for b_city_id, target in targets[targets > 0].items():
            rel_b = type_b[type_b.b_city_id == b_city_id]
            rel_b = rel_b[rel_b[accounting] <= target]
            size = min(len(rel_b), int(target))
            if size > 0:
                rel_b = rel_b.sample(size, weights=rel_b[weights])
                rel_b = rel_b[rel_b[accounting].cumsum() <= int(target)]
                buildings_idx.append(rel_b.copy())

    b["wj"] = 1.0 / (1.0 + np.log1p(b.b_total_jobs))
    sample(
        demolition_rates.typenonsqft,
        b[b.non_residential_sqft > 0],
        "non_residential_sqft",
        "wj",
    )
    b = b[b.non_residential_sqft == 0]
    b["wh"] = 1.0 / (1.0 + np.log1p(b.b_total_households))
    sample(
        demolition_rates.type81units,
        b[b.building_type_id == 81],
        "residential_units",
        "wh",
    )
    sample(
        demolition_rates.type82units,
        b[b.building_type_id == 82],
        "residential_units",
        "wh",
    )
    sample(
        demolition_rates.type83units,
        b[b.building_type_id == 83],
        "residential_units",
        "wh",
    )
    # sample(demolition_rates.type84units, b[b.building_type_id == 84], 'residential_units', 'wh')

    drop_buildings = pd.concat(buildings_idx).copy()[buildings_columns]
    drop_buildings = drop_buildings[~drop_buildings.index.duplicated(keep="first")]
    buildings_idx = drop_buildings.index
    drop_buildings["year_demo"] = year

    if orca.is_table("dropped_buildings"):
        prev_drops = orca.get_table("dropped_buildings").to_frame()
        orca.add_table("dropped_buildings", pd.concat([drop_buildings, prev_drops]))
    else:
        orca.add_table("dropped_buildings", drop_buildings)

    orca.add_table("buildings", buildings[buildings_columns].drop(buildings_idx))

    # unplace HH
    # todo: use orca.update_col_from_series
    households = households.to_frame(households.local_columns)
    households.loc[households.building_id.isin(buildings_idx), "building_id"] = -1
    orca.add_table("households", households)

    # unplace jobs
    # todo: use orca.update_col_from_series
    jobs = jobs.to_frame(jobs.local_columns)
    jobs.loc[jobs.building_id.isin(buildings_idx), "building_id"] = -1
    orca.add_table("jobs", jobs)
    # Todo: parcel use need to be updated


def parcel_average_price(use):
    # Copied from variables.py
    parcels_wrapper = orca.get_table("parcels")
    if len(orca.get_table("nodes_walk")) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels_wrapper.index)
    return misc.reindex(orca.get_table("nodes_walk")[use], parcels_wrapper.nodeid_walk)


@orca.injectable("cost_shifters")
def shifters():
    with open(os.path.join(misc.configs_dir(), "cost_shifters.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cfg


def cost_shifter_callback(self, form, df, costs):
    shifter_cfg = orca.get_injectable("cost_shifters")["calibration"]
    geography = shifter_cfg["calibration_geography_id"]
    shift_type = "residential" if form == "residential" else "non_residential"
    shifters = shifter_cfg["proforma_cost_shifters"][shift_type]

    for geo, geo_df in df.reset_index().groupby(geography):
        shifter = shifters[geo]
        costs[:, geo_df.index] *= shifter
    return costs


@orca.step("feasibility")
def feasibility(parcels):
    parcel_utils.run_feasibility(
        parcels,
        parcel_average_price,
        variables.parcel_is_allowed,
        cfg="proforma.yaml",
        modify_costs=cost_shifter_callback,
    )
    feasibility = orca.get_table("feasibility").to_frame()
    for mcdid, df in parcels.semmcd.to_frame().groupby("semmcd"):
        orca.add_table(
            "feasibility_" + str(mcdid), feasibility[feasibility.index.isin(df.index)]
        )


def add_extra_columns_nonres(df):
    # type: (pd.DataFrame) -> pd.DataFrame
    for col in [
        "improvement_value",
        "land_area",
        "tax_exempt",
        "sqft_price_nonres",
        "sqft_price_res",
        "sqft_per_unit",
        "hu_filter",
        "event_bid",
    ]:
        df[col] = 0
    df["year_built"] = orca.get_injectable("year")
    p = orca.get_table("parcels").to_frame(["zone_id", "city_id"])
    for col in ["zone_id", "city_id"]:
        df["b_" + col] = misc.reindex(p[col], df.parcel_id)
    return df.fillna(0)


def add_extra_columns_res(df):
    # type: (pd.DataFrame) -> pd.DataFrame
    df = add_extra_columns_nonres(df)
    if "ave_unit_size" in df.columns:
        df["sqft_per_unit"] = df["ave_unit_size"]
    else:
        df["sqft_per_unit"] = misc.reindex(
            orca.get_table("parcels").ave_unit_size, df.parcel_id
        )
    return df


def probable_type(row):
    """
    Function to pass to form_to_btype_callback in parcels_utils.add_buildings.
    Reads form for the given row in new buildings DataFrame, gets the
    building type probabilities from the form_btype_distributions injectable,
    and chooses a building type based on those probabilities.

    Parameters
    ----------
    row : Series

    Returns
    -------
    btype : int
    """
    form = row["form"]
    form_to_btype_dists = orca.get_injectable("form_btype_distributions")
    btype_dists = form_to_btype_dists[form]
    # keys() and values() guaranteed to be in same order
    btype = np.random.choice(a=list(btype_dists.keys()), p=list(btype_dists.values()))
    return btype


def register_btype_distributions(buildings):
    """
    Prior to adding new buildings selected by the developer model, this
    function registers a dictionary where keys are forms, and values are
    dictionaries of building types. In each sub-dictionary, the keys are
    building type IDs, and values are probabilities. These probabilities are
    generated from distributions of building types for each form in each
    large area.

    Parameters
    ----------
    buildings : DataFrame
        Buildings DataFrame (not wrapper) at the time of registration. Must
        include building_type_id column.

    Returns
    -------
    None
    """
    form_to_btype = orca.get_injectable("form_to_btype")
    form_btype_dists = {}
    for form in list(form_to_btype.keys()):
        bldgs = buildings.loc[buildings.building_type_id.isin(form_to_btype[form])]
        bldgs_by_type = bldgs.groupby("building_type_id").size()
        normed = bldgs_by_type / sum(bldgs_by_type)
        form_btype_dists[form] = normed.to_dict()

    orca.add_injectable("form_btype_distributions", form_btype_dists)


def run_developer(
    target_units,
    geoid,
    forms,
    buildings,
    supply_fname,
    parcel_size,
    ave_unit_size,
    current_units,
    cfg,
    add_more_columns_callback=None,
    unplace_agents=("households", "jobs"),
    profit_to_prob_func=None,
    custom_selection_func=None,
    pipeline=False,
):
    """
    copied form parcel_utils and modified
    """
    from developer import develop

    print(f"developing {str(forms)} for geography {geoid}")
    cfg = misc.config(cfg)
    dev = develop.Developer.from_yaml(
        orca.get_table("feasibility_" + str(geoid)).to_frame(),
        forms,
        target_units,
        parcel_size,
        ave_unit_size,
        current_units,
        orca.get_injectable("year"),
        str_or_buffer=cfg,
    )

    print(
        (
            "{:,} feasible buildings before running developer".format(
                len(dev.feasibility)
            )
        )
    )

    new_buildings = dev.pick(profit_to_prob_func, custom_selection_func)
    orca.add_table("feasibility_" + str(geoid), dev.feasibility)

    if new_buildings is None or len(new_buildings) == 0:
        return

    parcel_utils.add_buildings(
        dev.feasibility,
        buildings,
        new_buildings,
        probable_type,
        add_more_columns_callback,
        supply_fname,
        True,
        unplace_agents,
        pipeline,
    )


# @orca.step("residential_developer")
# def residential_developer(households, parcels, target_vacancies):
#     target_vacancies = target_vacancies.to_frame()
#     target_vacancies = target_vacancies[
#         target_vacancies.year == orca.get_injectable("year")
#     ]
#     orig_buildings = orca.get_table("buildings").to_frame(
#         ["residential_units", "large_area_id", "building_type_id"]
#     )
#     for lid, _ in parcels.large_area_id.to_frame().groupby("large_area_id"):
#         la_orig_buildings = orig_buildings[orig_buildings.large_area_id == lid]
#         target_vacancy = float(
#             target_vacancies[
#                 target_vacancies.large_area_id == lid
#             ].res_target_vacancy_rate
#         )
#         num_agents = (households.large_area_id == lid).sum()
#         num_units = la_orig_buildings.residential_units.sum()

#         print("Number of agents: {:,}".format(num_agents))
#         print("Number of agent spaces: {:,}".format(int(num_units)))
#         assert target_vacancy < 1.0
#         target_units = int(max((num_agents / (1 - target_vacancy) - num_units), 0))
#         print("Current vacancy = {:.2f}".format(1 - num_agents / float(num_units)))
#         print(
#             "Target vacancy = {:.2f}, target of new units = {:,}".format(
#                 target_vacancy, target_units
#             )
#         )

#         register_btype_distributions(la_orig_buildings)
#         run_developer(
#             target_units,
#             lid,
#             "residential",
#             orca.get_table("buildings"),
#             "residential_units",
#             parcels.parcel_size,
#             parcels.ave_unit_size,
#             parcels.total_units,
#             "res_developer.yaml",
#             add_more_columns_callback=add_extra_columns_res,
#         )


@orca.step("residential_developer")
def residential_developer(households, parcels, target_vacancies):
    target_vacancies = target_vacancies.to_frame()
    target_vacancies = target_vacancies[
        target_vacancies.year == orca.get_injectable("year")
    ]
    orig_buildings = orca.get_table("buildings").to_frame(
        ["residential_units", "semmcd", "building_type_id"]
    )
    for mcdid, _ in parcels.semmcd.to_frame().groupby("semmcd"):
        mcd_orig_buildings = orig_buildings[orig_buildings.semmcd == mcdid]
        target_vacancy = float(
            target_vacancies[target_vacancies.semmcd == mcdid].res_target_vacancy_rate
        )
        num_agents = (households.semmcd == mcdid).sum()
        num_units = mcd_orig_buildings.residential_units.sum()

        print("Number of agents: {:,}".format(num_agents))
        print("Number of agent spaces: {:,}".format(int(num_units)))
        assert target_vacancy < 1.0
        target_units = int(max((num_agents / (1 - target_vacancy) - num_units), 0))
        print("Current vacancy = {:.2f}".format(1 - num_agents / float(num_units)))
        print(
            "Target vacancy = {:.2f}, target of new units = {:,}".format(
                target_vacancy, target_units
            )
        )

        register_btype_distributions(mcd_orig_buildings)
        run_developer(
            target_units,
            mcdid,
            "residential",
            orca.get_table("buildings"),
            "residential_units",
            parcels.parcel_size,
            parcels.ave_unit_size,
            parcels.total_units,
            "res_developer.yaml",
            add_more_columns_callback=add_extra_columns_res,
        )


@orca.step()
def non_residential_developer(jobs, parcels, target_vacancies):
    target_vacancies = target_vacancies.to_frame()
    target_vacancies = target_vacancies[
        target_vacancies.year == orca.get_injectable("year")
    ]
    orig_buildings = orca.get_table("buildings").to_frame(
        ["job_spaces", "large_area_id", "building_type_id"]
    )
    for lid, _ in parcels.large_area_id.to_frame().groupby("large_area_id"):
        la_orig_buildings = orig_buildings[orig_buildings.large_area_id == lid]
        target_vacancy = float(
            target_vacancies[
                target_vacancies.large_area_id == lid
            ].non_res_target_vacancy_rate
        )
        num_agents = ((jobs.large_area_id == lid) & (jobs.home_based_status == 0)).sum()
        num_units = la_orig_buildings.job_spaces.sum()

        print("Number of agents: {:,}".format(num_agents))
        print("Number of agent spaces: {:,}".format(int(num_units)))
        assert target_vacancy < 1.0
        target_units = int(max((num_agents / (1 - target_vacancy) - num_units), 0))
        print("Current vacancy = {:.2f}".format(1 - num_agents / float(num_units)))
        print(
            "Target vacancy = {:.2f}, target of new units = {:,}".format(
                target_vacancy, target_units
            )
        )

        register_btype_distributions(la_orig_buildings)
        run_developer(
            target_units,
            lid,
            ["office", "retail", "industrial", "medical"],
            orca.get_table("buildings"),
            "job_spaces",
            parcels.parcel_size,
            parcels.ave_unit_size,
            parcels.total_job_spaces,
            "nonres_developer.yaml",
            add_more_columns_callback=add_extra_columns_nonres,
        )


@orca.step()
def build_networks(parcels):
    import yaml

    pdna.network.reserve_num_graphs(2)

    # networks in semcog_networks.h5
    with open(r"configs/available_networks.yaml", "r") as stream:
        dic_net = yaml.load(stream, Loader=yaml.FullLoader)

    st = pd.HDFStore(os.path.join(misc.data_dir(), "semcog_networks_py3.h5"), "r")

    lstnet = [
        {
            "name": "mgf14_ext_walk",
            "cost": "cost1",
            "prev": 10560,  # 2 miles
            "net": "net_walk",
        },
        {
            "name": "tdm_ext",
            "cost": "cost1",
            "prev": 60,  # 60 minutes
            "net": "net_drv",
        },
    ]

    for n in lstnet:
        n_dic_net = dic_net[n["name"]]
        nodes, edges = st[n_dic_net["nodes"]], st[n_dic_net["edges"]]
        net = pdna.Network(
            nodes["x"],
            nodes["y"],
            edges["from"],
            edges["to"],
            edges[[n_dic_net[n["cost"]]]],
        )
        net.precompute(n["prev"])
        net.init_pois(num_categories=10, max_dist=n["prev"], max_pois=5)

        orca.add_injectable(n["net"], net)

    # spatially join node ids to parcels
    p = parcels.local
    p["nodeid_walk"] = orca.get_injectable("net_walk").get_node_ids(
        p["centroid_x"], p["centroid_y"]
    )
    p["nodeid_drv"] = orca.get_injectable("net_drv").get_node_ids(
        p["centroid_x"], p["centroid_y"]
    )
    orca.add_table("parcels", p)


@orca.step()
def neighborhood_vars(jobs, households, buildings):
    b = buildings.to_frame(["large_area_id"])
    j = jobs.to_frame(jobs.local_columns)
    h = households.to_frame(households.local_columns)
    idx_invalid_building_id = np.in1d(j.building_id, b.index.values) == False

    if idx_invalid_building_id.sum() > 0:
        print(
            (
                "we have jobs with bad building id's there are #",
                idx_invalid_building_id.sum(),
            )
        )
        j.loc[idx_invalid_building_id, "building_id"] = np.random.choice(
            b.index.values, idx_invalid_building_id.sum()
        )
        # TODO: keep LA the same
        j["large_area_id"] = misc.reindex(b.large_area_id, j.building_id)
        orca.add_table("jobs", j)
    idx_invalid_building_id = np.in1d(h.building_id, b.index.values) == False
    if idx_invalid_building_id.sum() > 0:
        print(
            (
                "we have households with bad building id's there are #",
                idx_invalid_building_id.sum(),
            )
        )
        h.loc[idx_invalid_building_id, "building_id"] = np.random.choice(
            b.index.values, idx_invalid_building_id.sum()
        )
        # TODO: keep LA the same
        j["large_area_id"] = misc.reindex(b.large_area_id, h.building_id)
        orca.add_table("households", h)

    building_vars = set(orca.get_table("buildings").columns)

    nodes = networks.from_yaml(orca.get_injectable("net_walk"), "networks_walk.yaml")
    # print nodes.describe()
    # print pd.Series(nodes.index).describe()
    orca.add_table("nodes_walk", nodes)
    # Disaggregate nodal variables to building.
    for var in orca.get_table("nodes_walk").columns:
        if var not in building_vars:
            variables.make_disagg_var("nodes_walk", "buildings", var, "nodeid_walk")

    nodes = networks.from_yaml(orca.get_injectable("net_drv"), "networks_drv.yaml")
    # print nodes.describe()
    # print pd.Series(nodes.index).describe()
    orca.add_table("nodes_drv", nodes)
    # Disaggregate nodal variables to building.
    for var in orca.get_table("nodes_drv").columns:
        if var not in building_vars:
            variables.make_disagg_var("nodes_drv", "buildings", var, "nodeid_drv")


def _print_number_unplaced(df, fieldname="building_id"):
    """
    Just an internal function to use to compute and print info on the number
    of unplaced agents.
    """
    counts = (df[fieldname] == -1).sum()
    print("Total currently unplaced: %d" % counts)
