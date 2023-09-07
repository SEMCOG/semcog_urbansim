import os
import time
import yaml
import operator
from multiprocessing import Pool
from collections import defaultdict
import pickle

import numpy as np
import orca
import pandana as pdna
import pandas as pd
from urbansim.models import transition, relocation
from urbansim.utils import misc, networks
from urbansim_parcels import utils as parcel_utils

import utils
import lcm_utils

import dataset
import variables
from functools import reduce

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
# run elcm by specific job_sector sequence defined below
elcm_sector_order = [3, 6, 10, 11, 14, 9, 4, 2, 5, 16, 17, 8]
elcm_sector_order = {sector: idx for idx, sector in enumerate(elcm_sector_order)}
orca.add_injectable(
    "elcm_step_names",
    sorted(elcm_step_names, key=lambda x: elcm_sector_order[int(x[5:]) // 100000]),
)

for name, model in list(location_choice_models.items()):
    lcm_utils.register_choice_model_step(model.name, model.choosers)


@orca.step()
def elcm_home_based(jobs, households):
    wrap_jobs = jobs
    _print_number_unplaced(wrap_jobs, "building_id")
    jobs = wrap_jobs.to_frame(["building_id", "home_based_status", "large_area_id"])
    jobs = jobs[(jobs.home_based_status >= 1) & (jobs.building_id == -1)]
    hh = households.to_frame(["building_id", "large_area_id", "sp_filter"])
    hh = hh[(hh.building_id > 0) & (hh.sp_filter >= 0)]

    for la, la_job in jobs.groupby("large_area_id"):
        la_hh = hh[hh.large_area_id == la]
        la_job["building_id"] = la_hh.sample(
            len(la_job), replace=True
        ).building_id.values
        wrap_jobs.update_col_from_series(
            "building_id", la_job["building_id"], cast=True
        )

    _print_number_unplaced(wrap_jobs, "building_id")


@orca.injectable("mcd_hu_sampling_config")
def mcd_hu_sampling_config():
    # load mcd_hu_sampling step config file
    with open(os.path.join(misc.configs_dir(), "mcd_hu_sampling.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cfg


@orca.step()
def mcd_hu_sampling(buildings, households, mcd_total, bg_hh_increase):
    """
    Apply the mcd total forecast to Limit and calculate the pool of housing 
    units to match the distribution of the mcd_total growth table for the MCD
    Parameters
    ----------
    buildings : orca.DataFrameWrapper
        Buildings table 
    households : orca.DataFrameWrapper
        Households table 
    mcd_total : orca.DataFrameWrapper
        MCD total table
    bg_hh_increase : orca.DataFrameWrapper
        hh growth trend by block groups
    Returns
    -------
    None
    """
    # get current year
    year = orca.get_injectable("year")

    # get config
    config = orca.get_injectable("mcd_hu_sampling_config")
    vacant_variable = config["vacant_variable"]

    # get housing unit table from buildings
    blds = buildings.to_frame(
        [
            "building_id",
            "semmcd",
            vacant_variable,
            "building_age",
            "geoid",
            "mcd_model_quota",
            "hu_filter",
            "sp_filter",
        ]
    )
    # get vacant units with index and value >0
    vacant_units = blds[vacant_variable]
    vacant_units = vacant_units[vacant_units.index.values >= 0]
    vacant_units = vacant_units[vacant_units > 0]

    # generate housing units from vacant units
    indexes = np.repeat(vacant_units.index.values, vacant_units.values.astype("int"))
    housing_units = blds.loc[indexes]

    # the mcd_total for year
    mcd_total = mcd_total.to_frame([str(year)])

    # get current inplaced households
    hh = households.to_frame(["semmcd", "building_id"])
    hh = hh[hh.building_id != -1]

    # groupby semmcd and get count
    hh_by_city = hh.groupby("semmcd").size()

    # get the expected growth
    # growth = target_year_hh - current_hh
    mcd_growth = mcd_total[str(year)] - hh_by_city

    # temp set NaN growth to 0
    if mcd_growth.isna().sum() > 0:
        print("Warning: NaN exists in mcd_growth, replaced them with 0")
    mcd_growth = mcd_growth.fillna(0).astype(int)

    # Calculate using Block group HH count trend data
    bg_hh_increase = bg_hh_increase.to_frame()
    # use occupied, 3 year window trend = y_i - y_i-3
    bg_trend = bg_hh_increase.occupied - bg_hh_increase.previous_occupied
    bg_trend_norm_by_bg = (bg_trend - bg_trend.mean()) / bg_trend.std()
    bg_trend_norm_by_bg.name = "bg_trend"

    # init output mcd_model_quota Series
    new_units = pd.Series()

    # only selecting growth > 0
    mcd_growth = mcd_growth[mcd_growth > 0]

    # loop through mcd growth target
    for city in mcd_growth.index:
        # for each city, make n_units = n_choosers
        # sorted housing units by year built and bg growth trend

        # get valid city housing units for sampling
        city_units = housing_units[
            (housing_units.semmcd == city)
            & (
                # only sampling hu_filter == 0
                housing_units.hu_filter
                == 0
            )
            & (
                # only sampling sp_filter >= 0
                housing_units.sp_filter
                >= 0
            )
        ]

        # building_age normalized
        building_age = city_units.building_age
        building_age_norm = (building_age - building_age.mean()) / building_age.std()

        # bg trend normalized
        bg_trend_norm = (
            city_units[["geoid"]]
            .join(bg_trend_norm_by_bg, how="left", on="geoid")
            .bg_trend
        ).fillna(0)

        # sum of normalized score
        normalized_score = (-building_age_norm) + bg_trend_norm

        # set name to score
        normalized_score.name = "score"

        # use absolute index for sorting
        normalized_score = normalized_score.reset_index()

        # sorted by the score from high to low
        normalized_score = normalized_score.sort_values(
            by="score", ascending=False, ignore_index=False
        )

        # apply sorted index back to city_units
        city_units = city_units.iloc[normalized_score.index]
        # .sort_values(by='building_age', ascending=True)

        # pick the top k units
        growth = mcd_growth.loc[city]
        selected_units = city_units.iloc[:growth]
        if selected_units.shape[0] != growth:
            # mcd didn't match target due to lack of HU
            print(
                "MCD %s have %s housing unit but expected growth is %s"
                % (city, selected_units.shape[0], growth)
            )
        new_units = pd.concat([new_units, selected_units])

    # add mcd model quota to building table
    quota = new_units.index.value_counts()

    # !!important!! clean-up mcd_model_quota from last year before updating it
    buildings.update_col_from_series(
        "mcd_model_quota", pd.Series(0, index=blds.index), cast=True
    )

    # init new mcd_model_quota
    mcd_model_quota = pd.Series(0, index=blds.index)
    mcd_model_quota.loc[quota.index] = quota.values

    # update mcd_model_quota in buildings table
    buildings.update_col_from_series("mcd_model_quota", mcd_model_quota, cast=True)


@orca.step()
def update_bg_hh_increase(bg_hh_increase, households):
    """
    Update the block group household growth trend table used in the MCD sampling process.

    Args:
        bg_hh_increase (DataFrameWrapper): Blockgroup household growth trend table.
        households (DataFrameWrapper): Households table.

    Returns:
        None
    """
    # Base year 2020
    base_year = 2020
    year = orca.get_injectable("year")
    year_diff = year - base_year
    hh = households.to_frame(["geoid"]).reset_index()
    hh_by_bg = hh.groupby("geoid").count().household_id
    bg_hh = bg_hh_increase.to_frame()

    # Move occupied hh count one year down
    # 2->3, 1->2, 0->1
    bg_hh["occupied_year_minus_3"] = bg_hh["occupied_year_minus_2"]
    bg_hh["occupied_year_minus_2"] = bg_hh["occupied_year_minus_1"]
    bg_hh["occupied_year_minus_1"] = hh_by_bg.fillna(0).astype("int")

    # If the first few years, save the bg summary and use 2014 and 2019 data
    if year_diff > 4:
        # Update columns used for trend analysis
        bg_hh["occupied"] = hh_by_bg
        bg_hh["previous_occupied"] = bg_hh["occupied_year_minus_3"]

    # Update bg_hh_increase table
    orca.add_table("bg_hh_increase", bg_hh)

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
for repm_config in os.listdir(os.path.join(misc.models_dir(), "repm_2050")):
    model_name = repm_config.split(".")[0]

    if repm_config.startswith("res"):
        dep_var = "sqft_price_res"
    elif repm_config.startswith("nonres"):
        dep_var = "sqft_price_nonres"

    make_repm_func(model_name, "repm_2050/" + repm_config, dep_var)
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
def households_relocation_2050(households, annual_relocation_rates_for_households):
    relocation_rates = annual_relocation_rates_for_households.to_frame()
    relocation_rates = relocation_rates.rename(
        columns={"age_max": "age_of_head_max", "age_min": "age_of_head_min"}
    )
    relocation_rates.probability_of_relocating *= 0.2
    reloc = relocation.RelocationModel(relocation_rates, "probability_of_relocating")
    _print_number_unplaced(households, "building_id")
    print("un-placing")
    hh = households.to_frame(households.local_columns)

    # block all event buildings and special buildings (sp_filter<0)
    bb = orca.get_table("buildings").to_frame(orca.get_table("buildings").local_columns)
    blocklst = bb.loc[bb.sp_filter < 0].index
    hh = hh.loc[~hh.building_id.isin(blocklst)]

    idx_reloc = reloc.find_movers(hh)
    households.update_col_from_series(
        "building_id", pd.Series(-1, index=idx_reloc), cast=True
    )
    _print_number_unplaced(households, "building_id")


@orca.step()
def jobs_relocation_2050(jobs, annual_relocation_rates_for_jobs):
    relocation_rates = annual_relocation_rates_for_jobs.to_frame().reset_index()
    reloc = relocation.RelocationModel(relocation_rates, "job_relocation_probability")
    _print_number_unplaced(jobs, "building_id")
    print("un-placing")
    j = jobs.to_frame(jobs.local_columns)

    # block all event buildings and special buildings (sp_filter<0)
    bb = orca.get_table("buildings").to_frame(orca.get_table("buildings").local_columns)
    blocklst = bb.loc[bb.sp_filter < 0].index
    j = j.loc[~j.building_id.isin(blocklst)]

    idx_reloc = reloc.find_movers(j[j.home_based_status <= 0])
    jobs.update_col_from_series(
        "building_id", pd.Series(-1, index=idx_reloc), cast=True
    )
    _print_number_unplaced(jobs, "building_id")


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
    new.index.name = "household_id"
    pers.index.name = "person_id"
    out = [[new, pers]]
    target -= len(pers)
    best_qal = np.inf
    best = []
    for _ in range(3):
        # if there is no HH to transition for ct_inf, break
        if (
            sum(
                [
                    utils.filter_table(
                        hh, r, ignore={"total_number_of_households"}
                    ).shape[0]
                    for _, r in ct_inf.loc[iter_var].iterrows()
                ]
            )
            == 0
        ):
            # add empty hh and persons dh
            new = hh.loc[[]]
            pers = p.loc[[]]
            best = (new, pers)
            break
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
            new.index.name = "household_id"
            pers.index.name = "person_id"
            best = (new, pers)
            best_qal = qal
    out.append(best)
    return out


@orca.step()
def households_transition(
    households, persons, annual_household_control_totals, remi_pop_total, iter_var
):
    region_ct = annual_household_control_totals.to_frame()
    max_cols = region_ct.columns[region_ct.columns.str.endswith("_max")]
    region_ct[max_cols] = region_ct[max_cols].replace(-1, np.inf)
    region_ct[max_cols] += 1
    region_hh = households.to_frame(households.local_columns + ["large_area_id"])

    region_p = persons.to_frame(persons.local_columns)
    region_p.index = region_p.index.astype(int)

    if "changed_hhs" in orca.list_tables():
        ## add changed hhs and persons from previous year back (ensure transition sample availability )
        changed_hhs = orca.get_table("changed_hhs").local
        changed_hhs.index += region_hh.index.max()

        changed_ps = orca.get_table("changed_ps").local
        changed_ps.index += region_p.index.max()
        changed_ps["household_id"] = changed_ps["household_id"] + region_hh.index.max()

        region_hh = pd.concat([region_hh, changed_hhs])
        region_p = pd.concat([region_p, changed_ps])

    region_hh.index = region_hh.index.astype(int)
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
    pool = Pool(6)
    cunks_per_la = pool.map(presses_trans, arg_per_la)
    pool.close()
    pool.join()
    out = reduce(operator.concat, cunks_per_la)

    # Sync for testing
    # out = []
    # for la_arg in arg_per_la:
    #     out.append(presses_trans(la_arg))

    # fix indexes
    hhidmax = region_hh.index.values.max() + 1
    pidmax = region_p.index.values.max() + 1

    ## create {old_hh_id => new_hh_id} mapping
    hh_id_mapping = [x[0]["building_id"] for x in out]
    # list of number of hh added for each df in the hh_id_mapping
    hh_new_added = [(x == -1).sum() for x in hh_id_mapping]
    # cumulative sum for hh_new_added, add 0 at front
    hh_new_added_cumsum = [0] + list(np.cumsum(hh_new_added))
    for i, hhmap in enumerate(hh_id_mapping):
        ## create seperate mapping for each df in the list
        hhmap = hhmap.reset_index()
        hhmap["household_id_old"] = hhmap["household_id"]
        # assign hh_id to those newly added
        hhmap.loc[hhmap["building_id"] == -1, "household_id"] = list(
            range(
                hhidmax + hh_new_added_cumsum[i], hhidmax + hh_new_added_cumsum[i + 1]
            )
        )
        hh_id_mapping[i] = hhmap[["household_id_old", "household_id"]].set_index(
            "household_id_old"
        )

    ## hh df
    # merge with hh_id mapping and concat all hh dfs and reset their index
    out_hh = pd.concat(
        [
            pd.merge(
                x[0].reset_index(),
                hh_id_mapping[i],
                left_on="household_id",
                right_index=True,
            )
            for i, x in enumerate(out)
        ],
        verify_integrity=True,
        ignore_index=True,
        copy=False,
    )
    # sort
    out_hh = out_hh.sort_values(by="household_id")
    # set index to hh_id
    out_hh = out_hh.set_index("household_id_y")
    out_hh = out_hh[households.local_columns]
    out_hh.index.name = "household_id"
    ## persons df
    # merge with hh_id mapping and concat and reset their index
    out_person = pd.concat(
        [
            pd.merge(
                x[1].reset_index(),
                hh_id_mapping[i],
                left_on="household_id",
                right_index=True,
            )
            for i, x in enumerate(out)
        ],
        verify_integrity=True,
        ignore_index=True,
        copy=False,
    )
    new_p = (out_person.household_id_x != out_person.household_id_y).sum()
    out_person.loc[
        out_person.household_id_x != out_person.household_id_y, "person_id"
    ] = list(range(pidmax, pidmax + new_p))
    out_person["household_id"] = out_person["household_id_y"]
    out_person = out_person.set_index("person_id")

    orca.add_table("households", out_hh[households.local_columns])
    orca.add_table("persons", out_person[persons.local_columns])

def get_lpr_hh_seed_id_mapping(hh, p, hh_seeds, p_seeds):
    # get hh swapping mapping
    # 2hr runtime
    # recommend using cached result
    seeds = np.sort(hh.seed_id.unique())
    # get adding new worker seed_id mapping
    add_worker_dict = defaultdict(dict) 
    drop_worker_dict = defaultdict(dict) 
    for seed in seeds:
        print('seed: ', seed)
        # for each seed, find a counter seed which has 1 more worker and similar other attributes
        seed_hh = hh_seeds[hh_seeds.seed_id== seed].iloc[0]
        seed_p = p_seeds[p_seeds.seed_id == seed]
        hh_pool = hh_seeds
        hh_pool = hh_pool[hh_pool.persons == seed_hh.persons]
        hh_pool = hh_pool[hh_pool.race_id == seed_hh.race_id]
        hh_pool = hh_pool[hh_pool.aoh_bin == seed_hh.aoh_bin]
        hh_pool = hh_pool[hh_pool.children == seed_hh.children]

        seed_age_dist = np.sort(seed_p.age_bin.values)

        if seed_hh.workers + seed_hh.children < seed_hh.persons:
            hh_pool_add_worker = hh_pool[hh_pool.workers == seed_hh.workers + 1]
            # add worker with more hh income
            hh_pool_add_worker = hh_pool_add_worker[hh_pool_add_worker.inc_qt >= seed_hh.inc_qt]
            N = hh_pool_add_worker.shape[0]
            for i in range(N):
                local_p_seeds = p_seeds[p_seeds.seed_id == hh_pool_add_worker['seed_id'].iloc[i]]
                if all(np.sort(local_p_seeds.age_bin.values) == seed_age_dist):
                    new_age_bins = local_p_seeds.query('worker==1').age_bin.value_counts()
                    prev_age_bins = seed_p.query('worker==1').age_bin.value_counts()
                    for k, v in new_age_bins.items():
                        if k in prev_age_bins and v <= prev_age_bins[k]:
                            continue
                        add_age_bin = k
                    add_worker_dict[add_age_bin][seed] = hh_pool_add_worker.iloc[i].seed_id
                    break
        if seed_hh.workers > 0:
            hh_pool_drop_worker = hh_pool[hh_pool.workers == seed_hh.workers - 1]
            N = hh_pool_drop_worker.shape[0]
            for i in range(N):
                local_p_seeds = p_seeds[p_seeds.seed_id == hh_pool_drop_worker['seed_id'].iloc[i]]
                if all(np.sort(local_p_seeds.age_bin.values) == seed_age_dist):
                    new_age_bins = local_p_seeds.query('worker==1').age_bin.value_counts()
                    prev_age_bins = seed_p.query('worker==1').age_bin.value_counts()
                    for k, v in prev_age_bins.items():
                        if k in new_age_bins and v <= new_age_bins[k]:
                            continue
                        drop_age_bin = k
                    drop_worker_dict[drop_age_bin][seed] = hh_pool_drop_worker.iloc[i].seed_id
                    break
    # clean up some key with more than 1 worker added/removed in a single hh
    add_list = defaultdict(list)
    for age, add_swappable in add_worker_dict.items():
        for orig, target in add_swappable.items():
            q = '(worker == 1)&(age_bin==%s)'%age
            if p_seeds.loc[[orig]].query(q).shape[0]+1 != p_seeds.loc[[target]].query(q).shape[0]:
                add_list[age].append(orig)
    for age, ll in add_list.items():
        for dk in ll:
            del add_worker_dict[age][dk]
    drop_list = defaultdict(list)
    for age, drop_swappable in drop_worker_dict.items():
        for orig, target in drop_swappable.items():
            q = '(worker == 1)&(age_bin==%s)'%age
            if p_seeds.loc[[orig]].query(q).shape[0] != p_seeds.loc[[target]].query(q).shape[0]+1:
                drop_list[age].append(orig)
    for age, ll in drop_list.items():
        for dk in ll:
            del drop_worker_dict[age][dk]

    return add_worker_dict, drop_worker_dict

@orca.step()
def cache_hh_seeds(households, persons, iter_var):
    if iter_var != 2021:
        print('skipping cache_hh_seeds for forecast year')
        return
    # caching hh and persons seeds at the start of the model run
    hh = households.to_frame(households.local_columns + ["large_area_id"])
    hh["target_workers"] = 0
    hh['inc_qt'] = pd.qcut(hh.income, 4, labels=[1, 2, 3, 4])
    hh['aoh_bin'] = pd.cut(hh.age_of_head, [-1, 4, 17, 24, 34, 64, 200], labels=[1, 2, 3, 4, 5, 6])
    # generate age bins
    age_bin = [-1, 15, 19, 21, 24, 29, 34, 44, 54, 59, 61, 64, 69, 74, 199]
    age_bin_labels = [0,16,20,22,25,30,35,45,55,60,62,65,70,75,200]
    p = persons.to_frame(persons.local_columns + ["large_area_id"])
    p['age_bin'] = pd.cut(p.age, age_bin, labels=age_bin_labels[:-1])
    p['age_bin'] = p['age_bin'].fillna(0).astype(int)
    p = p.join(hh.seed_id, on='household_id')
    
    hh_seeds = hh.groupby('seed_id').first()
    p_seeds = p.groupby(['seed_id', 'member_id']).first()
    print('running cache_hh_seeds for base year')
    orca.add_table('hh_seeds', hh_seeds)
    orca.add_table('p_seeds', p_seeds)

# TODO:
# - Add worker by swapping hh with 1 worker more hh
# - try to limit the impact to other variables
# - same persons, children, income within range(?), car(?), race with worker + 1
# - update persons table(optional) and households table
@orca.step()
def fix_lpr(households, persons, hh_seeds, p_seeds, iter_var, employed_workers_rate):
    from numpy.random import choice

    hh = households.to_frame(households.local_columns + ["large_area_id"])
    hh_seeds = hh_seeds.to_frame()
    p_seeds = p_seeds.to_frame()
    changed_hhs = hh.copy()
    hh["target_workers"] = 0
    hh['inc_qt'] = pd.qcut(hh.income, 4, labels=[1, 2, 3, 4])
    hh['aoh_bin'] = pd.cut(hh.age_of_head, [-1, 4, 17, 24, 34, 64, 200], labels=[1, 2, 3, 4, 5, 6])
    # generate age bins
    age_bin = [-1, 15, 19, 21, 24, 29, 34, 44, 54, 59, 61, 64, 69, 74, 199]
    age_bin_labels = [0,16,20,22,25,30,35,45,55,60,62,65,70,75,200]
    p = persons.to_frame(persons.local_columns + ["large_area_id"])
    p['age_bin'] = pd.cut(p.age, age_bin, labels=age_bin_labels[:-1])
    p['age_bin'] = p['age_bin'].fillna(0).astype(int)

    p = p.join(hh.seed_id, on='household_id')
    changed_ps = p.copy()
    lpr = employed_workers_rate.to_frame(["age_min", "age_max", str(iter_var)])

    colls = [
        "persons",
        "race_id",
        "workers",
        "children",
        "large_area_id",
    ]  # , 'age_of_head'
    same = {tuple(idx): df[["income", "cars"]] for idx, df in hh.groupby(colls)}

    # reset seeds index for generating mappings
    hh_seeds = hh_seeds.reset_index()
    p_seeds = p_seeds.reset_index()

    USE_SWAPPING_SEED_MAPPING = True
    aw_path = 'data/add_worker_dict.pkl'
    dw_path = 'data/drop_worker_dict.pkl'
    if not os.path.exists(aw_path):
        USE_SWAPPING_SEED_MAPPING = False
        print(aw_path, ' not found. running get_lpr_hh_seed_id_mapping')
    if not os.path.exists(dw_path):
        USE_SWAPPING_SEED_MAPPING = False
        print(dw_path, ' not found. running get_lpr_hh_seed_id_mapping')
    if USE_SWAPPING_SEED_MAPPING:
        add_worker_df = pd.read_csv('data/add_worker_dict.csv', index_col=0)
        add_worker_df.columns = add_worker_df.columns.astype(int)
        drop_worker_df = pd.read_csv('data/drop_worker_dict.csv', index_col=0)
        drop_worker_df.columns = drop_worker_df.columns.astype(int)
    else:
        add_worker_dict, drop_worker_dict = get_lpr_hh_seed_id_mapping(hh, p, hh_seeds, p_seeds)
        add_worker_df = pd.DataFrame(add_worker_dict)
        add_worker_df.to_csv(aw_path, index=True)
        drop_worker_df = pd.DataFrame(drop_worker_dict)
        drop_worker_df.to_csv(aw_path, index=True)

    hh_seeds = hh_seeds.set_index('seed_id')
    p_seeds = p_seeds.set_index('seed_id')

    # p = p.reset_index().set_index('household_id')
    pg = p.groupby('household_id')

    hh_cols_to_swap = [col for col in hh.columns if col not in ['blkgrp', 'building_id', 'large_area_id']]
    p_cols_to_swap = [col for col in p.columns if col not in ['person_id', 'household_id', 'large_area_id', 'weight']]

    for large_area_id, row in lpr.iterrows():
        select = (
            (p.large_area_id == large_area_id)
            & (p.age >= row.age_min)
            & (p.age <= row.age_max)
        )
        emp_wokers_rate = row[str(iter_var)]
        lpr_workers = int(select.sum() * emp_wokers_rate)
        num_workers = (select & (p.worker == 1)).sum()

        # get dict for seeds mapping
        add_swappable = add_worker_df[row.age_min]
        add_swappable = add_swappable[add_swappable.notna()].astype(int).to_dict()
        drop_swappable = drop_worker_df[row.age_min]
        drop_swappable = drop_swappable[drop_swappable.notna()].astype(int).to_dict()

        if lpr_workers > num_workers:
            # employ some persons
            num_new_employ = int(lpr_workers - num_workers)
            while num_new_employ > 0:
                hh_swap_pool = hh[(hh.large_area_id == large_area_id) & (hh.seed_id.isin(add_swappable))]
                if hh_swap_pool.shape[0] == 0:
                    break
                to_add = min(hh_swap_pool.shape[0], num_new_employ)
                # sample num_new_employ
                hh_to_swap = hh_swap_pool.sample(to_add, replace=False)
                # target seed_ids
                target_hh_seed_id = hh_to_swap.seed_id.map(add_swappable)
                # overwrite old attributes except building_id, large_area_id, blkgrp
                hh.loc[hh_to_swap.index, hh_cols_to_swap] = hh_seeds.loc[target_hh_seed_id].reset_index()[hh_cols_to_swap].values
                # hh persons overwrite
                p_idx_to_update = np.array([], dtype=int)
                for hh_id in hh_to_swap.index:
                    hh_members = pg.get_group(hh_id)
                    p_idx_to_update = np.concatenate((p_idx_to_update, hh_members.index))
                p.loc[p_idx_to_update, p_cols_to_swap] = p_seeds.loc[target_hh_seed_id].reset_index()[p_cols_to_swap].values
                # update added_employ
                num_new_employ = int(lpr_workers - (
                    (p.large_area_id == large_area_id)
                    & (p.age >= row.age_min)
                    & (p.age <= row.age_max)
                    & (p.worker == 1)
                ).sum())

        else:
            # unemploy some persons
            num_drop_employ = int(num_workers - lpr_workers)
            while num_drop_employ > 0:
                hh_swap_pool = hh[(hh.large_area_id == large_area_id) & (hh.seed_id.isin(drop_swappable))]
                if hh_swap_pool.shape[0] == 0:
                    break
                to_drop = min(hh_swap_pool.shape[0], num_drop_employ)
                # sample num_new_employ
                hh_to_swap = hh_swap_pool.sample(to_drop, replace=False)
                # target seed_ids
                target_hh_seed_id = hh_to_swap.seed_id.map(drop_swappable)
                # overwrite old attributes except building_id, large_area_id, blkgrp
                hh.loc[hh_to_swap.index, hh_cols_to_swap] = hh_seeds.loc[target_hh_seed_id].reset_index()[hh_cols_to_swap].values
                # hh persons overwrite
                p_idx_to_update = np.array([], dtype=int)
                for hh_id in hh_to_swap.index:
                    hh_members = pg.get_group(hh_id)
                    p_idx_to_update = np.concatenate((p_idx_to_update, hh_members.index))
                p.loc[p_idx_to_update, p_cols_to_swap] = p_seeds.loc[target_hh_seed_id].reset_index()[p_cols_to_swap].values
                # update num_drop_employ
                num_drop_employ = int((
                    (p.large_area_id == large_area_id)
                    & (p.age >= row.age_min)
                    & (p.age <= row.age_max)
                    & (p.worker == 1)
                ).sum() - lpr_workers)

        after_selected = (
            (p.large_area_id == large_area_id)
            & (p.age >= row.age_min)
            & (p.age <= row.age_max)
            & (p.worker == True)
        )
        print(large_area_id, row.age_min, row.age_max, num_workers, lpr_workers, after_selected.sum())

    hh["old_workers"] = hh.workers
    hh.workers = p.groupby("household_id").worker.sum()
    hh.workers = hh.workers.fillna(0)
    changed = hh.workers != hh.old_workers
    print(f"changed number of HHs from LPR is {len(changed)})")

    # TODO: using hh controls to test out each segment number

    # save changed HHs and persons as valid samples for future transition model

    if len(changed[changed == True]) > 0:
        changed_hhs = changed_hhs[changed]
        changed_hhs["old_hhid"] = changed_hhs.index
        changed_hhs.index = range(1, 1 + len(changed_hhs))
        changed_hhs.index.name = "household_id"
        changed_hhs = changed_hhs.reset_index().set_index("old_hhid")

        changed_ps = changed_ps.loc[
            changed_ps.household_id.isin(changed_hhs[changed].index)
        ]
        changed_ps = changed_ps.rename(columns={"household_id": "old_hhid"})
        changed_ps = changed_ps.merge(
            changed_hhs[["household_id"]],
            left_on="old_hhid",
            right_index=True,
            how="left",
        )
        changed_ps.index = range(1, 1 + len(changed_ps))
        changed_ps = changed_ps.drop("old_hhid", axis=1)

        changed_hhs = changed_hhs.set_index("household_id")
        changed_hhs["building_id"] = -1

        print(f"saved {len(changed_hhs)} households for future samples")
    else:
        changed_hhs = hh.iloc[0:0]
        changed_ps = changed_ps.iloc[0:0]
    orca.add_table("changed_hhs", changed_hhs[households.local_columns])
    orca.add_table("changed_ps", changed_ps[persons.local_columns])

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
                jobs[
                    (jobs.sector_id == sector)
                    & (jobs.large_area_id == large_area_id)
                    & (jobs.building_id != -1)
                ]
                .groupby(["building_id"])
                .size()
            )
            # !! filter out -1 from the building pool
            counts_by_bid = counts_by_bid[counts_by_bid.index != -1]
            prop_by_bid = counts_by_bid / counts_by_bid.sum()
            choices = random_choice(
                segment.index.values, prop_by_bid.index.values, prop_by_bid.values
            )
            wrap_jobs.update_col_from_series("building_id", choices, cast=True)
    j_after_run = wrap_jobs.to_frame(wrap_jobs.local_columns)
    print(
        "done running job_scaling, remaining jobs in sectors",
        regional_sectors,
        "with -1 building_id: ",
        (
            (j_after_run.building_id == -1)
            & (j_after_run.sector_id.isin(regional_sectors))
        ).sum(),
    )


@orca.step()
def gq_pop_scaling_model(group_quarters, group_quarters_control_totals, parcels, year):
    def filter_local_gq(local_gqpop):
        protected = (
            ((local_gqpop.gq_code > 100) & (local_gqpop.gq_code < 200))
            | ((local_gqpop.gq_code > 500) & (local_gqpop.gq_code < 600))
            | (local_gqpop.gq_code == 701)
        )
        return local_gqpop[~protected]

    parcels = parcels.to_frame(parcels.local_columns)
    city_large_area = (
        parcels[["city_id", "large_area_id"]].drop_duplicates().set_index("city_id")
    )

    gqpop = group_quarters.to_frame(
        group_quarters.local_columns + ["city_id", "large_area_id"]
    )

    print("%s gqpop before scaling" % gqpop.shape[0])
    # gqhh = group_quarters_households.to_frame(group_quarters_households.local_columns)
    target_gq = group_quarters_control_totals.to_frame()
    target_gq = target_gq[target_gq.year == year]
    # add gq target to city table to iterate
    city_large_area["gq_target"] = target_gq["count"]
    city_large_area = city_large_area.fillna(0).sort_index()

    # if no control found, skip this year
    if target_gq.shape[0] == 0:
        print("Warning: No gq controls found for year %s, skipping..." % year)
        return

    for city_id, row in city_large_area.iterrows():
        local_gqpop = gqpop.loc[gqpop.city_id == city_id]
        diff = int(row.gq_target - len(local_gqpop))
        # diff = target_gq.loc[city_id]["count"] - len(local_gqpop)
        # keep certain GQ pop unchanged
        filtered_gqpop = filter_local_gq(local_gqpop)

        if len(local_gqpop) > 0:
            if len(filtered_gqpop) == 0:
                filtered_gqpop = local_gqpop

            if diff > 0:
                # diff = int(min(len(filtered_gqpop), abs(diff)))
                # if no existing GQ except protected, use large area sample

                # local_gqpop = gqpop.loc[gqpop.large_area_id == row.large_area_id]
                # filtered_gqpop = filter_local_gq(local_gqpop)

                newgq = filtered_gqpop.sample(diff, replace=True)
                newgq.index = gqpop.index.values.max() + 1 + np.arange(len(newgq))
                newgq["city_id"] = city_id
                gqpop = pd.concat((gqpop, newgq))

            elif diff < 0:
                diff = min(len(filtered_gqpop), abs(diff))
                removegq = filtered_gqpop.sample(diff, replace=False)
                gqpop.drop(removegq.index, inplace=True)

    print("%s gqpop after scaling" % gqpop.shape[0])
    print(
        "\tgq result - target",
        (gqpop.groupby("city_id").size().fillna(0) - city_large_area.gq_target).sum(),
    )

    gqpop.to_csv("data/gqpop_" + str(year) + ".csv")
    orca.add_table("group_quarters", gqpop[group_quarters.local_columns])


@orca.step()
def refiner(jobs, households, buildings, persons, year, refiner_events, group_quarters):
    # #35
    # location_ids = ["b_zone_id", "zone_id", "b_city_id", "city_id", "large_area_id"] # must include b_zone_id, and b_city for 2045 refinder_event table
    location_ids = ["zone_id", "city_id", "large_area_id"]
    jobs_columns = jobs.local_columns
    jobs = jobs.to_frame(jobs_columns + location_ids)
    group_quarters_columns = group_quarters.local_columns
    group_quarters = group_quarters.to_frame(group_quarters_columns + location_ids)
    households_columns = households.local_columns
    households = households.to_frame(households_columns + location_ids)
    households["household_id_old"] = households.index.values
    buildings_local_columns = buildings.local_columns
    buildings = buildings.to_frame(
        buildings.local_columns + location_ids + ["gq_building"]
    )
    dic_agent = {
        "jobs": jobs,
        "households": households,
        "group_quarters": group_quarters,
    }

    refinements = refiner_events.to_frame()
    refinements = refinements[refinements.year == year]
    assert refinements.action.isin(
        {"clone", "subtract_pop", "subtract", "add_pop", "add", "target_pop", "target"}
    ).all(), "Unknown action"
    assert refinements.agents.isin(
        {"jobs", "households", "group_quarters"}
    ).all(), "Unknown agents"

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

        if agent_type == "jobs":
            for _, record in trecords.iterrows():
                buildings.loc[
                    buildings.query(record.location_expression).index, "sp_filter"
                ] = -1  # all job event building will be filtered from reloc and LCM

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
        orca.add_table(
            "buildings", buildings[buildings_local_columns]
        )  # update buildings

    if refinements.agents.isin({"group_quarters"}).sum() > 0:
        group_quarters = dic_agent["group_quarters"]
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
        if "stories" not in sched_dev.columns:
            sched_dev["stories"] = 0
        zone = (
            # #35
            # sched_dev.b_zone_id
            sched_dev.zone_id
        )  # save buildings based zone and city ids for later updates. model could update columns using parcel zone and city ids.
        sched_dev = sched_dev.rename(
            columns={
                "nonres_sqft": "non_residential_sqft",
                "housing_units": "residential_units",
                "build_type": "building_type_id",
            }
        )
        # #35
        # city = sched_dev.b_city_id
        city = sched_dev.city_id
        ebid = sched_dev.building_id.copy()  # save event_id to be used later
        sched_dev = add_extra_columns_res(sched_dev)

        # #35
        # sched_dev["b_zone_id"] = zone
        # sched_dev["b_city_id"] = city
        sched_dev["zone_id"] = zone
        sched_dev["city_id"] = city
        sched_dev["hu_filter"] = 0
        sched_dev["event_id"] = ebid  # add back event_id
        # set sp_filter to -1 to nonres event to prevent future reloaction
        sched_dev.loc[sched_dev.non_residential_sqft > 0, "sp_filter"] = -1
        b = buildings.to_frame(buildings.local_columns)

        all_buildings = parcel_utils.merge_buildings(b, sched_dev[b.columns], False)
        print(
            "%s of buildings have been added in scheduled development events"
            % (all_buildings.shape[0] - b.shape[0])
        )
        orca.add_table("buildings", all_buildings)

        # Todo: maybe we need to impute some columns
        # Todo: parcel use need to be updated
        # Todo: record dev_id -> building_id


@orca.step()
def scheduled_demolition_events(
    buildings,
    parcels,
    households,
    jobs,
    iter_var,
    events_deletion,
    multi_parcel_buildings,
):
    multi_parcel_buildings = multi_parcel_buildings.to_frame()
    sched_dev = events_deletion.to_frame()
    sched_dev = sched_dev[sched_dev.year_built == iter_var].reset_index(drop=True)
    buildings_columns = buildings.local_columns
    if len(sched_dev) > 0:
        buildings = buildings.to_frame(
            buildings_columns + ["city_id"] + ["b_total_jobs", "b_total_households"]
        )
        drop_buildings = buildings[buildings.index.isin(sched_dev.building_id)].copy()
        buildings_idx = drop_buildings.index
        drop_buildings["year_demo"] = iter_var
        drop_buildings["step"] = "scheduled_demolition_events"

        if orca.is_table("dropped_buildings"):
            prev_drops = orca.get_table("dropped_buildings").to_frame()
            orca.add_table("dropped_buildings", pd.concat([drop_buildings, prev_drops]))
        else:
            orca.add_table("dropped_buildings", drop_buildings)

        new_buildings_table = buildings.drop(buildings_idx)[buildings_columns]
        orca.add_table("buildings", new_buildings_table)

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
        # Todo: parcel use need to be updated
        # get parcel_id if theres only one building in the parcel
        b_with_multi_parcels = multi_parcel_buildings[
            multi_parcel_buildings.building_id.isin(drop_buildings.index)
        ]
        parcels_idx_to_update = [
            pid
            # for pid in drop_buildings.parcel_id
            for pid in set(
                drop_buildings.parcel_id.values.tolist()
                + b_with_multi_parcels.parcel_id.values.tolist()
            )
            if pid not in new_buildings_table.parcel_id
        ]
        # update pct_undev to 0 if theres only one building in the parcel
        pct_undev_update = pd.Series(0, index=parcels_idx_to_update)
        # update parcels table
        parcels.update_col_from_series("pct_undev", pct_undev_update, cast=True)


@orca.step()
def random_demolition_events(
    buildings, parcels, households, jobs, year, demolition_rates, multi_parcel_buildings
):
    demolition_rates = demolition_rates.to_frame()
    demolition_rates *= 0.1 + (1.0 - 0.1) * (2050 - year) / (2050 - 2020)
    buildings_columns = buildings.local_columns
    buildings = buildings.to_frame(
        buildings.local_columns + ["city_id"] + ["b_total_jobs", "b_total_households"]
    )
    multi_parcel_buildings = multi_parcel_buildings.to_frame()

    b = buildings.copy()
    allowed = variables.parcel_is_allowed_2050()
    allowed_b = b.parcel_id.isin(allowed[allowed].index)
    buildings_idx = []

    def sample(targets, type_b, accounting, weights):
        # #35
        # for b_city_id, target in targets[targets > 0].items():
        #     rel_b = type_b[type_b.b_city_id == b_city_id]
        for city_id, target in targets[targets > 0].items():
            rel_b = type_b[type_b.city_id == city_id]
            rel_b = rel_b[rel_b[accounting] <= target]
            size = min(len(rel_b), int(target))
            if size > 0:
                rel_b = rel_b.sample(size, weights=rel_b[weights])
                rel_b = rel_b[rel_b[accounting].cumsum() <= int(target)]
                buildings_idx.append(rel_b.copy())

    b.loc[allowed_b, "wj"] = 1.0 / (1.0 + np.log1p(b.loc[allowed_b, "b_total_jobs"]))
    nonres_b = b.loc[allowed_b]
    sample(
        demolition_rates.typenonsqft,
        nonres_b[nonres_b.non_residential_sqft > 0],
        "non_residential_sqft",
        "wj",
    )
    nonres_b = b.non_residential_sqft == 0
    b.loc[allowed_b & nonres_b, "wh"] = 1.0 / (
        1.0 + np.log1p(b.loc[allowed_b & nonres_b, "b_total_households"])
    )
    filter_b = b.loc[allowed_b & nonres_b]
    sample(
        demolition_rates.type81units,
        filter_b[filter_b.building_type_id == 81],
        "residential_units",
        "wh",
    )
    sample(
        demolition_rates.type82units,
        filter_b[filter_b.building_type_id == 82],
        "residential_units",
        "wh",
    )
    sample(
        demolition_rates.type83units,
        filter_b[filter_b.building_type_id == 83],
        "residential_units",
        "wh",
    )
    # sample(demolition_rates.type84units, b[b.building_type_id == 84], 'residential_units', 'wh')

    # github issue #30
    if not buildings_idx:
        return

    drop_buildings = pd.concat(buildings_idx).copy()
    drop_buildings = drop_buildings[~drop_buildings.index.duplicated(keep="first")]
    buildings_idx = drop_buildings.index
    drop_buildings["year_demo"] = year
    drop_buildings["step"] = "random_demolition_events"

    if orca.is_table("dropped_buildings"):
        prev_drops = orca.get_table("dropped_buildings").to_frame()
        orca.add_table("dropped_buildings", pd.concat([drop_buildings, prev_drops]))
    else:
        orca.add_table("dropped_buildings", drop_buildings)

    new_buildings_table = buildings[buildings_columns].drop(buildings_idx)

    orca.add_table("buildings", new_buildings_table)

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
    # get parcel_id if theres only one building in the parcel
    b_with_multi_parcels = multi_parcel_buildings[
        multi_parcel_buildings.building_id.isin(drop_buildings.index)
    ]
    parcels_idx_to_update = [
        pid
        for pid in set(
            drop_buildings.parcel_id.values.tolist()
            + b_with_multi_parcels.parcel_id.values.tolist()
        )
        if pid not in new_buildings_table.parcel_id
    ]
    # update pct_undev to 0 if theres only one building in the parcel
    pct_undev_update = pd.Series(0, index=parcels_idx_to_update)
    # update parcels table
    parcels.update_col_from_series("pct_undev", pct_undev_update, cast=True)


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
        variables.parcel_is_allowed_2050,
        cfg="proforma.yaml",
        modify_costs=cost_shifter_callback,
    )
    feasibility = orca.get_table("feasibility").to_frame()
    # MCD feasibility
    for mcdid, df in parcels.semmcd.to_frame().groupby("semmcd"):
        orca.add_table(
            "feasibility_" + str(mcdid), feasibility[feasibility.index.isin(df.index)]
        )
    # large_area feasibility
    for lid, df in parcels.large_area_id.to_frame().groupby("large_area_id"):
        orca.add_table(
            "feasibility_" + str(lid), feasibility[feasibility.index.isin(df.index)]
        )


def add_extra_columns_nonres(df):
    # type: (pd.DataFrame) -> pd.DataFrame
    for col in [
        "market_value",
        "improvement_value",
        "land_area",
        "tax_exempt",
        "sqft_price_nonres",
        "sqft_price_res",
        "sqft_per_unit",
        # "hu_filter",
        "event_id",
        "sp_filter",
        "mcd_model_quota",
    ]:
        df[col] = 0
    df["year_built"] = orca.get_injectable("year")
    p = orca.get_table("parcels").to_frame(["zone_id", "city_id"])
    for col in ["zone_id", "city_id"]:
        # #35
        # df["b_" + col] = misc.reindex(p[col], df.parcel_id)
        df[col] = misc.reindex(p[col], df.parcel_id)
    return df.fillna(0)


def add_extra_columns_res(df:pd.DataFrame) -> pd.DataFrame:
    """
    Add extra columns to a DataFrame containing residential property information.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing residential property information.

    Returns:
    pd.DataFrame: DataFrame with added extra columns and calculated values.
    """
    # add nonres columns
    df = add_extra_columns_nonres(df)

    # add sqft_per_units
    if "ave_unit_size" in df.columns:
        df["sqft_per_unit"] = df["ave_unit_size"]
    elif ("res_sqft" in df.columns) & ("residential_units" in df.columns):
        df["sqft_per_unit"] = df["res_sqft"] / df["residential_units"]
    else:
        df["sqft_per_unit"] = misc.reindex(
            orca.get_table("parcels").ave_unit_size, df.parcel_id
        )

    # github issue #31
    # generating default `mcd_model_quota` as the same as the `residential_units`
    # df["mcd_model_quota"] = df["residential_units"]

    # set default mcd quota to 0
    df["mcd_model_quota"] = 0

    return df.fillna(0)


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


def register_btype_distributions(buildings: pd.DataFrame):
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
        return 0, []

    # get the list of parcel_id whose pct_undev need to be updated
    pid_need_updates = [
        pid for pid in new_buildings.parcel_id if pid not in buildings.parcel_id
    ]

    # set default hu_filter to 0
    new_buildings["hu_filter"] = 0

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
    # return the number of units added and the list of parcel_id for updating pct_undev
    return (
        new_buildings.residential_units.sum() - new_buildings.current_units.sum(),
        pid_need_updates,
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
def residential_developer(
    households, parcels, target_vacancies_mcd, mcd_total, debug_res_developer
):
    """
    Simulate residential property development process

    This function simulates the process of residential property development for MCDs.
    It calculates the target number of new residential units to be developed based on target vacancies,
    existing units, and desired occupancy rates. The function also updates the parcel and building tables
    to reflect the new developments and their occupancy.

    Parameters:
    households (orca.DataFrameWrapper): Household data table.
    parcels (orca.DataFrameWrapper): Parcels data table.
    target_vacancies_mcd (orca.DataFrameWrapper): Target vacancies by MCD.
    mcd_total (orca.DataFrameWrapper): MCD households targets
    debug_res_developer (orca.DataFrameWrapper): Debugging table

    Returns:
    None
    """
    # get current year
    year = orca.get_injectable("year")

    # get target vacancies by mcd for current year
    target_vacancies = target_vacancies_mcd.to_frame()
    target_vacancies = target_vacancies[str(year)]

    # get original buildings
    orig_buildings = orca.get_table("buildings").to_frame(
        ["residential_units", "semmcd", "building_type_id"]
    )

    # the mcd_total for year
    mcd_total = mcd_total.to_frame([str(year)])[str(year)]

    # load debugger df
    debug_res_developer = debug_res_developer.to_frame()

    # loop through mcd id
    for mcdid, _ in parcels.semmcd.to_frame().groupby("semmcd"):
        print(f"developing residential units for {mcdid}")

        # get original buildings in current mcd
        mcd_orig_buildings = orig_buildings[orig_buildings.semmcd == mcdid]

        # handle missing mcdid
        if mcdid not in mcd_total.index:
            continue

        # get target vacancy
        target_vacancy = float(target_vacancies[mcdid])

        # current hh from hh table
        cur_agents = (households.semmcd == mcdid).sum()

        # target hh from mcd_total table
        target_agents = mcd_total.loc[mcdid]

        # number of current total housing units
        num_units = mcd_orig_buildings.residential_units.sum()

        print("Number of current agents: {:,}".format(cur_agents))
        print("Number of target agents: {:,}".format(target_agents))
        print("Number of agent spaces: {:,}".format(int(num_units)))
        print("Current vacancy = {:.2f}".format(1.0 - cur_agents / float(num_units)))
        assert target_vacancy < 1.0
        target_units = int(max((target_agents / (1.0 - target_vacancy) - num_units), 0))
        print(
            "Target vacancy = {:.2f}, target of new units = {:,}\n".format(
                target_vacancy, target_units
            )
        )

        # calculate prior form_btype_distributions
        register_btype_distributions(mcd_orig_buildings)

        # run developer step
        units_added, parcels_idx_to_update = run_developer(
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

        # update pct_undev to 100 if theres only one building in the parcel
        pct_undev_update = pd.Series(100, index=parcels_idx_to_update)

        # update parcels table
        parcels.update_col_from_series("pct_undev", pct_undev_update, cast=True)

        debug_res_developer = debug_res_developer.append(
            {
                "year": year,
                "mcd": mcdid,
                "target_units": target_units,
                "units_added": units_added,
            },
            ignore_index=True,
        )
        if units_added < target_units:
            print(
                " ***  Not enough housing units have been built by the developer model for mcd %s, target: %s, built: %s"
                % (mcdid, target_units, int(units_added))
            )
    # log the target and result in this year's run
    orca.add_table("debug_res_developer", debug_res_developer)


@orca.step()
def non_residential_developer(jobs, parcels, target_vacancies):
    """
    Non-residential space developer step.

    This Orca step handles the development of non-residential spaces in different large areas based on target
    vacancy rates and job demand. It calculates the necessary number of non-residential spaces to achieve the
    target vacancy rate and then runs the non-residential developer model.

    Parameters:
    jobs (orca.DataFrameWrapper): Jobs
    parcels (orca.DataFrameWrapper): Parcels
    target_vacancies (orca.DataFrameWrapper): target vacancy rates for large areas.

    Returns:
    None
    """
    # get target vacancies
    target_vacancies = target_vacancies.to_frame()
    target_vacancies = target_vacancies[
        target_vacancies.year == orca.get_injectable("year")
    ]

    # get original buildings table
    orig_buildings = orca.get_table("buildings").to_frame(
        ["job_spaces", "large_area_id", "building_type_id"]
    )

    # loop through large area
    for lid, _ in parcels.large_area_id.to_frame().groupby("large_area_id"):
        # get large area buildings
        la_orig_buildings = orig_buildings[orig_buildings.large_area_id == lid]

        # get current large area vacancy target
        target_vacancy = float(
            target_vacancies[
                target_vacancies.large_area_id == lid
            ].non_res_target_vacancy_rate
        )

        # number of non-homebased jobs in the large area
        num_agents = ((jobs.large_area_id == lid) & (jobs.home_based_status == 0)).sum()

        # number of total job spaces for LA
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

        # calculate prior form_btype_distributions
        register_btype_distributions(la_orig_buildings)

        # run nonres developer step
        spaces_added, parcels_idx_to_update = run_developer(
            target_units,
            lid,
            ["office", "retail", "industrial", "medical", "entertainment"],
            orca.get_table("buildings"),
            "job_spaces",
            parcels.parcel_size,
            parcels.ave_unit_size,
            parcels.total_job_spaces,
            "nonres_developer.yaml",
            add_more_columns_callback=add_extra_columns_nonres,
        )

        # update pct_undev to 100 if theres only one building in the parcel
        pct_undev_update = pd.Series(100, index=parcels_idx_to_update)
        # update parcels table
        parcels.update_col_from_series("pct_undev", pct_undev_update, cast=True)


@orca.step()
def update_sp_filter(buildings):
    """
    Update the 'sp_filter' column of the 'buildings' table for selected building types.

    This step updates the 'sp_filter' column of the 'buildings' table based on the specified
    'building_type_id' values. It sets the 'sp_filter' value to -1 for buildings with building
    types that match the selected building type IDs. This step is used to exclude building from
    demolition and LCM processes 

    Parameters:
    buildings (orca.DataFrameWrapper): Buildings data table.

    Returns:
    None
    """
    # update sp_filter to -1 for selected building_types
    selected_btypes = {
        11: "Educational",
        13: "Religious and Civic",
        14: "Governmental",
        52: "Hospital",
        53: "Residential Care Facility",
        92: "Library",
        93: "Dormitory Quarters",
        94: "Death Care Services",
        95: "Parking Garage",
    }

    updated_buildings = buildings.to_frame(buildings.local_columns)
    print(
        "Updating %s buildings sp_filter to -1"
        % (
            updated_buildings.loc[
                updated_buildings.building_type_id.isin(selected_btypes)
            ].shape[0]
        )
    )

    # set sp_filter to -1
    updated_buildings.loc[
        updated_buildings.building_type_id.isin(selected_btypes), "sp_filter"
    ] = -1

    # update buildings table
    orca.add_table("buildings", updated_buildings)


@orca.step()
## for 2050 forecast, ready to replace the old one
def build_networks_2050(parcels):
    import yaml

    # networks in semcog_networks.h5
    with open(
        "/home/da/semcog_urbansim/configs/available_networks_2050.yaml", "r"
    ) as stream:
        dic_net = yaml.load(stream, Loader=yaml.FullLoader)

    year = orca.get_injectable("year")
    utils.run_log(f"\tyear: {year} | {time.ctime()}")

    # change travel data to 2030, enable when travel data 2030 is inplace
    if year == 2030:
        orca.add_table("travel_data", orca.get_table("travel_data_2030").to_frame())
        orca.clear_columns("zones")

    if year < 2030:
        lstnet = [
            {
                "name": "osm_roads_walk_2020",
                "cost": "cost1",
                "prev": 26500,  # 5 miles
                "net": "net_walk",
            },
            {
                "name": "highway_ext_2020",
                "cost": "cost1",
                "prev": 60,  # 60 minutes
                "net": "net_drv",
            },
        ]
    else:
        lstnet = [
            {
                "name": "osm_roads_walk_2020",
                "cost": "cost1",
                "prev": 26500,  # 5 miles
                "net": "net_walk",
            },
            {
                "name": "highway_ext_2030",
                "cost": "cost1",
                "prev": 60,  # 60 minutes
                "net": "net_drv",
            },
        ]

    ## TODO, remove 2015, 2019 after switching to full 2050 model
    if (year in [2015, 2020, 2021, 2030]) or ("net_walk" not in orca.list_tables()):
        st = pd.HDFStore(os.path.join(misc.data_dir(), "semcog_2050_networks.h5"), "r")
        pdna.network.reserve_num_graphs(2)

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
            "prev": 26500,  # 2 miles
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
def neighborhood_vars(jobs, households, buildings, pseudo_building_2020):
    b = buildings.to_frame(["large_area_id"])
    j = jobs.to_frame(jobs.local_columns)
    h = households.to_frame(households.local_columns)
    pseudo_buildings = pseudo_building_2020.to_frame()

    ## jobs
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

    ## households
    idx_invalid_building_id = np.in1d(h.building_id, b.index.values) == False
    # ignore hh in pseudo_buildings
    idx_invalid_building_id = idx_invalid_building_id & ~(
        h.building_id.isin(pseudo_buildings.index)
    )
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
        h["large_area_id"] = misc.reindex(b.large_area_id, h.building_id)
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


@orca.step()
def drop_pseudo_buildings(households, buildings, pseudo_building_2020):
    """Unplace households from them
        - 1729 pseudo hh in 2050 forecast

    Args:
        households (DataFrameWrapper): households
        buildings (DataFrameWrapper): buildings
        pseudo_building_2020 (DataFrameWrapper): pseudo_building_2020
    """
    # define k: number of pseudo hh to drop each year
    k = 90

    # get households with sp_filter
    hh = households.to_frame(households.local_columns + ["sp_filter"])

    # N: number of existing pseudo households
    N = hh[hh.sp_filter == -2].shape[0]

    # if empty, return
    if N == 0:
        return

    # if less than k, replace k
    if N < k:
        k = N

    # sample k pseudo household to drop
    hh_to_drop = hh[hh.sp_filter == -2].sample(k)

    # unplace households and set sampled hh with building_id -1
    hh.loc[hh_to_drop.index, "building_id"] = -1

    # set resiential units to hh counts, avoid vacant units in pseudo buildings
    hhs_by_pseudo_b = (
        hh[(hh.sp_filter == -2) & (hh.building_id > -1)].groupby("building_id").size()
    )
    pb = pseudo_building_2020.local
    bb = buildings.local
    bb.loc[pb.index, "residential_units"] = 0
    bb.loc[hhs_by_pseudo_b.index, "residential_units"] = hhs_by_pseudo_b

    print("Dropped %s hh from current pseudo buildings." % k)

    # update households and buildings
    orca.add_table("households", hh)
    orca.add_table("buildings", bb)


@orca.step()
def refine_housing_units(households, buildings, mcd_total):
    """ Refine housing units before mcd_hu_sampling to allow it matching mcd_total

    Args:
        households (DataFrame Wrapper): households
        buildings (DataFrame Wrapper): buildings
        mcd_total (DataFrame Wrapper): mcd_total
    """
    year = orca.get_injectable("year")
    b = buildings.to_frame(
        buildings.local_columns + ["hu_filter", "sp_filter", "semmcd"]
    )
    mcd_total = mcd_total.to_frame([str(year)])

    # get units
    bunits = b["residential_units"]
    bunits = bunits[bunits.index.values >= 0]
    bunits = bunits[bunits > 0]
    # generate housing units from units
    indexes = np.repeat(bunits.index.values, bunits.values.astype("int"))
    housing_units = b.loc[indexes]
    # filter out unplaceable HU
    housing_units = housing_units[housing_units["hu_filter"] == 0]
    housing_units = housing_units[housing_units["sp_filter"] >= 0]
    hu_by_mcd = b.groupby(['semmcd']).sum().residential_units.astype(int) 

    mcd_target = mcd_total[str(year)]

    hu_mcd_diff = pd.DataFrame([], index=hu_by_mcd.index.union(mcd_target.index))
    hu_mcd_diff.index.name = "semmcd"
    hu_mcd_diff["hu"] = hu_by_mcd
    hu_mcd_diff["target"] = mcd_target
    hu_mcd_diff = hu_mcd_diff.fillna(0)
    hu_mcd_diff["diff"] = (hu_mcd_diff["target"] - hu_mcd_diff["hu"]).astype(int)
    hu_mcd_diff_gt_0 = hu_mcd_diff[hu_mcd_diff["diff"] > 0]

    for city, row in hu_mcd_diff_gt_0.iterrows():
        add_hu = int(row["diff"] * 1.1)
        local_units = housing_units.loc[
            (housing_units.building_type_id.isin([81, 82, 83]))
            & (housing_units.city_id == city)
        ]
        # filter out hu_filter and sp_filter
        local_units = local_units[local_units["hu_filter"] == 0]
        local_units = local_units[local_units["sp_filter"] >= 0]
        new_units = local_units.sample(
            add_hu, replace=False, random_state=1
        ).index.value_counts()
        b.loc[new_units.index, "residential_units"] += new_units
        print(
            "Adding %s units to city %s, actually added %s"
            % (add_hu, city, new_units.sum())
        )

    # update res_units in building table
    buildings.update_col_from_series(
        "residential_units", b["residential_units"], cast=True
    )

def _print_number_unplaced(df, fieldname="building_id"):
    """
    Just an internal function to use to compute and print info on the number
    of unplaced agents.
    """
    counts = (df[fieldname] == -1).sum()
    print("Total currently unplaced: %d" % counts)


def remove_unplaced_agents():
    """
    unplaced jobs and households and jobs are removed 
    """
    for tbl in ["households", "jobs"]:
        df = orca.get_table(tbl).local
        df = df.loc[df.building_id != -1]
        orca.add_table(tbl, df)
