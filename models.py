import os
import time
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
    hh = households.to_frame(["building_id", "large_area_id"])
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
    with open(os.path.join(misc.configs_dir(), "mcd_hu_sampling.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cfg


@orca.step()
def mcd_hu_sampling(buildings, households, mcd_total, bg_hh_increase):
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
    # get current year
    year = orca.get_injectable("year")
    # get housing unit table from buildings
    config = orca.get_injectable("mcd_hu_sampling_config")
    vacant_variable = config["vacant_variable"]
    blds = buildings.to_frame(
        [
            "building_id",
            "semmcd",
            vacant_variable,
            "building_age",
            "geoid",
            "mcd_model_quota",
            "hu_filter",
        ]
    )
    vacant_units = blds[vacant_variable]
    vacant_units = vacant_units[vacant_units.index.values >= 0]
    vacant_units = vacant_units[vacant_units > 0]
    # generate housing units from vacant units
    indexes = np.repeat(vacant_units.index.values, vacant_units.values.astype("int"))
    housing_units = blds.loc[indexes]

    # the mcd_total for year and year-1
    mcd_total = mcd_total.to_frame([str(year)])
    hh = households.to_frame(["semmcd", "building_id"])
    hh = hh[hh.building_id != -1]
    hh_by_city = hh.groupby("semmcd").count().building_id
    # 4 mcds missing in the mcd_total table [2073, 2172, 6142, 2252]
    # get the growth by subtract the previous year
    # growth = target_year_hh - current_hh
    mcd_growth = mcd_total[str(year)] - hh_by_city
    # temp set NaN to 0
    if mcd_growth.isna().sum() > 0:
        print("Warning: NaN exists in mcd_growth, replaced them with 0")
    mcd_growth = mcd_growth.fillna(0).astype(int)
    ####
    bg_hh_increase = bg_hh_increase.to_frame()
    # use occupied, 3 year window trend = y_i - y_i-3
    bg_trend = bg_hh_increase.occupied - bg_hh_increase.previous_occupied
    bg_trend_norm_by_bg = (bg_trend - bg_trend.mean()) / bg_trend.std()
    bg_trend_norm_by_bg.name = "bg_trend"

    # init output df
    new_units = pd.Series()
    # only selecting growth > 0
    mcd_growth = mcd_growth[mcd_growth > 0]
    for city in mcd_growth.index:
        # for each city, make n_units = n_choosers
        # sorted by year built
        city_units = housing_units[
            (housing_units.semmcd == city)
            & (
                # only sampling hu_filter == 0
                housing_units.hu_filter
                == 0
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
        if selected_units.shape[0] != growth:
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
    mcd_model_quota = pd.Series(0, index=blds.index)
    mcd_model_quota.loc[quota.index] = quota.values
    buildings.update_col_from_series("mcd_model_quota", mcd_model_quota, cast=True)
    ### generating debug table
    debug = True
    if debug:
        blds = buildings.to_frame(
            ["building_id", "semmcd", "residential_units", "building_age", "geoid"]
        )
        res_units = blds["residential_units"]
        res_units = res_units[res_units.index.values >= 0]
        res_units = res_units[res_units > 0]
        # generate housing units from vacant units
        indexes = np.repeat(res_units.index.values, res_units.values.astype("int"))
        hus = blds.loc[indexes]
        new_hus = hus[hus.building_age < 8].groupby("geoid").count().semmcd
        new_hus.name = "hu_under_8"
        old_hus = hus[hus.building_age >= 8].groupby("geoid").count().semmcd
        old_hus.name = "hu_over_8"
        sampled = (
            buildings.to_frame(["geoid", "mcd_model_quota"]).groupby("geoid").sum()
        )
        combined = pd.concat([new_hus, old_hus, sampled], axis=1).fillna(0)
        orca.add_table("hu_sampling_bg_summary", combined)


@orca.step()
def update_bg_hh_increase(bg_hh_increase, households):
    # baseyear 2020
    base_year = 2020
    year = orca.get_injectable("year")
    year_diff = year - base_year
    hh = households.to_frame(["geoid"]).reset_index()
    hh_by_bg = hh.groupby("geoid").count().household_id
    # hh_by_bg.index = hh_by_bg.index.astype(int)
    bg_hh = bg_hh_increase.to_frame()
    bg_hh["occupied_year_minus_3"] = bg_hh["occupied_year_minus_2"]
    bg_hh["occupied_year_minus_2"] = bg_hh["occupied_year_minus_1"]
    # some bg missing in initial bg_hh_increase table
    # TODO: email to Sirisha and add them to a copy of the table
    # Added all 0s to the missing bg 261158335004
    bg_hh["occupied_year_minus_1"] = hh_by_bg.fillna(0).astype("int")
    if year_diff > 4:
        # if the first few years, save the bg summary and use 2014 and 2019 data
        bg_hh["occupied"] = hh_by_bg
        bg_hh["previous_occupied"] = bg_hh["occupied_year_minus_3"]
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
    pool = Pool(2)
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


@orca.step()
def fix_lpr(households, persons, iter_var, employed_workers_rate):
    from numpy.random import choice

    hh = households.to_frame(households.local_columns + ["large_area_id"])
    hh["target_workers"] = 0
    p = persons.to_frame(persons.local_columns + ["large_area_id"])
    lpr = employed_workers_rate.to_frame(["age_min", "age_max", str(iter_var)])
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
        emp_wokers_rate = row[str(iter_var)]
        lpr_workers = int(select.sum() * emp_wokers_rate)
        # lpr_workers = int(lpr_segment)
        num_workers = (select & employed).sum()

        if lpr_workers > num_workers:
            # employ some persons
            num_new_employ = int(lpr_workers - num_workers)
            new_employ.append(
                choice(p[select & (~employed)].index, num_new_employ, False)
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
def gq_pop_scaling_model(group_quarters, group_quarters_control_totals, year):
    gqpop = group_quarters.to_frame(group_quarters.local_columns + ["city_id"])
    target_gq = group_quarters_control_totals.to_frame()
    target_gq = target_gq[target_gq.year == year]
    # if no control found, skip this year
    if target_gq.shape[0] == 0:
        print("Warning: No gq controls found for year %s, skipping..." % year)
        return

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
    allowed_b = allowed_b.loc[allowed_b.sp_filter >= 0]
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


def add_extra_columns_res(df):
    # type: (pd.DataFrame) -> pd.DataFrame
    df = add_extra_columns_nonres(df)
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
    df["mcd_model_quota"] = df["residential_units"]
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
        return 0, []

    # get the list of parcel_id whose pct_undev need to be updated
    pid_need_updates = [
        pid for pid in new_buildings.parcel_id if pid not in buildings.parcel_id
    ]

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
    # get current year
    year = orca.get_injectable("year")
    target_vacancies = target_vacancies_mcd.to_frame()
    target_vacancies = target_vacancies[str(year)]
    orig_buildings = orca.get_table("buildings").to_frame(
        ["residential_units", "semmcd", "building_type_id"]
    )
    # the mcd_total for year and year-1
    mcd_total = mcd_total.to_frame([str(year)])[str(year)]
    debug_res_developer = debug_res_developer.to_frame()
    for mcdid, _ in parcels.semmcd.to_frame().groupby("semmcd"):
        mcd_orig_buildings = orig_buildings[orig_buildings.semmcd == mcdid]
        # handle missing mcdid
        if mcdid not in mcd_total.index:
            continue
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
        print("Current vacancy = {:.2f}".format(1 - cur_agents / float(num_units)))
        assert target_vacancy < 1.0
        target_units = int(max((target_agents / (1 - target_vacancy) - num_units), 0))
        print(
            "Target vacancy = {:.2f}, target of new units = {:,}".format(
                target_vacancy, target_units
            )
        )

        register_btype_distributions(mcd_orig_buildings)
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

        # TODO: update parcels.pct_undev to 100 for units_added
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
                "Not enough housing units have been built by the developer model for mcd %s, target: %s, built: %s"
                % (mcdid, target_units, int(units_added))
            )
    # log the target and result in this year's run
    orca.add_table("debug_res_developer", debug_res_developer)


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
    updated_buildings.loc[
        updated_buildings.building_type_id.isin(selected_btypes), "sp_filter"
    ] = -1
    orca.add_table("buildings", updated_buildings)


@orca.step()
def update_sp_filter(buildings):
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
    updated_buildings.loc[
        updated_buildings.building_type_id.isin(selected_btypes), "sp_filter"
    ] = -1
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
