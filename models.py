import os
import time
import yaml
import operator
from multiprocessing import Pool
from collections import defaultdict
import pickle

import numpy as np
import random
import orca
import pandana as pdna
import pandas as pd
from urbansim.models import transition, relocation
from urbansim.utils import misc, networks
from urbansim_parcels import utils as parcel_utils
from forecast_estimation.utils import load_taz_vars_from_orca, load_2015_taz_vars_from_hdf, load_2010_taz_vars_from_folder

import utils
import lcm_utils
import input_paths
from functools import reduce

# set configs if they are not set
if not orca.is_injectable('hlcm_model_path'):
    orca.add_injectable('hlcm_model_path', '/mnt/hgfs/RDF2050/estimation/models/models_24Mar5')

if not orca.is_injectable('elcm_model_path'):
    orca.add_injectable('elcm_model_path', '/mnt/hgfs/RDF2050/estimation/models/elcm_models_24Jun05')

if not orca.is_injectable('yaml_configs'):
    orca.add_injectable('yaml_configs', 'yaml_configs_elcm_hlcm.yaml')

if not orca.is_injectable('ENABLE_SCENARIO'):
    orca.add_injectable('ENABLE_SCENARIO', False)

import dataset
import variables

# Setup Scenario controls
if orca.get_injectable('ENABLE_SCENARIO'):
    hh_controls_path = orca.get_injectable('scenario_hh_control_path')
    new_hh_controls = pd.read_csv(hh_controls_path, index_col=0)
    orca.add_table('annual_household_control_totals', new_hh_controls)

    # Scenario population still ships as TOTAL population -> loaded as the legacy
    # remi_pop_total table. households_transition falls back to it (with a
    # warning) while allow_total_pop_fallback is True. For a scenario run with
    # allow_total_pop_fallback=False, provide a household-population scenario file
    # and register it as `remi_hh_pop` instead.
    remi_total_pop_path = orca.get_injectable('scenario_remi_total_pop')
    new_remi_total_pop = pd.read_csv(remi_total_pop_path, index_col=0)
    orca.add_table('remi_pop_total', new_remi_total_pop)

    if orca.is_injectable('scenario_emp_control_path'):
        emp_controls_path = orca.get_injectable('scenario_emp_control_path')
        new_emp_controls = pd.read_csv(emp_controls_path, index_col=0)
        orca.add_table('annual_employment_control_totals', new_emp_controls)

# Set up location choice model objects.
# Register as injectable to be used throughout simulation
hh_location_choice_models, emp_location_choice_models = {}, {}
hlcm_step_names = []
elcm_step_names = []

hlcm_model_path = orca.get_injectable('hlcm_model_path')
elcm_model_path = orca.get_injectable('elcm_model_path')
yaml_configs = orca.get_injectable('yaml_configs')

# load hlcm model config from path and save to yaml
lcm_utils.load_hlcm_model_configs_from_path(hlcm_model_path, yaml_configs)
lcm_utils.load_elcm_model_configs_from_path(elcm_model_path, yaml_configs)

# load model_configs
model_configs = lcm_utils.get_model_category_configs(yaml_configs)

for model_category_name, model_category_attributes in model_configs.items():
    if model_category_attributes["model_type"] == "location_choice":
        model_config_files = model_category_attributes["config_filenames"]

        for model_config in model_config_files:

            if model_category_name == "hlcm":
                # load torch-based hlcm model
                model = lcm_utils.load_torch_lcm(os.path.join(hlcm_model_path, 'pts', model_config), model_category_attributes)
                hlcm_step_names.append(model_config)
                hh_location_choice_models[model_config] = model

            if model_category_name == "elcm":
                # load torch-based elcm model
                model = lcm_utils.load_torch_lcm(os.path.join(elcm_model_path, 'pts', model_config), model_category_attributes)
                elcm_step_names.append(model_config)
                emp_location_choice_models[model_config] = model

orca.add_injectable("hh_location_choice_models", hh_location_choice_models)
orca.add_injectable("emp_location_choice_models", emp_location_choice_models)

# sort hlcm by name, will follow large_area -> first_cat -> second_cat -> third_cat
orca.add_injectable("hlcm_step_names", sorted(hlcm_step_names))

# sort elcm: run elcm by specific job_sector sequence defined below
elcm_sector_order = [3, 6, 10, 11, 14, 9, 4, 2, 5, 16, 17, 8]
elcm_sector_order = {sector: idx for idx, sector in enumerate(elcm_sector_order)}
orca.add_injectable(
    "elcm_step_names",
    sorted(elcm_step_names, key=lambda x: elcm_sector_order[int(x.split('.')[0].split('_')[-1][6:])]),
)

for name, model in list(hh_location_choice_models.items()):
    lcm_utils.register_hlcm_model_step(name, alt_capacity=model_configs['hlcm']['vacant_variable'])

for name, model in list(emp_location_choice_models.items()):
    lcm_utils.register_elcm_model_step(
        name, 
        alt_capacity=model_configs['elcm']['vacant_variable'], 
        elcm_calibration_config=model_configs['elcm']['calibration']
    )


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
            "large_area_id",
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
                housing_units.hu_filter == 0
            )
            & (
                # only sampling sp_filter >= 0
                housing_units.sp_filter >= 0
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
    
    # TODO: check if quota is greater or equal to #of unplaced households 
    la_ids = blds.large_area_id.unique()
    h = households.local
    for la_id in la_ids:
        la_quota = new_units[new_units.large_area_id == la_id].shape[0]
        la_unplaced_hh = h[(h.large_area_id == la_id) & (h.building_id == -1)].shape[0]
        print( "%s: la_quota %s la_unplaced_hh %s" % (la_id, la_quota, la_unplaced_hh))
        if la_quota < la_unplaced_hh: 
            # not enough la_quota for unplaced hhs
            # sample LA housing units to match
            diff = la_unplaced_hh - la_quota
            la_housing_units = housing_units[
                (housing_units.large_area_id == la_id)
                & (housing_units.hu_filter == 0) # housing units should be filtered
                & (housing_units.sp_filter >= 0)
            ]
            la_new_units = new_units[new_units.large_area_id == la_id]
            rem = la_housing_units.index.value_counts().sub( la_new_units.index.value_counts(), fill_value=0).astype(int)
            rem_by_bid = rem[(rem > 0)]
            print( "%s missing %s HU: total remaining vacancy of %s" % (la_id, diff, rem_by_bid.sum()))
            while diff > 0:
                # TODO: rem_by_bid may be empty
                picked = rem_by_bid[rem_by_bid > 0].sample(1).index[0]
                rem_by_bid.loc[picked] = rem_by_bid.loc[picked] - 1
                new_units = pd.concat([new_units, blds.loc[[picked]]])
                diff -= 1

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
    base_year = orca.get_injectable("base_year")
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
def init_taz_hlcm_trend_by_year():
    """ Initialize taz_hlcm_trend_by_year injectable objection in orca
    - load 2045 input hdf 
    - load households and buildings from orca
    - compute variables 
    - saving DF to obj
    - update injectable
    """
    # init taz_hlcm_trend object
    taz_hlcm_trend_by_year = {}

    # # initiating 2010 attribute df
    input_2040 = orca.get_injectable('forecast_input_2040')
    df_2010 = load_2010_taz_vars_from_folder(input_2040)
    taz_hlcm_trend_by_year['2010'] = df_2010
    # # initiating 2015 attribute df
    hdf_input_2045 = orca.get_injectable('hdf_input_2045')
    df_2015 = load_2015_taz_vars_from_hdf(hdf_input_2045)
    taz_hlcm_trend_by_year['2015'] = df_2015
    print('Finishing init TAZ vars from hdf', hdf_input_2045)
    # initiating baseyear attribute df
    df_cur = load_taz_vars_from_orca()
    taz_hlcm_trend_by_year[str(orca.get_injectable("base_year"))] = df_cur
    print('Finishing loading TAZ vars from orca...')

    # add to injectable
    orca.add_injectable('taz_hlcm_trend_by_year', taz_hlcm_trend_by_year)

    # init job sector and building type weights
    job_btype = orca.get_table('jobs').to_frame(['sector_id', 'building_type_id'])

    # Count occurrences
    joint_counts = job_btype.groupby(['building_type_id', 'sector_id']).size().unstack(fill_value=0)

    # Normalize across building types to get conditional probabilities
    prob_matrix = joint_counts.div(joint_counts.sum(axis=0), axis=1)
    orca.add_injectable('job_btype_baseyear_prob_matrix', prob_matrix)

    # Calculate baseyear TAZ households with_children ratio
    # and save to table tract_hh_type_base_ratios
    hh = orca.get_table("households")
    # hh_types = ['children_has_children', 'children_no_children']
    hh_segments = lcm_utils.get_hlcm_segment()
    # Initialize result dictionary
    result = {}
    for hh_types in hh_segments:
        hh_df = hh.to_frame(list(hh_types) + ['tract_id'])
        # Calculate total households by TAZ
        total_hh = hh_df.groupby("tract_id").size()
        # Households satisfying all hh_type == 1
        mask = np.logical_and.reduce([hh_df[hh_type] == 1 for hh_type in hh_types])
        # Count households of this type
        hh_type_count = hh_df.loc[mask, 'tract_id'].value_counts()
        # Compute ratio by TAZ
        tract_hh_type_ratio = (hh_type_count / total_hh).fillna(0).clip(0, 1)
        # Add to result dictionary
        result["tract_hh_type_ratio_%s" % "_".join(hh_types)] = tract_hh_type_ratio 
    # Combine into DataFrame with TAZ as index
    tract_hh_type_base_ratios = pd.DataFrame(result).fillna(0)
    # Ensure tract_id is index
    tract_hh_type_base_ratios.index.name = "tract_id"
    # Register with Orca
    orca.add_table("tract_hh_type_base_ratios", tract_hh_type_base_ratios)


@orca.step()
def update_taz_hlcm_trend(taz_hlcm_trend_by_year, year, households, buildings):
    """Update taz_hlcm_trend_by_year for the year
    """
    base_year = orca.get_injectable("base_year")

    # get current trend df
    df_cur = load_taz_vars_from_orca()
    
    # update taz_hlcm_trend_by_year
    taz_hlcm_trend_by_year[str(year)] = df_cur
    orca.add_injectable('taz_hlcm_trend_by_year', taz_hlcm_trend_by_year)

    # load building_id to zone_id mapping
    b_to_taz = buildings.to_frame(['zone_id']).zone_id

    # define 10yr trend variables
    year_delta = 10
    # define building variables
    cur_year = base_year if year <= base_year+10 else year
    
    # if not exist, use flat trend
    if str(cur_year-year_delta) in taz_hlcm_trend_by_year:
        # generate TAZ trend variables
        prev_df = taz_hlcm_trend_by_year[str(cur_year-year_delta)]
        cur_df = taz_hlcm_trend_by_year[str(cur_year)]
    else:
        prev_df = taz_hlcm_trend_by_year[str(cur_year)]
        cur_df = taz_hlcm_trend_by_year[str(cur_year)]
    diff = cur_df - prev_df

    # Experimental: 
    # * For Dearborn, taz zone 420-472,
    # selected_taz_ids = [idx for idx in range(420, 473) if idx in diff.index]
    # N = len(selected_taz_ids) # total number of applicable TAZs
    # # increase hh_count by 50%, (distributed evenly among TAZs, same method below)
    # diff.loc[selected_taz_ids, 'hh_count'] += (max(diff.loc[selected_taz_ids, 'hh_count'].sum() // 2, 1000 ) // (N)) 
    # # increase hh_pop by 100%pp
    # diff.loc[selected_taz_ids, 'hh_pop'] += (max(diff.loc[selected_taz_ids, 'hh_pop'].sum(), 3000 ) // (N)) 
    # # increase with_children hh by 100%
    # diff.loc[selected_taz_ids, 'with_children'] += (max(diff.loc[selected_taz_ids, 'with_children'].sum(), 1000 ) // (N)) 
    # # reduce one_persons_hh count by 100%
    # diff.loc[selected_taz_ids, 'one_person_hh'] -= (max(diff.loc[selected_taz_ids, 'one_person_hh'].sum(), 1000 ) // (N)) 

    for var in df_cur.columns:
        print("registering building variable", var+"_taz_10yr_change")
        @orca.column("buildings", var+"_taz_10yr_change")
        def func():
            return b_to_taz.map(diff[var]).fillna(0).astype(int)

    # define 5yr trend variables
    year_delta = 5
    # define building variables
    cur_year = base_year if year <= base_year+5 else year
    
    # if not exist, use flat trend
    if str(cur_year-year_delta) in taz_hlcm_trend_by_year:
        # generate TAZ trend variables
        prev_df = taz_hlcm_trend_by_year[str(cur_year-year_delta)]
        cur_df = taz_hlcm_trend_by_year[str(cur_year)]
    else:
        prev_df = taz_hlcm_trend_by_year[str(cur_year)]
        cur_df = taz_hlcm_trend_by_year[str(cur_year)]
    diff = cur_df - prev_df


    # Experimental: 
    # * For Dearborn, taz zone 420-472,
    selected_taz_ids = [idx for idx in range(420, 473) if idx in diff.index]
    N = len(selected_taz_ids) # total number of applicable TAZs
    # increase hh_count by 50%, (distributed evenly among TAZs, same method below)
    diff.loc[selected_taz_ids, 'hh_count'] += (max(diff.loc[selected_taz_ids, 'hh_count'].sum() // 2, 1000 ) // (N)) 
    # increase hh_pop by 100%pp
    diff.loc[selected_taz_ids, 'hh_pop'] += (max(diff.loc[selected_taz_ids, 'hh_pop'].sum(), 2000 ) // (N)) 
    # increase with_children hh by 100%
    diff.loc[selected_taz_ids, 'with_children'] += (max(diff.loc[selected_taz_ids, 'with_children'].sum(), 2000 ) // (N)) 
    # reduce one_persons_hh count by 50%
    diff.loc[selected_taz_ids, 'one_person_hh'] -= (max(diff.loc[selected_taz_ids, 'one_person_hh'].sum() // 2, 1000 ) // (N)) 

    for var in df_cur.columns:
        print("registering building variable", var+"_taz_10yr_change")
        @orca.column("buildings", var+"_taz_10yr_change")
        def func():
            return b_to_taz.map(diff[var]).fillna(0).astype(int)

    # define 5yr trend variables
    year_delta = 5
    # define building variables
    cur_year = base_year if year <= base_year+5 else year
    
    # if not exist, use flat trend
    if str(cur_year-year_delta) in taz_hlcm_trend_by_year:
        # generate TAZ trend variables
        prev_df = taz_hlcm_trend_by_year[str(cur_year-year_delta)]
        cur_df = taz_hlcm_trend_by_year[str(cur_year)]
    else:
        prev_df = taz_hlcm_trend_by_year[str(cur_year)]
        cur_df = taz_hlcm_trend_by_year[str(cur_year)]
    diff = cur_df - prev_df


    # Experimental: 
    # * For Dearborn, taz zone 420-472,
    # selected_taz_ids = [idx for idx in range(420, 473) if idx in diff.index]
    # N = len(selected_taz_ids) # total number of applicable TAZs
    # # increase hh_count by 50%, (distributed evenly among TAZs, same method below)
    # diff.loc[selected_taz_ids, 'hh_count'] += (max(diff.loc[selected_taz_ids, 'hh_count'].sum() // 2, 1000 ) // (N)) 
    # # increase hh_pop by 100%pp
    # diff.loc[selected_taz_ids, 'hh_pop'] += (max(diff.loc[selected_taz_ids, 'hh_pop'].sum(), 3000 ) // (N)) 
    # # increase with_children hh by 100%
    # diff.loc[selected_taz_ids, 'with_children'] += (max(diff.loc[selected_taz_ids, 'with_children'].sum(), 1000 ) // (N)) 
    # # reduce one_persons_hh count by 100%
    # diff.loc[selected_taz_ids, 'one_person_hh'] -= (max(diff.loc[selected_taz_ids, 'one_person_hh'].sum(), 1000 ) // (N)) 

    for var in df_cur.columns:
        print("registering building variable", var+"_taz_5yr_change")
        @orca.column("buildings", var+"_taz_5yr_change")
        def func():
            return b_to_taz.map(diff[var]).fillna(0).astype(int)


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
    Generator function for single-model REPMs (Lasso/YAML-based).
    Kept for reference, not used in production.
    """

    @orca.step(model_name)
    def func():
        buildings = orca.get_table("buildings")
        nodes_walk = orca.get_table("nodes_walk")
        print(yaml_file)
        return utils.hedonic_simulate(yaml_file, buildings, nodes_walk, dep_var)

    return func


def make_xgb_repm_func(model_name, xgb_model_dir, dep_var):
    """
    Generator function for XGBoost-based REPMs.

    Parameters
    ----------
    model_name : str
        Name of the model step (e.g., "res_repm381")
    xgb_model_dir : str
        Path to XGBoost model directory (e.g., "configs/repm_xgb")
    dep_var : str
        Target variable name ("sqft_price_res" or "sqft_price_nonres")
    """

    @orca.step(model_name)
    def func():
        from repm_xgb_utils import load_repm_xgb_model

        buildings = orca.get_table("buildings")

        # Get year with fallback
        year = orca.get_injectable("year") if orca.is_injectable("year") else None

        # Load trained model
        model_wrapper = load_repm_xgb_model(model_name, model_dir=xgb_model_dir)

        # Get hedonic_id from model metadata for filtering
        hedonic_id = int(model_wrapper.metadata['hedonic_id'])

        # Get filter columns from metadata
        size_col = model_wrapper.metadata['size_col']
        price_col = model_wrapper.metadata['price_col']

        # Get all feature names needed by model
        feature_names = model_wrapper.feature_names

        # Load all needed columns from buildings table (with caching via utils)
        needed_cols = list(set(['hedonic_id', size_col, price_col] + feature_names))
        buildings_df = utils.get_cached_buildings_df(buildings, needed_cols, year)

        # Filter to this hedonic segment with valid space.
        # Do NOT filter on price_col — new buildings start at 0 and need pricing.
        filter_mask = (
            (buildings_df['hedonic_id'] == hedonic_id) &
            (buildings_df[size_col] > 0)
        )

        # Check for missing features and fill with 0
        missing_features = set(feature_names) - set(buildings_df.columns)
        if missing_features:
            print(f"  Warning: {len(missing_features)} missing features, filling with 0")
            for feat in missing_features:
                buildings_df[feat] = 0

        # Handle inf/nan values
        buildings_df = buildings_df.replace([np.inf, -np.inf], 0)
        buildings_df = buildings_df.fillna(0)

        # Select features in correct order for filtered buildings only
        X = buildings_df.loc[filter_mask, feature_names]

        if len(X) == 0:
            print(f"  {model_name}: No buildings match filter (hedonic_id={hedonic_id})")
            return pd.Series(dtype=float)

        # Make predictions (log-transformed prices)
        log_prices = model_wrapper.predict(X)

        # Inverse transform: expm1 is inverse of log1p
        prices = np.expm1(log_prices)

        # Create series for update (only for filtered buildings)
        building_indices = buildings_df.index[filter_mask]
        price_series = pd.Series(prices, index=building_indices)

        # Clamp values (same as old system)
        price_series = price_series.clip(lower=1, upper=700)

        # Store predictions for comparison (via utils)
        pred_key = 'res' if dep_var == 'sqft_price_res' else 'nonres'
        utils.add_xgb_prediction(pred_key, hedonic_id, price_series, model_name,
                                 model_wrapper.metadata['metrics']['r2_val'])

        # Update buildings table
        buildings.update_col_from_series(dep_var, price_series, cast=True)

        print(f"  {model_name}: Updated {len(price_series)} buildings (R²={model_wrapper.metadata['metrics']['r2_val']:.4f})")

        return price_series

    return func


@orca.step()
def repm_comparison_log():
    """
    Compare XGBoost vs Lasso predictions and log results.
    Calls the implementation in utils.py.
    """
    utils.repm_comparison_log()


# Register XGBoost REPM steps
repm_step_names = []
xgb_repm_dir = "configs/repm_xgb"

# Use absolute path for checking existence
xgb_model_full_path = os.path.abspath(xgb_repm_dir)
if not os.path.exists(xgb_model_full_path):
    # Try relative to models directory
    xgb_model_full_path = os.path.join(misc.models_dir(), "repm_xgb")
    xgb_repm_dir = os.path.join(misc.models_dir(), "repm_xgb")

if os.path.exists(xgb_model_full_path):
    for model_dir in sorted(os.listdir(xgb_model_full_path)):
        model_path = os.path.join(xgb_model_full_path, model_dir)

        # Skip non-directories and grid search files
        if not os.path.isdir(model_path):
            continue
        if model_dir.startswith('grid_search') or model_dir.startswith('.'):
            continue

        # Check for required metadata file
        metadata_path = os.path.join(model_path, "metadata.pkl")
        if not os.path.exists(metadata_path):
            print(f"Warning: Skipping {model_dir} - no metadata.pkl found")
            continue

        model_name = model_dir

        # Determine dep_var from model name
        if model_name.startswith("res"):
            dep_var = "sqft_price_res"
        elif model_name.startswith("nonres"):
            dep_var = "sqft_price_nonres"
        else:
            print(f"Warning: Unknown model type {model_name}, skipping")
            continue

        # Create step function - pass the parent directory, model_name is the subfolder
        make_xgb_repm_func(model_name, xgb_repm_dir, dep_var)
        repm_step_names.append(model_name)

    # Store model names
    orca.add_injectable("_xgb_repm_model_names", repm_step_names)
    orca.add_injectable("repm_step_names", repm_step_names)

    # Add comparison step after REPM models (disabled for speed)
    # repm_step_names.append("repm_comparison_log")

    print(f"Registered {len(repm_step_names)} XGBoost REPM models (comparison step disabled)")
else:
    print("ERROR: XGBoost REPM directory not found at", xgb_model_full_path)
    orca.add_injectable("repm_step_names", [])


@orca.step()
def real_estate_adjustment(buildings, parcels, year):
    remi_base_year = orca.get_injectable("remi_base_year")
    if year < remi_base_year:
        return
    income_ratios = orca.get_injectable("remi_income_ratios")
    la_ratios = income_ratios.get(year, {})
    if not la_ratios:
        return

    # Capture LA-average base prices on first call; flag used below for one-time backfill
    first_call = not orca.is_injectable("remi_base_la_prices")
    if first_call:
        bd0 = buildings.to_frame(["large_area_id", "sqft_price_res"])
        base = (bd0[bd0["sqft_price_res"] > 0]
                .groupby("large_area_id")["sqft_price_res"].mean()
                .to_dict())
        orca.add_injectable("remi_base_la_prices", base)
        print(f"  [real_estate_adjustment] base LA prices captured at year {year}: "
              + str({la: f"${v:.1f}" for la, v in sorted(base.items())}))

    base_la_prices = orca.get_injectable("remi_base_la_prices")
    bd = buildings.to_frame(["large_area_id", "sqft_price_res", "year_built",
                              "residential_units", "sqft_per_unit", "parcel_id",
                              "building_type_id"])
    bd_pos = bd[bd["sqft_price_res"] > 0]
    updated = bd_pos["sqft_price_res"].copy()
    new_build_idx = []

    # On the first call, treat every simulation-era build (year_built >= base_year+1) as a
    # new build so first-year (placement-year) events get nodes_walk neighborhood pricing
    # rather than being frozen as "existing" stock at REPM's zero-feature underprediction.
    sim_start = orca.get_injectable("base_year") + 1
    new_from = sim_start if first_call else year

    for la, ratio in la_ratios.items():
        base_avg = base_la_prices.get(la, 0.0)
        if base_avg <= 0:
            continue
        target_avg = base_avg * ratio

        mask_la    = bd_pos["large_area_id"] == la
        mask_exist = mask_la & (bd_pos["year_built"] < new_from)
        mask_new   = mask_la & (bd_pos["year_built"] >= new_from)

        # Scale existing buildings: anchor LA avg to income-ratio target
        exist_bldgs = bd_pos[mask_exist]
        if len(exist_bldgs) == 0:
            continue
        exist_avg = exist_bldgs["sqft_price_res"].mean()
        if exist_avg <= 0:
            continue
        scale = target_avg / exist_avg
        updated.loc[mask_exist] = (exist_bldgs["sqft_price_res"] * scale).clip(upper=1000)

        # New builds: use the same parcel-level price the feasibility pro-forma used
        # (nodes_walk["residential"] mapped through parcels.nodeid_walk), scaled to
        # current year. Falls back to exist_avg if walk node unavailable.
        if mask_new.any():
            new_bldgs = bd_pos[mask_new]
            nw_res = orca.get_table("nodes_walk")["residential"]
            pcl_node = parcels.to_frame(["nodeid_walk"])["nodeid_walk"]
            parcel_prices = new_bldgs["parcel_id"].map(pcl_node).map(nw_res)
            prices = (parcel_prices.fillna(exist_avg) * scale).clip(upper=1000)
            updated.loc[mask_new] = prices
            new_build_idx.extend(new_bldgs.index.tolist())

    buildings.update_col_from_series("sqft_price_res", updated, cast=True)

    # Backfill missing attributes (improvement_value=0, market_value=0, land_area=0).
    parcel_sqft = parcels.to_frame(["parcel_sqft"])["parcel_sqft"]

    if first_call:
        # One-time backfill for all pre-existing buildings missing assessed values
        bd_full = buildings.to_frame(["residential_units", "sqft_per_unit",
                                       "parcel_id", "improvement_value"])
        missing_idx = bd_full[
            (bd_full["improvement_value"] == 0) & (bd_full["residential_units"] > 0)
        ].index
        prices_for_missing = updated.reindex(missing_idx)
        valid = prices_for_missing[prices_for_missing > 0].index
        if len(valid) > 0:
            nb_pre = bd_full.loc[valid]
            total_val = (prices_for_missing.loc[valid]
                         * nb_pre["residential_units"].clip(lower=1)
                         * nb_pre["sqft_per_unit"].clip(lower=1))
            # Ratios from base buildings: 
            # market_value = 0.70 × price×units×sqft_per_unit
            # improvement_value = 0.70 × market_value
            market_val = (total_val * 0.70).astype(int)
            buildings.update_col_from_series("market_value",      market_val, cast=True)
            buildings.update_col_from_series("improvement_value", (market_val * 0.70).astype(int), cast=True)
            land_area = nb_pre["parcel_id"].map(parcel_sqft).fillna(0).astype(int)
            buildings.update_col_from_series("land_area", land_area, cast=True)
            print(f"  [real_estate_adjustment] one-time backfill: {len(valid):,} buildings assessed at year {year}")

    if new_build_idx:
        nb = bd.loc[new_build_idx]
        total_val = (updated.loc[new_build_idx]
                     * nb["residential_units"].clip(lower=1)
                     * nb["sqft_per_unit"].clip(lower=1))
        market_val = (total_val * 0.70).astype(int)
        buildings.update_col_from_series("market_value",      market_val, cast=True)
        buildings.update_col_from_series("improvement_value", (market_val * 0.70).astype(int), cast=True)
        land_area = nb["parcel_id"].map(parcel_sqft).fillna(0).astype(int)
        buildings.update_col_from_series("land_area", land_area, cast=True)

    summary = {la: f"{r:.3f}x" for la, r in la_ratios.items() if la in base_la_prices}
    print(f"  [real_estate_adjustment] year={year} income ratios: {summary}"
          + (f"  new builds assessed: {len(new_build_idx)}" if new_build_idx else ""))


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


# ---------------------------------------------------------------------------
# Zero-cell donor gap-filling for the household transition model
# ---------------------------------------------------------------------------
# A control row whose 8-dim category matches no household is silently skipped
# by urbansim's TabularTotalsTransition (its `if len(subset) == 0: continue`),
# so its households — and their population — never appear. The control-total
# script now emits nonzero rows for categories absent in the synthesized base
# year, so the simulation must be able to fill them.
#
# Strategy (mirrors the control-side fallback): for each empty control row,
# find the nearest donor by relaxing the least-structural dimensions, clone it,
# then OVERRIDE the relaxed attributes back to the control row's target bin
# (syncing person records). The donor lends only its retained attributes plus
# its person structure; every dropped dimension is restored to the target.
#
# Relax order = most expendable first. The FIRST dim dropped is synthesized in
# every filled cell (it sits in every non-empty prefix), so it must be the
# lowest-stakes attribute. age_of_head and persons are NEVER relaxed (they fix
# headship and population); large_area_id is never relaxed (same-LA donors only).
#
# Order rationale:
#   cars     - not an HLCM segment, no person edit -> drop first (synth everywhere)
#   workers  - recomputed from persons by workers_adjustment_model each year, so its value
#              here is downstream-reconciled -> cheap to synthesize, drop early
#   income   - HLCM segments on it, but the override samples WITHIN the target
#              income bin so the income segment is preserved regardless -> drop
#              after workers to better keep the income<->workers joint realism
#   race_id  - CONDITIONAL: only relaxed when the (LA, age, race) cell is empty
#              (a genuine demographic gap, e.g. LA 93/115 Black 18-24). For an
#              ordinary fine-combo gap the race is already correct and is never
#              relabeled.
#   children - HLCM segments on has_children AND it is the heaviest person
#              rewrite (re-aging members) -> drop late
#   persons  - last-resort catch-all: resize the donor household to the target
#              size (add/drop members). Only reached when no same-LA, same-age
#              household of any race/structure of that size exists at all. Keeps
#              age_of_head (the sole never-relaxed dim besides large_area), so a
#              same-age head is always retained; population stays correct because
#              the resized size matches the control row's persons bin.
# age_of_head and large_area_id are never relaxed.
# This generalises (and replaces) the hand-maintained step7_fix_zero_samples.
GAP_RELAX_ORDER = ["cars", "workers", "income", "race_id", "children", "persons"]
GAP_RELAX_COLS = {
    "income": ["income_min", "income_max"],
    "cars": ["cars_min", "cars_max"],
    "race_id": ["race_id"],
    "workers": ["workers_min", "workers_max"],
    "children": ["children_min", "children_max"],
    "persons": ["persons_min", "persons_max"],
}
GAP_DONORS_PER_CELL = 25   # distinct donors injected per empty cell, for variety
WORKING_AGE_MIN = 16


def _gap_rand_int(lo, hi, rng):
    """Random int in [lo, hi) where hi is the already-+1'd exclusive max
    (np.inf for open-ended bins, in which case lo is used)."""
    lo = int(lo)
    if not np.isfinite(hi):
        return lo
    hi = int(hi)
    return lo if hi <= lo + 1 else int(rng.integers(lo, hi))


def _gap_rand_income(lo, hi, rng):
    """Random income within the target quartile bin (open top bin -> 1.5x lo)."""
    lo = float(lo)
    if not np.isfinite(hi):
        return lo * 1.5 if lo > 0 else 1000.0
    return float(rng.uniform(lo, float(hi)))


def _gap_sync_workers(pers, target_workers):
    """Set exactly min(target, working-age members) members to worker=1."""
    cand = pers.index[pers["age"] >= WORKING_AGE_MIN]
    target = min(int(target_workers), len(cand))
    pers["worker"] = 0
    if target > 0:
        pers.loc[cand[:target], "worker"] = 1
    pers.loc[pers["worker"] == 0, "industry"] = 0
    return target


def _gap_sync_children(pers, target_children, rng):
    """Adjust ages of non-head members so own-children (<18) == target, holding
    total persons fixed. Head (relate==0) untouched. Heaviest edit (logged)."""
    age_dt = pers["age"].dtype
    non_head = pers.index[pers["relate"] != 0]
    n_child = min(int(target_children), len(non_head))
    child_idx = non_head[:n_child]
    adult_idx = non_head[n_child:]
    if len(child_idx):
        pers.loc[child_idx, "age"] = rng.integers(0, 18, size=len(child_idx)).astype(age_dt)
        pers.loc[child_idx, "worker"] = 0
    young = adult_idx[pers.loc[adult_idx, "age"] < 18]
    if len(young):
        pers.loc[young, "age"] = rng.integers(18, 65, size=len(young)).astype(age_dt)
    return n_child


def _gap_sync_persons(pers, target_persons, rng):
    """Resize a household's person rows to exactly target_persons, keeping the
    single head (relate==0). Drops random non-head members when shrinking, or
    clones existing members (marked non-head) when growing. member_id is
    renumbered. children/workers are re-synced by the caller afterwards."""
    target = int(target_persons)
    cur = len(pers)
    if cur != target:
        rs = int(rng.integers(0, 2**31))
        head = pers[pers["relate"] == 0]
        non_head = pers[pers["relate"] != 0]
        if cur > target:
            keep = max(target - len(head), 0)
            non_head = non_head.sample(min(keep, len(non_head)), random_state=rs) \
                if keep and len(non_head) else non_head.iloc[:0]
            pers = pd.concat([head, non_head])
        else:
            pool = non_head if len(non_head) else head
            add = pool.sample(target - cur, replace=True, random_state=rs).copy()
            add["relate"] = (pers["relate"].dtype.type(2))  # mark as non-head members
            pers = pd.concat([pers, add])
        # cloning with replacement duplicates index labels; reset to a unique
        # index so the downstream children/workers .loc edits are well-defined
        # (the caller reassigns final person_ids anyway).
        pers = pers.reset_index(drop=True)
    pers["member_id"] = np.arange(1, len(pers) + 1, dtype=pers["member_id"].dtype)
    return pers


GAP_RANGED_ATTRS = ["age_of_head", "persons", "children", "cars", "workers", "income"]


def _gap_category_presence(ct, hh):
    """Vectorised empty-cell detector. Bins every household once into its
    control-category key (race_id + the bin-min of each ranged attribute) and
    returns row_is_empty(row). Replaces a per-control-row filter_table scan of
    the full pool, which was ~half the transition step's runtime.

    Exactness guard: the binning is equivalent to filter_table only when each
    attribute's control bins tile contiguously ([min, max) with max == next
    min). The top bin's max may be finite (e.g. the finite persons subset ends
    at 10; the top income quartile is capped at the base-year max income) —
    values at/above it, below the lowest min, or NaN match no row, same as
    filter_table. If any attribute's bins do not tile, falls back to the exact
    per-row filter_table check.
    """
    edges, uppers = {}, {}
    for a in GAP_RANGED_ATTRS:
        pairs = (ct[[f"{a}_min", f"{a}_max"]].drop_duplicates()
                 .sort_values(f"{a}_min").to_numpy(dtype=float))
        mins, maxs = pairs[:, 0], pairs[:, 1]
        if not np.all(maxs[:-1] == mins[1:]):
            # non-contiguous bins -> exact (slow) fallback
            return lambda row: len(
                utils.filter_table(hh, row, ignore={"total_number_of_households"})
            ) == 0
        edges[a], uppers[a] = mins, maxs[-1]

    cols = {"race_id": hh["race_id"].to_numpy(dtype=float)}
    valid = np.ones(len(hh), dtype=bool)
    for a in GAP_RANGED_ATTRS:
        e = edges[a]
        vals = pd.to_numeric(hh[a], errors="coerce").to_numpy(dtype=float)
        pos = np.searchsorted(e, vals, side="right") - 1
        valid &= (pos >= 0) & ~np.isnan(vals) & (vals < uppers[a])
        cols[f"{a}_min"] = e[np.clip(pos, 0, len(e) - 1)]
    key_arr = np.column_stack([cols["race_id"]] + [cols[f"{a}_min"] for a in GAP_RANGED_ATTRS])
    present = set(map(tuple, key_arr[valid]))

    def row_is_empty(row):
        key = (float(row["race_id"]),) + tuple(float(row[f"{a}_min"]) for a in GAP_RANGED_ATTRS)
        return key not in present

    return row_is_empty


def fill_control_gaps(ct, hh, p, iter_var, seed=0):
    """Inject attribute-overridden donor households for empty control rows so
    TabularTotalsTransition can fill them. Returns (hh, p, diagnostics)."""
    rng = np.random.default_rng(seed)
    la = int(hh["large_area_id"].iloc[0]) if "large_area_id" in hh.columns and len(hh) else -1
    next_hid = int(hh.index.max()) + 1
    next_pid = int(p.index.max()) + 1

    row_is_empty = _gap_category_presence(ct, hh)

    hh_records, hh_ids, p_chunks, diags = [], [], [], []

    for ridx, row in ct.iterrows():
        if row["total_number_of_households"] <= 0:
            continue
        if not row_is_empty(row):
            continue  # already fillable

        # Relax race only for a genuine demographic gap: the whole
        # (LA, age_of_head, race) cell empty. For a populated cell the race is
        # already correct and must not be relabeled.
        race_cell_empty = len(hh[
            (hh["race_id"] == row["race_id"])
            & (hh["age_of_head"] >= row["age_of_head_min"])
            & (hh["age_of_head"] < row["age_of_head_max"])
        ]) == 0

        dropped, donors, ignore = [], None, {"total_number_of_households"}
        for attr in GAP_RELAX_ORDER:
            if attr == "race_id" and not race_cell_empty:
                continue  # don't relabel race for a populated (LA, age, race) cell
            dropped.append(attr)
            ignore |= set(GAP_RELAX_COLS[attr])
            cand = utils.filter_table(hh, row, ignore=ignore)
            if len(cand) > 0:
                donors = cand
                break

        if donors is None or len(donors) == 0:
            # Empty even after relaxing every dim except large_area and
            # age_of_head -> no same-LA household of this head-age exists at all
            # (would need a cross-LA donor or relaxing age). Surfaced below.
            cat = {c: row[c] for c in
                   ["race_id", "age_of_head_min", "age_of_head_max",
                    "persons_min", "persons_max"] if c in row}
            diags.append({"row": ridx, "rung": "NO_DONOR", "cat": cat,
                          "target": int(row["total_number_of_households"]), "donors": 0})
            continue

        k = min(GAP_DONORS_PER_CELL, len(donors), int(row["total_number_of_households"]))
        donor_sample = donors.sample(k, random_state=int(rng.integers(0, 2**31)))

        for old_hid, donor in donor_sample.iterrows():
            clone = donor.to_dict()
            dp = p[p["household_id"] == old_hid].copy()

            if "income" in dropped:
                clone["income"] = _gap_rand_income(row["income_min"], row["income_max"], rng)
            if "cars" in dropped:
                clone["cars"] = _gap_rand_int(row["cars_min"], row["cars_max"], rng)
            if "race_id" in dropped:
                clone["race_id"] = int(row["race_id"])
                dp["race_id"] = int(row["race_id"])
            # persons resize first (changes the member set), then children
            # before workers: re-age members, then assign worker flags over the
            # final working-age set. The persons table is the source of truth for
            # whichever count was edited, so the two stay consistent (and survive
            # workers_adjustment_model, which recomputes workers from persons). Counts NOT dropped
            # keep the donor's matching value.
            if "persons" in dropped:
                dp = _gap_sync_persons(
                    dp, _gap_rand_int(row["persons_min"], row["persons_max"], rng), rng)
                clone["persons"] = int(len(dp))
            if "children" in dropped:
                _gap_sync_children(
                    dp, _gap_rand_int(row["children_min"], row["children_max"], rng), rng)
                n_child = int((dp["age"] < 18).sum())
                clone["children"] = n_child
                if "noc" in clone:
                    clone["noc"] = n_child
            if "workers" in dropped:
                _gap_sync_workers(
                    dp, _gap_rand_int(row["workers_min"], row["workers_max"], rng))
                clone["workers"] = int(dp["worker"].sum())

            clone["building_id"] = -1
            new_hid = next_hid
            next_hid += 1
            dp["household_id"] = new_hid
            dp.index = range(next_pid, next_pid + len(dp))
            dp.index.name = "person_id"
            next_pid += len(dp)
            hh_records.append(clone)
            hh_ids.append(new_hid)
            p_chunks.append(dp)

        diags.append({"row": ridx, "rung": "+".join(dropped),
                      "target": int(row["total_number_of_households"]), "donors": k})

    if hh_records:
        add_hh = pd.DataFrame(hh_records, index=pd.Index(hh_ids, name=hh.index.name))
        add_hh = add_hh[hh.columns].astype(hh.dtypes.to_dict())
        add_p = pd.concat(p_chunks).astype(p.dtypes.to_dict())
        hh = pd.concat([hh, add_hh])
        p = pd.concat([p, add_p])

    filled = [d for d in diags if d["rung"] != "NO_DONOR"]
    nodonor = [d for d in diags if d["rung"] == "NO_DONOR"]
    if filled or nodonor:
        rungs = {}
        for d in filled:
            rungs[d["rung"]] = rungs.get(d["rung"], 0) + 1
        resid_hh = sum(d["target"] for d in nodonor)
        print(f"[gap-fill] LA {la} yr {iter_var}: filled {len(filled)} empty control "
              f"cell(s), injected {sum(d['donors'] for d in filled)} donor(s); "
              f"rungs {rungs}; unfillable {len(nodonor)} cell(s) / {resid_hh} hh")
        for d in nodonor:
            print(f"[gap-fill]   NO DONOR (target {d['target']} hh): {d['cat']}")
    return hh, p, diags


def presses_trans(xxx_todo_changeme1):
    (ct, hh, p, target, iter_var, la_seed) = xxx_todo_changeme1
    # Seed this worker's global NumPy RNG from the per-(large_area, year) seed
    # derived in the parent (see households_transition). The UrbanSim core
    # transition draws from the global RNG, so seeding it here makes each LA's
    # result depend only on its own seed + data — reproducible regardless of
    # which pool worker runs it or in what order (fixes the fork-RNG problem),
    # and independent across large areas. A retry of this task re-seeds, so it
    # reproduces the same draw rather than diverging.
    np.random.seed(la_seed)
    ct_finite = ct[ct.persons_max <= 100]
    ct_inf = ct[ct.persons_max > 100]
    # Inject donor households for any finite control cell with no match, so the
    # transition below can realise it instead of silently skipping it. The
    # gap-filler uses its own stream derived from the same per-LA seed.
    hh, p = fill_control_gaps(ct_finite, hh, p, iter_var, seed=la_seed)[:2]
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

    # Unlike the finite cells above, the open-ended 10+ bin is NOT gap-filled:
    # the loop below realises it only by cloning existing 10+ households, so a
    # cell with a positive target but no matching donor produces zero households
    # and is silently skipped. This is intentional for now (synthesizing a
    # household size for an open-ended bin needs a chosen cap) but should be
    # visible. Log any such cell; see the model wiki transition todo for the
    # fuller fix options.
    la = int(hh["large_area_id"].iloc[0]) if "large_area_id" in hh.columns and len(hh) else -1
    for _, r in ct_inf.loc[iter_var].iterrows():
        if r["total_number_of_households"] <= 0:
            continue
        if utils.filter_table(hh, r, ignore={"total_number_of_households"}).shape[0] == 0:
            cat = {c: r[c] for c in
                   ["race_id", "age_of_head_min", "age_of_head_max",
                    "persons_min", "persons_max"] if c in r}
            print(f"[gap-fill] LA {la} yr {iter_var}: open-ended 10+ control cell "
                  f"has no donor household (target "
                  f"{int(r['total_number_of_households'])} hh, not synthesized): {cat}")

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


# Errors worth retrying inside a pool worker — transient resource/IO blips.
# Everything else (logic/data bugs) is deterministic: retrying with identical
# inputs only fails again, so it is re-raised immediately with its real traceback.
_RETRYABLE_WORKER_ERRORS = (OSError, MemoryError)


def _retry_wrapper(args, max_retries=3):
    """Run presses_trans in a pool worker. Retries only transient errors and
    surfaces real bugs immediately with the full traceback. Must stay at module
    level to be picklable for pool.map."""
    import traceback
    for attempt in range(1, max_retries + 1):
        try:
            return presses_trans(args)
        except _RETRYABLE_WORKER_ERRORS:
            print(f"[households_transition] transient worker failure "
                  f"(attempt {attempt}/{max_retries}):\n{traceback.format_exc()}")
            if attempt == max_retries:
                raise  # re-raise the original error, traceback intact
        except Exception:
            # deterministic failure — retrying won't help; fail fast and loud
            print("[households_transition] worker failed (not retrying):\n"
                  + traceback.format_exc())
            raise


def _resolve_hh_pop_target():
    """Resolve the household-population target table used to size the open-ended
    10+-person bin in households_transition.

    Prefers `remi_hh_pop` — household population (TOTAL population minus group
    quarters), the correct quantity. Falls back to the legacy `remi_pop_total`
    (TOTAL population, including group quarters) only while the
    `allow_total_pop_fallback` injectable is true (the default). TOTAL population
    over-states the household-person target by each large area's GQ population,
    which biases 10+-person household sizes upward; set
    `allow_total_pop_fallback = False` once the model input provides
    `remi_hh_pop`, so a wrong (total-pop) file can never silently feed the model.
    """
    if orca.is_table("remi_hh_pop"):
        return orca.get_table("remi_hh_pop")

    allow = (orca.get_injectable("allow_total_pop_fallback")
             if orca.is_injectable("allow_total_pop_fallback") else True)
    if orca.is_table("remi_pop_total"):
        if not allow:
            raise RuntimeError(
                "households_transition: 'remi_hh_pop' not found and "
                "allow_total_pop_fallback is False. The legacy 'remi_pop_total' "
                "holds TOTAL population (incl. group quarters), which over-sizes "
                "10+-person households. Regenerate the model input with a "
                "'remi_hh_pop' table (household population = total - GQ)."
            )
        print("WARNING households_transition: 'remi_hh_pop' not found — falling "
              "back to legacy 'remi_pop_total' (TOTAL population incl. group "
              "quarters). This over-sizes 10+-person households. Provide "
              "'remi_hh_pop' and set allow_total_pop_fallback=False.")
        return orca.get_table("remi_pop_total")

    raise RuntimeError(
        "households_transition: neither 'remi_hh_pop' nor 'remi_pop_total' is "
        "available for the household-population target."
    )


def _assert_control_coverage(region_hh, region_ct, iter_var):
    """P2 guard: flag any current household that matches no control category.

    UrbanSim's TabularTotalsTransition builds its output as the concatenation of
    the per-control-row segments, so a household that falls in *no* control
    category is silently dropped from the table — its population quietly lost.
    Full coverage is an emergent property of how the control totals are built,
    not something the code enforces; a base-year refresh or a control-bin change
    could start dropping households with no signal. This makes it visible.

    Warns by default; set the `require_full_control_coverage` injectable to raise.
    Cheap (vectorised bin + merge), runs once per year.
    """
    ranged = ["age_of_head", "persons", "children", "cars", "workers", "income"]
    ct = region_ct
    if getattr(ct.index, "name", None) == "year" and iter_var in ct.index:
        ct = ct.loc[[iter_var]]
    key_cols = ["large_area_id", "race_id"] + [f"{a}_min" for a in ranged]
    ctrl_keys = ct[key_cols].drop_duplicates()

    # bin each household to its control-category identity (exact LA + race, plus
    # the bin-min each ranged attribute falls into) and look it up in the controls
    binned = {
        "large_area_id": region_hh["large_area_id"].to_numpy(),
        "race_id": region_hh["race_id"].to_numpy(),
    }
    above = np.zeros(len(region_hh), dtype=bool)
    for a in ranged:
        edges = np.sort(ct[f"{a}_min"].unique())
        vals = pd.to_numeric(region_hh[a], errors="coerce").fillna(edges[0])
        vals = vals.clip(lower=edges[0]).to_numpy()
        binned[f"{a}_min"] = edges[np.searchsorted(edges, vals, side="right") - 1]
        # A value at/above the highest bin's _max is snapped into the top bin by
        # the lower-bound search above, so the key merge alone would call it
        # "covered". Flag it as unmatched too, mirroring _gap_category_presence's
        # `vals < uppers[a]` validity test. Open-ended top bins carry _max == inf
        # here (households_transition replaced -1 with inf before this guard), so
        # this never false-positives on them; it only catches a future finite
        # top bin with households above it.
        above |= vals >= float(ct[f"{a}_max"].max())
    hh_keys = pd.DataFrame(binned, index=region_hh.index)

    merged = hh_keys.merge(ctrl_keys.assign(_ok=1), on=key_cols, how="left")
    unmatched = merged["_ok"].isna().to_numpy() | above
    n = int(unmatched.sum())
    if n == 0:
        return
    by_la = region_hh.loc[unmatched].groupby("large_area_id").size().to_dict()
    msg = (f"households_transition: {n} of {len(region_hh)} households match no "
           f"control category for {iter_var} and would be silently dropped by the "
           f"transition (by large area: {by_la}). Check control-total coverage "
           f"against the households table.")
    strict = (orca.get_injectable("require_full_control_coverage")
              if orca.is_injectable("require_full_control_coverage") else False)
    if strict:
        raise RuntimeError(msg)
    print("WARNING " + msg)


def _resolve_hh_pop_target():
    """Resolve the household-population target table used to size the open-ended
    10+-person bin in households_transition.

    Prefers `remi_hh_pop` — household population (TOTAL population minus group
    quarters), the correct quantity. Falls back to the legacy `remi_pop_total`
    (TOTAL population, including group quarters) only while the
    `allow_total_pop_fallback` injectable is true (the default). TOTAL population
    over-states the household-person target by each large area's GQ population,
    which biases 10+-person household sizes upward; set
    `allow_total_pop_fallback = False` once the model input provides
    `remi_hh_pop`, so a wrong (total-pop) file can never silently feed the model.
    """
    if orca.is_table("remi_hh_pop"):
        return orca.get_table("remi_hh_pop")

    allow = (orca.get_injectable("allow_total_pop_fallback")
             if orca.is_injectable("allow_total_pop_fallback") else True)
    if orca.is_table("remi_pop_total"):
        if not allow:
            raise RuntimeError(
                "households_transition: 'remi_hh_pop' not found and "
                "allow_total_pop_fallback is False. The legacy 'remi_pop_total' "
                "holds TOTAL population (incl. group quarters), which over-sizes "
                "10+-person households. Regenerate the model input with a "
                "'remi_hh_pop' table (household population = total - GQ)."
            )
        print("WARNING households_transition: 'remi_hh_pop' not found — falling "
              "back to legacy 'remi_pop_total' (TOTAL population incl. group "
              "quarters). This over-sizes 10+-person households. Provide "
              "'remi_hh_pop' and set allow_total_pop_fallback=False.")
        return orca.get_table("remi_pop_total")

    raise RuntimeError(
        "households_transition: neither 'remi_hh_pop' nor 'remi_pop_total' is "
        "available for the household-population target."
    )


@orca.step()
def households_transition(
    households, persons, annual_household_control_totals, iter_var
):
    region_ct = annual_household_control_totals.to_frame()
    max_cols = region_ct.columns[region_ct.columns.str.endswith("_max")]
    region_ct[max_cols] = region_ct[max_cols].replace(-1, np.inf)
    region_ct[max_cols] += 1
    region_hh = households.to_frame(households.local_columns + ["large_area_id"])
    region_hh.index = region_hh.index.astype(int)

    region_p = persons.to_frame(persons.local_columns)
    region_p.index = region_p.index.astype(int)

    # P2 guard: warn (or raise) if any household matches no control category and
    # would be silently dropped by the totals transition (see function docstring).
    _assert_control_coverage(region_hh, region_ct, iter_var)

    # NOTE (2026-06): the blanket donor injections that used to live here —
    # appending the full hh_seeds/p_seeds set (issue #56) and the previous
    # year's LPR-changed households — were removed. They guaranteed donor
    # availability by flooding the pool (~64k seed households/year), which the
    # totals transition offset by randomly removing real, placed households
    # (hidden churn + drift toward base-year composition). fill_control_gaps
    # (called in presses_trans) now guarantees donor availability surgically,
    # injecting donors only for control cells that would otherwise be empty.
    # Validated: zero unfilled control cells across all large areas without
    # the blanket appends. hh_seeds/p_seeds tables themselves are still
    # maintained by cache_hh_seeds for the labor-participation swaps.

    region_target = _resolve_hh_pop_target().to_frame()

    def cut_to_la(xxx_todo_changeme):
        (large_area_id, hh) = xxx_todo_changeme
        p = region_p[region_p.household_id.isin(hh.index)]
        target = int(region_target.loc[large_area_id, str(iter_var)])
        ct = region_ct[region_ct.large_area_id == large_area_id]
        del ct["large_area_id"]
        # Per-(large_area, year) seed derived in the parent (where the run-level
        # random_seed injectable is available) and passed into the pool worker,
        # which seeds its global RNG with it. Concrete int so the worker needs
        # no orca access.
        la_seed = int(utils.get_rng("households_transition", iter_var, large_area_id)
                      .integers(0, 2**32))
        return ct, hh, p, target, iter_var, la_seed

    arg_per_la = list(map(cut_to_la, region_hh.groupby("large_area_id")))

    pool = Pool(8)
    try:
        cunks_per_la = pool.map(_retry_wrapper, arg_per_la)
    except Exception as e:
        print(f"Pool execution failed: {e}")
        pool.terminate()
        raise
    finally:
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
        if hh_new_added_cumsum[i + 1] > hh_new_added_cumsum[i]:
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

def get_worker_swap_seed_mapping(hh, p, hh_seeds, p_seeds):
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
    first_sim_year = orca.get_injectable("base_year") + 1
    # run if hh_seeds not found
    if iter_var != first_sim_year and orca.is_table("hh_seeds"):
        print('skipping cache_hh_seeds for forecast year')
        return

    # if resume running from forecast year
    if iter_var != first_sim_year and not orca.is_table("hh_seeds"):
        input_hdf = pd.HDFStore(orca.get_injectable("input_hdf_path"), 'r')
        parcels = input_hdf['parcels']
        b = input_hdf['buildings']
        b = b.join(parcels[['large_area_id']], on='parcel_id')
        hh = input_hdf['households']
        hh = hh.join(b[['large_area_id']], on='building_id')
        p = input_hdf['persons']
        p = p.join(hh[['large_area_id']], on='household_id')
        input_hdf.close()
    else:
        hh = households.to_frame(households.local_columns + ["large_area_id"])
        p = persons.to_frame(persons.local_columns + ["large_area_id"])

    # caching hh and persons seeds at the start of the model run
    hh["target_workers"] = 0
    hh['inc_qt'] = pd.qcut(hh.income, 4, labels=[1, 2, 3, 4])
    hh['aoh_bin'] = pd.cut(hh.age_of_head, [-1, 4, 17, 24, 34, 64, 200], labels=[1, 2, 3, 4, 5, 6])
    # generate age bins
    age_bin = [-1, 15, 19, 21, 24, 29, 34, 44, 54, 59, 61, 64, 69, 74, 199]
    age_bin_labels = [0,16,20,22,25,30,35,45,55,60,62,65,70,75,200]
    p['age_bin'] = pd.cut(p.age, age_bin, labels=age_bin_labels[:-1])
    p['age_bin'] = p['age_bin'].fillna(0).astype(int)
    p = p.join(hh.seed_id, on='household_id')
    
    hh_seeds = hh.groupby('seed_id').first()
    p_seeds = p.groupby(['seed_id', 'member_id']).first()
    print('running cache_hh_seeds for base year')
    orca.add_table('hh_seeds', hh_seeds)
    orca.add_table('p_seeds', p_seeds)

@orca.step()
def workers_adjustment_model(households, persons, hh_seeds, p_seeds, iter_var, employed_workers_rate):
    """Adjust household/person worker counts to employed-worker rate targets.

    For each large area and age band, compares the current number of employed
    workers against the target implied by the `employed_workers_rate` table and
    closes the gap by swapping households with a counterpart seed household
    that has one more (or one fewer) worker and matched persons / race /
    age-bin / children. Household `workers` is then recomputed from the person
    `worker` flags, and income/cars of changed households are resampled from
    peers with the same composition and new worker count.

    Formerly named `fix_lpr`; renamed 2026-06. Note the rates are employed-
    worker rates (employment), not labor-force participation rates.
    """
    from numpy.random import choice

    hh = households.to_frame(households.local_columns + ["large_area_id"])
    hh_seeds = hh_seeds.to_frame()
    p_seeds = p_seeds.to_frame()
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

    # Worker-swap seed mappings are expensive to build (~2h); cache as CSV and
    # reuse. (Previously the existence check looked for .pkl files that were
    # never written, and the drop-worker table overwrote the add-worker file,
    # so the cache could never activate.)
    USE_SWAPPING_SEED_MAPPING = True
    aw_path = 'data/add_worker_dict.csv'
    dw_path = 'data/drop_worker_dict.csv'
    if not os.path.exists(aw_path):
        USE_SWAPPING_SEED_MAPPING = False
        print(aw_path, ' not found. running get_worker_swap_seed_mapping')
    if not os.path.exists(dw_path):
        USE_SWAPPING_SEED_MAPPING = False
        print(dw_path, ' not found. running get_worker_swap_seed_mapping')
    if USE_SWAPPING_SEED_MAPPING:
        add_worker_df = pd.read_csv(aw_path, index_col=0)
        add_worker_df.columns = add_worker_df.columns.astype(int)
        drop_worker_df = pd.read_csv(dw_path, index_col=0)
        drop_worker_df.columns = drop_worker_df.columns.astype(int)
    else:
        add_worker_dict, drop_worker_dict = get_worker_swap_seed_mapping(hh, p, hh_seeds, p_seeds)
        add_worker_df = pd.DataFrame(add_worker_dict)
        add_worker_df.to_csv(aw_path, index=True)
        drop_worker_df = pd.DataFrame(drop_worker_dict)
        drop_worker_df.to_csv(dw_path, index=True)

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
                hh_to_swap = hh_swap_pool.sample(
                    to_add, replace=False,
                    random_state=utils.step_rng("workers_adjustment_add", large_area_id, row.age_min))
                # target seed_ids
                target_hh_seed_id = hh_to_swap.seed_id.map(add_swappable)
                # overwrite old attributes except building_id, large_area_id, blkgrp
                hh_src = hh_seeds.loc[target_hh_seed_id].reset_index()[hh_cols_to_swap]
                for _col in hh_cols_to_swap:
                    hh.loc[hh_to_swap.index, _col] = hh_src[_col].values.astype(hh[_col].dtype)
                # hh persons overwrite
                p_idx_to_update = np.array([], dtype=int)
                for hh_id in hh_to_swap.index:
                    hh_members = pg.get_group(hh_id)
                    p_idx_to_update = np.concatenate((p_idx_to_update, hh_members.index))
                p_src = p_seeds.loc[target_hh_seed_id].reset_index()[p_cols_to_swap]
                for _col in p_cols_to_swap:
                    p.loc[p_idx_to_update, _col] = p_src[_col].values.astype(p[_col].dtype)
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
                hh_to_swap = hh_swap_pool.sample(
                    to_drop, replace=False,
                    random_state=utils.step_rng("workers_adjustment_drop", large_area_id, row.age_min))
                # target seed_ids
                target_hh_seed_id = hh_to_swap.seed_id.map(drop_swappable)
                # overwrite old attributes except building_id, large_area_id, blkgrp
                hh_src = hh_seeds.loc[target_hh_seed_id].reset_index()[hh_cols_to_swap]
                for _col in hh_cols_to_swap:
                    hh.loc[hh_to_swap.index, _col] = hh_src[_col].values.astype(hh[_col].dtype)
                # hh persons overwrite
                p_idx_to_update = np.array([], dtype=int)
                for hh_id in hh_to_swap.index:
                    hh_members = pg.get_group(hh_id)
                    p_idx_to_update = np.concatenate((p_idx_to_update, hh_members.index))
                p_src = p_seeds.loc[target_hh_seed_id].reset_index()[p_cols_to_swap]
                for _col in p_cols_to_swap:
                    p.loc[p_idx_to_update, _col] = p_src[_col].values.astype(p[_col].dtype)
                # update num_drop_employ
                num_drop_employ = int((
                    (p.large_area_id == large_area_id)
                    & (p.age >= row.age_min)
                    & (p.age <= row.age_max)
                    & (p.worker == 1)
                ).sum() - lpr_workers)

        achieved = int((
            (p.large_area_id == large_area_id)
            & (p.age >= row.age_min)
            & (p.age <= row.age_max)
            & (p.worker == True)
        ).sum())
        print(f"[workers_adjustment] LA {large_area_id} age {row.age_min}-{row.age_max}: "
              f"workers {num_workers} -> {achieved} (target {lpr_workers})")
        # swap pool can exhaust before the target is met — surface the shortfall
        # rather than letting it pass silently
        if abs(achieved - lpr_workers) > max(1, int(0.01 * lpr_workers)):
            print(f"WARNING [workers_adjustment] LA {large_area_id} age "
                  f"{row.age_min}-{row.age_max}: employment target not met "
                  f"(achieved {achieved}, target {lpr_workers}, "
                  f"shortfall {lpr_workers - achieved}) — swap pool likely exhausted")

    hh["old_workers"] = hh.workers
    hh.workers = p.groupby("household_id").worker.sum()
    hh.workers = hh.workers.fillna(0)
    changed = hh.workers != hh.old_workers
    print(f"worker counts changed for {int(changed.sum())} of {len(changed)} households")

    # NOTE (2026-06): the changed_hhs/changed_ps tables that were saved here as
    # extra transition donors were removed together with the blanket seed
    # append in households_transition — fill_control_gaps now guarantees donor
    # availability in the transition directly.

    # Resample income/cars for changed households from peers with the same
    # (persons, race, NEW worker count, children, large area), so household
    # economics stay consistent with the adjusted worker count.
    resample_missed = 0
    resample_rng = utils.step_rng("workers_adjustment_resample")
    for match_colls, chh in hh[changed].groupby(colls):
        try:
            match = same[tuple(match_colls)]
            new_workers = resample_rng.choice(match.index, len(chh), True)
            hh.loc[chh.index, ["income", "cars"]] = match.loc[
                new_workers, ["income", "cars"]
            ].values
        except KeyError:
            # no peer group with this combination — income/cars kept as-is
            resample_missed += len(chh)
    if resample_missed:
        print(f"income/cars resample: no peer group for {resample_missed} "
              "changed households (values left unchanged)")

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

    # Defensive: a GQ resident whose building_id no longer resolves to a city
    # (city_id is derived via building_id -> building -> parcel, so a demolished
    # building yields NaN) would be silently skipped by the city loop below and
    # linger as an orphan row pointing at a nonexistent building. Drop them.
    invalid_city = gqpop["city_id"].isna()
    n_invalid = int(invalid_city.sum())
    if n_invalid:
        print(
            "gq_pop_scaling_model: dropping %d GQ residents with unresolved city_id "
            "(building_id missing from buildings, e.g. demolished)" % n_invalid
        )
        gqpop = gqpop[~invalid_city]

    # gqhh = group_quarters_households.to_frame(group_quarters_households.local_columns)
    target_gq = group_quarters_control_totals.to_frame()
    target_gq = target_gq[target_gq.year == year]

    # if no control found, skip this year
    if target_gq.shape[0] == 0:
        print("Warning: No gq controls found for year %s, skipping..." % year)
        return

    # add gq target to city table to iterate (NaN where a city has no control row)
    city_large_area["gq_target"] = target_gq["count"]

    # Defensive: any city that currently has GQ pop must have a control row,
    # otherwise the "no control -> skip" filter below leaves it untouched. Warn
    # loudly rather than silently zeroing it (the previous fillna(0) path would
    # have deleted its non-protected GQ pop).
    control_cities = set(target_gq.index)
    gq_cities_missing_control = [
        c for c in gqpop["city_id"].dropna().unique() if c not in control_cities
    ]
    if gq_cities_missing_control:
        print(
            "WARNING gq_pop_scaling_model: %d city(ies) have GQ pop but no %s control "
            "row; leaving them unchanged: %s"
            % (len(gq_cities_missing_control), year, sorted(gq_cities_missing_control))
        )

    # Only scale cities that have a control row; a NaN target means no control,
    # so skip (do not fillna(0) and delete).
    city_large_area = city_large_area[city_large_area["gq_target"].notna()].sort_index()

    for city_id, row in city_large_area.iterrows():
        local_gqpop = gqpop.loc[gqpop.city_id == city_id]
        diff = int(row.gq_target - len(local_gqpop))
        # diff = target_gq.loc[city_id]["count"] - len(local_gqpop)
        # keep certain GQ pop unchanged
        filtered_gqpop = filter_local_gq(local_gqpop)

        if len(local_gqpop) > 0:
            if len(filtered_gqpop) == 0:
                filtered_gqpop = local_gqpop

            # deterministic per-city stream: reproducible across runs and local
            # (a change in one city does not perturb another's draws)
            rng = utils.get_rng("gq_pop_scaling_model", year, city_id)

            if diff > 0:
                # diff = int(min(len(filtered_gqpop), abs(diff)))
                # if no existing GQ except protected, use large area sample

                # local_gqpop = gqpop.loc[gqpop.large_area_id == row.large_area_id]
                # filtered_gqpop = filter_local_gq(local_gqpop)

                newgq = filtered_gqpop.sample(diff, replace=True, random_state=rng)
                newgq.index = gqpop.index.values.max() + 1 + np.arange(len(newgq))
                newgq["city_id"] = city_id
                gqpop = pd.concat((gqpop, newgq))

            elif diff < 0:
                diff = min(len(filtered_gqpop), abs(diff))
                removegq = filtered_gqpop.sample(diff, replace=False, random_state=rng)
                gqpop.drop(removegq.index, inplace=True)

    print("%s gqpop after scaling" % gqpop.shape[0])
    print(
        "\tgq result - target",
        (gqpop.groupby("city_id").size().fillna(0) - city_large_area.gq_target).sum(),
    )

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

        agents = pd.concat([agents, agents_sample])
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
                agents = pd.concat([agents, agents_sample])
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
                agents = pd.concat([agents, agents_sample])
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
            agents_pool = pd.concat([agents_pool, selected_agents], ignore_index=True)
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
            agents_pool = pd.concat([agents_pool, selected_agents], ignore_index=True)
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
            agents_pool = pd.concat([agents_pool, selected_agents], ignore_index=True)
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
def scheduled_development_events(buildings, iter_var, events_addition, refiner_events):
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
        sched_dev["sp_filter"] = 0
        sched_dev["event_id"] = ebid  # add back event_id

        # set sp_filter to -1 to nonres event with refiner events to prevent future reloaction
        refinements = refiner_events.to_frame()
        refinements = refinements[refinements.year >= iter_var]
        for _, record in refinements.iterrows():
            dev_w_ref = sched_dev[sched_dev.non_residential_sqft > 0].query(record.location_expression)
            if len(dev_w_ref) > 0:
                sched_dev.loc[dev_w_ref.index, "sp_filter"] = -1
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
):
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
        # reset pct_undev to 0 for parcels left with no buildings after demolition
        remaining_pids = set(new_buildings_table.parcel_id)
        parcels_idx_to_update = [
            pid
            for pid in set(drop_buildings.parcel_id)
            if pid not in remaining_pids
        ]
        pct_undev_update = pd.Series(0, index=parcels_idx_to_update)
        parcels.update_col_from_series("pct_undev", pct_undev_update, cast=True)


@orca.step()
def random_demolition_events(
    buildings, parcels, households, jobs, year, demolition_rates
):
    demolition_rates = demolition_rates.to_frame()
    base_year = orca.get_injectable("base_year")
    final_year = orca.get_injectable("final_year")
    demolition_rates *= 0.1 + (1.0 - 0.1) * (final_year - year) / (final_year - base_year)
    buildings_columns = buildings.local_columns
    buildings = buildings.to_frame(
        buildings.local_columns + ["city_id"] + ["b_total_jobs", "b_total_households"]
    )

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
                rel_b = rel_b.sample(size, weights=rel_b[weights],
                                     random_state=utils.step_rng("mcd_hu_sampling_res", city_id))
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
    # reset pct_undev to 0 for parcels left with no buildings after demolition
    remaining_pids = set(new_buildings_table.parcel_id)
    parcels_idx_to_update = [
        pid
        for pid in set(drop_buildings.parcel_id)
        if pid not in remaining_pids
    ]
    pct_undev_update = pd.Series(0, index=parcels_idx_to_update)
    parcels.update_col_from_series("pct_undev", pct_undev_update, cast=True)


# ── Demolition scoring helpers ────────────────────────────────────────────────

_DEMO_CFG_CACHE = {}   # avoids re-reading YAML every simulation year

def _load_demolition_cfg():
    path = os.path.join(misc.configs_dir(), "demolition_model.yaml")
    mtime = os.path.getmtime(path)
    if _DEMO_CFG_CACHE.get("mtime") != mtime:
        with open(path) as f:
            _DEMO_CFG_CACHE["cfg"]   = yaml.load(f, Loader=yaml.FullLoader)
            _DEMO_CFG_CACHE["mtime"] = mtime
    return _DEMO_CFG_CACHE["cfg"]


def _logistic_score(df, section, features):
    """Return a Series of logistic probabilities for rows in df."""
    means     = section["feature_mean"]
    stds      = section["feature_std"]
    coefs     = section["coef"]
    intercept = section["intercept"]
    score     = np.full(len(df), intercept, dtype=float)
    for feat in features:
        if feat not in df.columns:
            continue
        x = (df[feat].fillna(means.get(feat, 0.0)) - means.get(feat, 0.0)) \
            / max(stds.get(feat, 1.0), 1e-8)
        score += coefs.get(feat, 0.0) * x
    prob = 1.0 / (1.0 + np.exp(-np.clip(score, -15.0, 15.0)))
    return pd.Series(prob, index=df.index)


_DEMO_PRESSURE_CACHE = {}   # per-city baseline + rolling history for the dynamic pressure index


def _city_pressure_signals(b_eligible):
    """Per-city means of three endogenous 'teardown pressure' proxies.

    Older average age, higher vacancy, and a higher land-vs-improvement
    value ratio all correlate with elevated demolition risk in the
    calibration data -- aggregating them to city level gives a proxy for
    how a city's redevelopment pressure is shifting, built entirely from
    data the simulation already maintains and updates every year.
    """
    g = b_eligible.groupby("city_id")
    return pd.DataFrame({
        "age":  g["building_age"].mean(),
        "vac":  g["res_vacancy_rate"].mean(),
        "ltir": g["land_to_impr_ratio"].mean(),
    })


def _demolition_pressure_ratio(b_eligible, year, base_year, window=5):
    """City-level multiplier reflecting how 'teardown pressure' has moved
    relative to the base year, smoothed to damp feedback-loop oscillation.

    demolition_rates is a STATIC snapshot (one row per city_id, no year
    dimension): left alone it reproduces the same relative geography for
    every simulated year. This ratio lets RELATIVE emphasis across cities
    drift with conditions the simulation already tracks (aging stock,
    vacancy, land-to-improvement value) while the calibrated baseline still
    anchors the overall pattern (ratio == 1.0 in the base year). See
    "Dynamic Pressure Index" in docs/models/demolition.md for the rationale,
    the damping design, and known limitations of this proxy.
    """
    signals = _city_pressure_signals(b_eligible)

    if "baseline" not in _DEMO_PRESSURE_CACHE:
        _DEMO_PRESSURE_CACHE["baseline"] = signals
        _DEMO_PRESSURE_CACHE["history"]  = {}

    base = _DEMO_PRESSURE_CACHE["baseline"]
    ratios = pd.DataFrame({
        col: signals[col] / base[col].clip(lower=1e-6)
        for col in signals.columns
    }).reindex(base.index)
    composite = ratios.mean(axis=1).fillna(1.0)

    # Rolling average over `window` years damps year-to-year sampling noise
    # and the demolition -> vacancy -> demolition feedback loop; the clip
    # keeps a single anomalous year from whipsawing the control totals.
    hist = _DEMO_PRESSURE_CACHE["history"]
    hist[year] = composite
    recent = [hist[y] for y in sorted(hist) if y > year - window]
    smoothed = pd.concat(recent, axis=1).mean(axis=1)
    return smoothed.clip(lower=0.5, upper=2.0)


@orca.step()
def scored_demolition_events(buildings, parcels, households, jobs, year, demolition_rates, base_year, final_year):
    """
    Scored replacement for random_demolition_events.

    Buildings are ranked by a logistic regression score (age, improvement value
    per sqft, land-to-improvement ratio, tax-exempt status, Wayne County flag)
    and sampled to hit city-level control totals from demolition_rates.

    When configs/demolition_model.yaml has is_calibrated=false the function
    falls back to the original inverse-log-occupancy weights so the simulation
    runs unchanged until calibration_demolition.py has been executed.
    """
    cfg              = _load_demolition_cfg()
    res_calibrated    = cfg.get("residential", {}).get("is_calibrated", False)
    nonres_calibrated = cfg.get("nonresidential", {}).get("is_calibrated", False)
    rate_mult         = cfg.get("scenario_rate_multiplier", 1.0)
    min_age           = cfg.get("min_age_eligible", 10)
    max_res_occ       = cfg.get("max_res_occupancy_eligible")
    max_nonres_occ    = cfg.get("max_nonres_occupancy_eligible")
    any_calibrated    = res_calibrated or nonres_calibrated

    buildings_columns = buildings.local_columns
    b = buildings.to_frame(
        buildings_columns + [
            "city_id", "b_total_jobs", "b_total_households", "job_spaces",
            "building_age", "vacant_residential_units", "res_vacancy_rate",
            "impr_value_per_sqft", "land_to_impr_ratio",
        ]
    )

    # Wayne County flag needed by logistic model (large_area_id is on parcels)
    pcl = parcels.to_frame(["large_area_id"])
    b["is_wayne"]  = b.parcel_id.map(pcl.large_area_id).eq(5).astype(float)

    # ── Eligibility ──────────────────────────────────────────────────────────
    not_eligible = pd.Series(False, index=b.index)
    if any_calibrated:
        not_eligible |= (b.building_age < min_age) | (b.sp_filter < 0) | (b.event_id > 0)
        if max_res_occ is not None:
            res_occ = 1.0 - b["res_vacancy_rate"]
            not_eligible |= (b.residential_units > 0) & (res_occ > max_res_occ)
        if max_nonres_occ is not None:
            nonres_occ = b["b_total_jobs"] / b["job_spaces"].clip(lower=1)
            not_eligible |= (b.non_residential_sqft > 0) & (nonres_occ > max_nonres_occ)
    eligible = ~not_eligible

    # ── Control totals: taper × scenario multiplier × dynamic pressure ──────
    # Taper matches random_demolition_events: demolition volume scales from
    # ~100% at base_year down to 10% at final_year; scenario_rate_multiplier
    # layers on top (defaults to 1.0, so fallback mode reproduces the legacy
    # trajectory exactly). Uses orca's base_year/final_year injectables rather
    # than hardcoded 2020/2050 so this still works on the forecast_2055 horizon.
    taper = 0.1 + (1.0 - 0.1) * (final_year - year) / (final_year - base_year)

    # demolition_rates is a STATIC city-level snapshot with no year dimension --
    # on its own it reproduces the SAME relative geography for every simulated
    # year. _demolition_pressure_ratio lets that geography drift with
    # conditions the simulation already tracks each year (aging stock,
    # vacancy, land-to-improvement value), anchored to the calibrated
    # baseline so it only shifts RELATIVE emphasis across cities, not the
    # overall volume (the taper still governs that). See "Dynamic Pressure
    # Index" in docs/models/demolition.md for the full rationale.
    demolition_rates = demolition_rates.to_frame()
    if any_calibrated:
        pressure = _demolition_pressure_ratio(b[eligible], year, base_year)
        adjusted_rates = demolition_rates.mul(pressure, axis=0).fillna(demolition_rates)
        base_totals = demolition_rates.sum(axis=0)
        adjusted_totals = adjusted_rates.sum(axis=0).replace(0, np.nan)
        demolition_rates = adjusted_rates.mul(base_totals / adjusted_totals, axis=1).fillna(0)
    demolition_rates = demolition_rates * taper * rate_mult

    # ── Demolition scores ────────────────────────────────────────────────────
    if res_calibrated:
        res_score    = _logistic_score(b, cfg["residential"],    ["building_age", "impr_value_per_sqft", "land_to_impr_ratio", "is_exempt", "is_wayne"])
    else:
        # Fallback: inverse-log-occupancy (identical to old random_demolition_events)
        res_score    = 1.0 / (1.0 + np.log1p(b["b_total_households"]))

    if nonres_calibrated:
        nonres_score = _logistic_score(b, cfg["nonresidential"], ["building_age", "impr_value_per_sqft", "land_to_impr_ratio", "is_exempt", "is_wayne"])
    else:
        # Fallback: inverse-log-occupancy (identical to old random_demolition_events)
        nonres_score = 1.0 / (1.0 + np.log1p(b["b_total_jobs"]))

    b["res_score"]    = res_score.where(eligible, 0.0)
    b["nonres_score"] = nonres_score.where(eligible, 0.0)

    # ── Sampling ─────────────────────────────────────────────────────────────
    allowed   = variables.parcel_is_allowed_2050()
    allowed_b = b.parcel_id.isin(allowed[allowed].index)

    buildings_idx = []

    def sample(targets, type_b, accounting, score_col):
        for city_id, target in targets[targets > 0].items():
            rel_b = type_b[type_b.city_id == city_id]
            rel_b = rel_b[rel_b[accounting] <= target]
            if len(rel_b) == 0:
                continue
            w    = rel_b[score_col].fillna(1e-6).clip(lower=1e-6)
            size = min(len(rel_b), int(target))
            if size > 0:
                # Use numpy Generator.choice for weighted sampling without
                # replacement; pandas 3 imposes size*max_weight<=1 which
                # fails when one building has a dominant score.
                rng = utils.step_rng("mcd_hu_sampling_nonres", city_id)
                w_arr = w.to_numpy(dtype=float, copy=True)
                w_arr /= w_arr.sum()
                chosen = rng.choice(len(rel_b), size=size, replace=False, p=w_arr)
                sampled = rel_b.iloc[chosen]
                sampled = sampled[sampled[accounting].cumsum() <= int(target)]
                buildings_idx.append(sampled)

    nonres_eligible = b.loc[allowed_b & eligible]
    sample(
        demolition_rates.typenonsqft,
        nonres_eligible[nonres_eligible.non_residential_sqft > 0],
        "non_residential_sqft",
        "nonres_score",
    )
    res_eligible = b.loc[allowed_b & eligible & (b.non_residential_sqft == 0)]
    sample(demolition_rates.type81units, res_eligible[res_eligible.building_type_id == 81], "residential_units", "res_score")
    sample(demolition_rates.type82units, res_eligible[res_eligible.building_type_id == 82], "residential_units", "res_score")
    sample(demolition_rates.type83units, res_eligible[res_eligible.building_type_id == 83], "residential_units", "res_score")

    if not buildings_idx:
        return

    drop_buildings = pd.concat(buildings_idx).copy()
    drop_buildings = drop_buildings[~drop_buildings.index.duplicated(keep="first")]
    buildings_idx  = drop_buildings.index
    drop_buildings["year_demo"]           = year
    drop_buildings["demolition_pathway"]  = "scored" if any_calibrated else "legacy"
    drop_buildings["step"]                = "scored_demolition_events"

    if orca.is_table("dropped_buildings"):
        prev_drops = orca.get_table("dropped_buildings").to_frame()
        orca.add_table("dropped_buildings", pd.concat([drop_buildings, prev_drops]))
    else:
        orca.add_table("dropped_buildings", drop_buildings)

    new_buildings_table = b[buildings_columns].drop(buildings_idx)
    orca.add_table("buildings", new_buildings_table)

    households = households.to_frame(households.local_columns)
    households.loc[households.building_id.isin(buildings_idx), "building_id"] = -1
    orca.add_table("households", households)

    jobs = jobs.to_frame(jobs.local_columns)
    jobs.loc[jobs.building_id.isin(buildings_idx), "building_id"] = -1
    orca.add_table("jobs", jobs)

    parcels_idx_to_update = [
        pid
        for pid in set(drop_buildings.parcel_id)
        if pid not in remaining_pids
    ]
    pct_undev_update = pd.Series(0, index=parcels_idx_to_update)
    parcels.update_col_from_series("pct_undev", pct_undev_update, cast=True)

    print(
        f"scored_demolition_events {year}: dropped {len(buildings_idx):,} buildings "
        f"({'calibrated' if any_calibrated else 'legacy weights'})"
    )


def parcel_average_price(use):
    parcels_wrapper = orca.get_table("parcels")
    if len(orca.get_table("nodes_walk")) == 0:
        return pd.Series(index=parcels_wrapper.index)
    cfg = orca.get_injectable("btype_form_map")
    form_to_col = {f: col for col, v in cfg.items() for f in v["forms"]}
    col = form_to_col.get(use, use)
    return misc.reindex(orca.get_table("nodes_walk")[col], parcels_wrapper.nodeid_walk)


@orca.injectable("cost_shifters")
def shifters():
    with open(os.path.join(misc.configs_dir(), "cost_shifters.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cfg


def cost_shifter_callback(self, form, df, costs):
    yr = orca.get_injectable("year")
    pce_ratios = orca.get_injectable("remi_pce_ratios")
    costs = costs * pce_ratios.get(yr, 1.0)
    if form in orca.get_injectable("res_forms"):
        return costs
    shifter_cfg = orca.get_injectable("cost_shifters")["calibration"]
    geography = shifter_cfg["calibration_geography_id"]
    shifters = shifter_cfg["proforma_cost_shifters"]["non_residential"]
    for geo, geo_df in df.reset_index().groupby(geography):
        shifter = shifters.get(geo, 1.0)
        costs[:, geo_df.index] *= shifter
    return costs


@orca.step("feasibility")
def feasibility(parcels, buildings, btype_form_map):
    bldgs = buildings.to_frame(
        ["large_area_id", "sqft_price_res", "sqft_price_nonres", "building_type_id",
         "parcel_id", "residential_units"]
    )
    pcl_la_s = parcels.to_frame(["large_area_id"])["large_area_id"]

    # Enforce single-building constraint for SF parcels (land_use_type_id=11).
    sf_parcel_ids = set(
        parcels.to_frame(["land_use_type_id"])
        .query("land_use_type_id == 11").index
    )
    occupied_sf = (
        bldgs[(bldgs["parcel_id"].isin(sf_parcel_ids)) & (bldgs["residential_units"] > 0)]
        ["parcel_id"].unique()
    )
    parcels.update_col_from_series(
        "pct_undev", pd.Series(100, index=occupied_sf), cast=True
    )

    # Build per-nodes-column floor data: {col: (parcel_thr, parcel_rep, reg_avg)}
    form_to_col = {f: col for col, v in btype_form_map.items() for f in v["forms"]}
    _floor = {}
    for col, v in btype_form_map.items():
        bld = bldgs[bldgs["building_type_id"].isin(v["btypes"]) & (bldgs[v["price_col"]] > 0)]
        la_avg = bld.groupby("large_area_id")[v["price_col"]].mean()
        reg_avg = float(bld[v["price_col"]].mean()) if len(bld) else 0.0
        if col == "residential":
            # distressed-market treatment: floor LAs below 80% of regional avg;
            # Detroit (LA5) uses Wayne County (LA3) avg as replacement
            la_rep = la_avg.clip(lower=reg_avg * 0.8).copy()
            la_rep[5] = float(la_avg.get(3, reg_avg))
        else:
            la_rep = la_avg.copy()
        _floor[col] = (
            pcl_la_s.map(la_avg),
            pcl_la_s.map(la_rep).fillna(reg_avg),
            reg_avg,
        )

    # log residential floor summary
    thr_res, _, reg_avg_res = _floor["residential"]
    raw_res = parcel_average_price("apartment")
    n_below = (raw_res < thr_res.reindex(raw_res.index)).sum()
    bld_res = bldgs[bldgs["building_type_id"].isin(btype_form_map["residential"]["btypes"]) & (bldgs["sqft_price_res"] > 0)]
    la_avg_res = bld_res.groupby("large_area_id")["sqft_price_res"].mean()
    print(f"  [feasibility] res: {n_below:,} parcels below LA avg → floored; "
          f"LA5 replacement=${la_avg_res.get(3, reg_avg_res):.0f} (=LA3 avg); "
          f"reg avg=${reg_avg_res:.0f}")

    def _price_with_floor(use):
        prices = parcel_average_price(use)
        col = form_to_col.get(use, use)
        if col not in _floor:
            return prices
        thr, rep, reg_avg = _floor[col]
        prices = prices.where(
            prices >= thr.reindex(prices.index),
            rep.reindex(prices.index, fill_value=reg_avg)
        )
        return prices

    parcel_utils.run_feasibility(
        parcels,
        _price_with_floor,
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


# Each proforma form maps 1:1 to a building_type_id — no stochastic sampling needed.
def probable_type(row):
    """Return building_type_id for a new building — direct lookup from form_to_btype."""
    form = row["form"]
    return orca.get_injectable("form_to_btype")[form][0]


def build_parcel_selection_features(parcels_df, buildings_df, zones_df, year,
                                    lookback_years=7):
    """Compute derived parcel-level features for developer site selection scoring.

    Matches the feature engineering in estimate_developer_selection.py so that
    simulation-time scoring is consistent with estimation.

    Parameters
    ----------
    parcels_df   : parcel DataFrame with zone_id, census_bg_id, recent_mover_rate,
                   and walk-distance amenity columns
    buildings_df : buildings DataFrame with parcel_id, year_built, residential_units
    zones_df     : zones DataFrame with percent_vacant_residential_units,
                   jobs_within_30_min, transit_jobs_30min
    year         : current simulation year (used for lookback window)

    Returns
    -------
    DataFrame indexed by parcel_id with columns:
        local_vacancy, development_momentum, accessibility_composite, recent_mover_rate
    """
    def _get(df, col):
        return df[col] if col in df.columns else pd.Series(np.nan, index=df.index)

    def _norm(s):
        s = s.fillna(s.median() if not s.isna().all() else 0.0)
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.0, index=s.index)
        return (s - mn) / (mx - mn)

    feat = pd.DataFrame(index=parcels_df.index)

    # local_vacancy: zone-level pct vacant residential units
    if "zone_id" in parcels_df.columns and "percent_vacant_residential_units" in zones_df.columns:
        feat["local_vacancy"] = parcels_df["zone_id"].map(
            zones_df["percent_vacant_residential_units"]
        )
    else:
        feat["local_vacancy"] = np.nan

    # development_momentum: annualised recent residential units per block group
    lookback_year = year - lookback_years
    recent_res = buildings_df[
        (buildings_df["year_built"] >= lookback_year) &
        (buildings_df["residential_units"] > 0)
    ]
    if "census_bg_id" in parcels_df.columns:
        recent_with_bg = recent_res.join(
            parcels_df[["census_bg_id"]], on="parcel_id", how="left"
        )
        bg_rate = (
            recent_with_bg.groupby("census_bg_id")["residential_units"].sum()
            / max(lookback_years, 1)
        )
        feat["development_momentum"] = (
            parcels_df["census_bg_id"].map(bg_rate).fillna(0)
        )
    else:
        feat["development_momentum"] = np.nan

    # accessibility_composite: weighted combo of drive/transit jobs and walk distances
    if "zone_id" in parcels_df.columns:
        drive_jobs   = parcels_df["zone_id"].map(_get(zones_df, "jobs_within_30_min"))
        transit_jobs = parcels_df["zone_id"].map(_get(zones_df, "transit_jobs_30min"))
    else:
        drive_jobs   = pd.Series(np.nan, index=parcels_df.index)
        transit_jobs = pd.Series(np.nan, index=parcels_df.index)

    feat["accessibility_composite"] = (
        0.35 * _norm(drive_jobs)
      + 0.20 * _norm(transit_jobs)
      - 0.20 * _norm(_get(parcels_df, "grocery_stores_walk_near_max90"))
      - 0.15 * _norm(_get(parcels_df, "fixed_route_bus_walk_near_max90"))
      - 0.10 * _norm(_get(parcels_df, "schools_k8_walk_near_max90"))
    )

    # recent_mover_rate
    feat["recent_mover_rate"] = _get(parcels_df, "recent_mover_rate")

    # log_parcel_sqft
    if "parcel_sqft" in parcels_df.columns:
        feat["log_parcel_sqft"] = np.log1p(parcels_df["parcel_sqft"].clip(lower=0))
    else:
        feat["log_parcel_sqft"] = np.nan

    # bldg_impr_land_ratio: improvement value / land value (low → underutilised)
    if "bldgimprval" in parcels_df.columns and "landvalue" in parcels_df.columns:
        land = parcels_df["landvalue"].clip(lower=1)
        feat["bldg_impr_land_ratio"] = (parcels_df["bldgimprval"] / land).clip(upper=20)
    else:
        feat["bldg_impr_land_ratio"] = np.nan

    # land_use_type_id: pass-through for per-LUT model dispatch
    feat["land_use_type_id"] = _get(parcels_df, "land_use_type_id")

    return feat


def _score_with_model(feasibility, model_entry, parcel_features_df):
    """Score a feasibility slice using one serialised logistic regression model.

    Returns a pd.Series of selection probabilities (softmax of linear utility).
    """
    feature_cols = model_entry["features"]
    coef_arr     = np.array(model_entry["coef"])
    intercept    = float(model_entry["intercept"])
    mean_arr     = np.array(model_entry["scaler_mean"])
    std_arr      = np.array(model_entry["scaler_std"])

    feat = pd.DataFrame(index=feasibility.index)

    avail = [c for c in feature_cols if c in parcel_features_df.columns]
    if avail:
        feat = feat.join(parcel_features_df[avail], how="left")

    feat = feat.reindex(columns=feature_cols).fillna(0.0)
    X_sc = (feat.values.astype(float) - mean_arr) / np.where(std_arr > 0, std_arr, 1.0)
    utility = X_sc.dot(coef_arr) + intercept
    u_shifted = utility - utility.max()
    exp_u = np.exp(u_shifted)
    return pd.Series(exp_u / exp_u.sum(), index=feasibility.index)


def compute_demo_rebuild_boost(parcels, year, boost, decay, window, scheduled_factor=1.0):
    """Time-decayed site-selection boost for orphaned demolished SF parcels.

    Eligible parcels are SF-zoned (land_use_type_id == 11) residential lots that were
    demolished within `window` years, are currently empty, and have no future
    events_addition entry to rebuild them.  Weight = boost * decay^(years_since_demo),
    so a lot just demolished ranks highest and priority fades over the window — mirroring
    real tear-down/rebuild timing.

    Parcels whose demolition came from a scheduled event (a planned teardown for
    redevelopment) are prioritised over random-attrition demolitions: while they remain
    unbuilt, their weight is multiplied by `scheduled_factor`.

    Parameters
    ----------
    parcels : orca.DataFrameWrapper
        Parcels table; provides `land_use_type_id` for the SF (== 11) filter.
    year : int
        Current simulation year.
    boost : float
        Peak multiplier applied in the year of demolition (age 0).
    decay : float
        Per-year decay factor in [0, 1]; weight = boost * decay^years_since_demo.
    window : int
        Number of years a demolished parcel stays eligible for the boost.
    scheduled_factor : float, default 1.0
        Extra multiplier for parcels demolished by a scheduled event.

    Returns
    -------
    pd.Series
        float multiplier indexed by parcel_id (int); empty Series if nothing to boost.
    """
    empty = pd.Series(dtype=float)
    if not orca.is_table("dropped_buildings"):
        return empty
    db = orca.get_table("dropped_buildings").to_frame(
        ["parcel_id", "year_demo", "residential_units", "step"]
    )
    lut = parcels.to_frame(["land_use_type_id"])["land_use_type_id"]
    demo = db[
        (db["residential_units"] > 0)
        & (db["year_demo"] >= year - window)
        & (db["parcel_id"].map(lut) == 11)
    ]
    if len(demo) == 0:
        return empty
    # most-recent demolition per parcel drives the decay
    demo_year = demo.groupby("parcel_id")["year_demo"].max()
    # flag parcels that had a scheduled (planned) demolition — prioritised over random
    scheduled_pids = set(
        demo.loc[demo["step"] == "scheduled_demolition_events", "parcel_id"]
    )
    # currently empty: drop parcels that hold a building now (incl. already rebuilt)
    occupied = set(orca.get_table("buildings").to_frame(["parcel_id"])["parcel_id"])
    demo_year = demo_year[~demo_year.index.isin(occupied)]
    # no future scheduled event: the events pipeline will rebuild those, skip them
    if orca.is_table("events_addition"):
        ea = orca.get_table("events_addition").to_frame(["parcel_id", "year_built"])
        future_evt = set(ea.loc[ea["year_built"] > year, "parcel_id"])
        demo_year = demo_year[~demo_year.index.isin(future_evt)]
    if len(demo_year) == 0:
        return empty
    age = (year - demo_year).clip(lower=0)
    weight = boost * (decay ** age)
    # lift scheduled-demolition parcels above random-attrition ones
    sched_mask = weight.index.isin(scheduled_pids)
    weight[sched_mask] *= scheduled_factor
    return weight


def make_res_selection_func(lut_models, parcel_features_df, demo_boost=None, demo_max_share=0.5):
    """Return a custom_selection_func closure for residential developer site selection.

    Uses per-LUT logistic regression coefficients estimated from revealed developer
    choices.  Each parcel is scored by the model trained on its land_use_type_id;
    LUTs without a dedicated model fall back to the pooled model.

    Parameters
    ----------
    lut_models : dict
        Content of the "residential" key from developer_selection_coefs.yaml.
        Keys: "fallback" (always present) and integer LUT IDs for active LUTs.
    parcel_features_df : pd.DataFrame
        Pre-loaded parcel-level feature columns (index = parcel_id).
        Must include "land_use_type_id" for per-LUT dispatch.
    demo_boost : pd.Series or None
        Optional multiplier (index = parcel_id) applied to selection probabilities to
        prioritise orphaned demolished SF parcels.  See compute_demo_rebuild_boost.
    demo_max_share : float
        Cap on the combined post-boost probability mass of boosted parcels, so they
        cannot starve normal demand-driven development.
    """
    fallback = lut_models["fallback"]
    per_lut  = {k: v for k, v in lut_models.items() if k != "fallback"}

    def score(dev, df, p, target_units):
        """custom_selection_func signature: (Developer, df, p, target_units) → build_idx."""
        from developer import proposal_select

        probs = pd.Series(0.0, index=df.index)

        if "land_use_type_id" in parcel_features_df.columns and per_lut:
            lut_col = parcel_features_df["land_use_type_id"].reindex(df.index)
            for lut_id, grp_idx in lut_col.groupby(lut_col).groups.items():
                model_entry = per_lut.get(int(lut_id), fallback)
                slice_probs = _score_with_model(df.loc[grp_idx], model_entry, parcel_features_df)
                probs.loc[grp_idx] = slice_probs.values
        else:
            probs = _score_with_model(df, fallback, parcel_features_df)

        p_arr = probs.values.astype(float)

        # Time-decayed rebuild priority for orphaned demolished SF parcels
        boosted = None
        if demo_boost is not None and len(demo_boost):
            mult = demo_boost.reindex(df.index).fillna(1.0).values
            p_arr = p_arr * mult
            boosted = mult > 1.0

        # Re-normalise to a numpy array that sums to exactly 1.0
        p_arr = np.clip(p_arr, 0.0, None)
        total = p_arr.sum()
        if total > 0:
            p_arr /= total
        else:
            p_arr = np.ones(len(p_arr)) / len(p_arr)

        # Cap boosted parcels' combined mass so they can't dominate the whole MCD
        if boosted is not None and boosted.any():
            demo_mass = p_arr[boosted].sum()
            other_mass = 1.0 - demo_mass
            if demo_mass > demo_max_share and other_mass > 0:
                p_arr[boosted]  *= demo_max_share / demo_mass
                p_arr[~boosted] *= (1.0 - demo_max_share) / other_mass

        # Correct any floating-point residual so np.random.choice is satisfied
        p_arr[-1] += 1.0 - p_arr.sum()

        return proposal_select.weighted_random_choice(df, p_arr, target_units)

    return score


def _calculate_pct_undev(parcels, parcels_idx_to_update, year):
    """
    get realistic pct_undev coverage
    """
    if not parcels_idx_to_update:
        return
    bldgs = orca.get_table("buildings").to_frame(
        ["parcel_id", "residential_units", "sqft_per_unit",
         "non_residential_sqft", "stories", "year_built"]
    )
    new_b = bldgs[
        (bldgs["year_built"] == year) &
        (bldgs["parcel_id"].isin(parcels_idx_to_update))
    ].copy()
    if len(new_b) == 0:
        return
    new_b["building_sqft"] = (
        new_b["residential_units"] * new_b["sqft_per_unit"]
        + new_b["non_residential_sqft"]
    )
    new_b["footprint"] = new_b["building_sqft"] / new_b["stories"].clip(lower=1)
    coverage = (
        new_b.groupby("parcel_id")["footprint"].sum()
        / parcels.parcel_sqft.reindex(new_b["parcel_id"].unique())
        * 100
    ).clip(0, 100)
    current_pct = parcels.pct_undev.reindex(coverage.index)
    new_pct = (current_pct + coverage).clip(0, 100).astype("int16")
    parcels.update_col_from_series("pct_undev", new_pct, cast=True)


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

    print("{:,} feasible buildings before running developer".format(len(dev.feasibility)))

    # weighted_random_choice uses max_profit/parcel_size as probability weights.
    # Zero parcel_size → inf weight; multiple inf → NaN probabilities → crash.
    # Negative/zero max_profit also invalid. Filter both before pick().
    if profit_to_prob_func is None and custom_selection_func is None:
        ps = parcel_size.reindex(dev.feasibility.index).fillna(0)
        dev.feasibility = dev.feasibility[
            (dev.feasibility["max_profit"] > 0) & (ps > 0)
        ]

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
    # calculate spaces_added
    if supply_fname == 'job_spaces':
        spaces_added = new_buildings.job_spaces.sum() - new_buildings.current_units.sum()
    else:
        # default to residential units
        spaces_added = new_buildings.residential_units.sum() - new_buildings.current_units.sum()
    # return the number of units added and the list of parcel_id for updating pct_undev
    return (
        spaces_added,
        pid_need_updates,
    )


@orca.injectable("res_developer_selection_coefs", cache=True)
def res_developer_selection_coefs():
    coef_path = os.path.join(misc.configs_dir(), "developer_selection_coefs.yaml")
    if not os.path.exists(coef_path):
        return None
    with open(coef_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


@orca.step("residential_developer")
def residential_developer(
    households, parcels, target_vacancies_mcd, debug_res_developer, res_forms
):
    """
    Simulate residential development per MCD in three steps:

    1. Target units for MCD: blend two signals in housing units
         target_raw = w_gap*V + w_demand*R
         V = vacancy_gap_signed, R = mover_index × decay(year)
         decay: 1.0 in base_year → mover_decay_final in final_year, linear
       Hard ceiling: feasible_units from pro-forma feasibility.

    2. LA alignment: scale each large_area's MCD targets so total built
       does not exceed la_max_ratio × LA 7-yr rolling rate.

    3. Site selection: using per-LUT logistic site selection.
    """
    # get current year
    year = orca.get_injectable("year")

    # get target vacancies by mcd for current year
    target_vacancies = target_vacancies_mcd.to_frame()[str(year)]

    orig_buildings = orca.get_table("buildings").to_frame(
        ["residential_units", "semmcd", "building_type_id",
         "year_built", "recent_mover_rate"]
    )

    with open(os.path.join(misc.configs_dir(), "res_developer.yaml")) as f:
        _res_cfg = yaml.load(f, Loader=yaml.FullLoader)
    w_gap    = _res_cfg.get("target_weight_vacancy_gap", 0.5)
    w_demand = _res_cfg.get("target_weight_demand",      0.5)
    la_max_ratio         = _res_cfg.get("la_max_ratio", 1.2)
    hist_floor_factor    = _res_cfg.get("hist_floor_factor", 0.5)
    hist_floor_min_rate  = _res_cfg.get("hist_floor_min_rate", 30)
    demo_rebuild_boost     = _res_cfg.get("demo_rebuild_boost", 20.0)
    demo_rebuild_decay     = _res_cfg.get("demo_rebuild_decay", 0.6)
    demo_rebuild_window    = _res_cfg.get("demo_rebuild_window", 5)
    demo_rebuild_max_share = _res_cfg.get("demo_rebuild_max_share", 0.5)
    demo_rebuild_sched_factor = _res_cfg.get("demo_rebuild_scheduled_factor", 3.0)
    lookback_year = year - 7

    mover_decay_final = _res_cfg.get("mover_decay_final", 0.5)
    base_year  = orca.get_injectable("base_year")
    # C signal decays linearly from full weight in base year to mover_decay_final by final year
    mover_decay = 1.0 - (1.0 - mover_decay_final) * (year - base_year) / 30

    # compute per-MCD build rates for hist_floor (7-year window ending at base year)
    _hist = orig_buildings[orig_buildings.year_built.between(base_year - 6, base_year)]
    mcd_hist_rate = (_hist.groupby("semmcd")["residential_units"].sum() / 7.0).to_dict()

    # rebuild-priority weight for orphaned demolished SF parcels
    recent_demo_boost = compute_demo_rebuild_boost(
        parcels, year, demo_rebuild_boost, demo_rebuild_decay, demo_rebuild_window,
        demo_rebuild_sched_factor
    )
    print(f"  Rebuild priority: boosting {len(recent_demo_boost):,} orphaned demolished SF parcels")

    # build site-selection func from estimated coefs; fall back to profit-rank if missing
    _coefs = orca.get_injectable("res_developer_selection_coefs")
    custom_sel_func = None
    if _coefs and "residential" in _coefs:
        _rawp = orca.get_table("parcels").to_frame([
            "zone_id", "census_bg_id", "recent_mover_rate",
            "grocery_stores_walk_near_max90", "fixed_route_bus_walk_near_max90",
            "schools_k8_walk_near_max90", "parcel_sqft", "bldgimprval",
            "landvalue", "land_use_type_id",
        ])
        _rawb = orca.get_table("buildings").to_frame(
            ["parcel_id", "year_built", "residential_units"]
        )
        _z = orca.get_table("zones").to_frame([
            "percent_vacant_residential_units", "jobs_within_30_min", "transit_jobs_30min",
        ])
        _p = build_parcel_selection_features(
            _rawp, _rawb, _z, year
        )
        _lut_models = _coefs["residential"]
        custom_sel_func = make_res_selection_func(
            _lut_models, _p, recent_demo_boost, demo_rebuild_max_share
        )
        n_lut = len([k for k in _lut_models if k != "fallback"])
        print(f"  Using estimated site-selection model (fallback + {n_lut} per-LUT models)")
    else:
        print("  No site-selection coefs found — using profit-rank fallback")

    debug_res_developer = debug_res_developer.to_frame()

    # LA rates for step 2 alignment
    pcl_la = orca.get_table("parcels").to_frame(["semmcd", "large_area_id"])
    mcd_to_la = pcl_la.groupby("semmcd")["large_area_id"].first()
    orig_buildings_la = orig_buildings.copy()
    orig_buildings_la["large_area_id"] = orig_buildings_la["semmcd"].map(mcd_to_la)
    # rolling rate excludes current year (events/refiner are not yet "history")
    la_sim_rate = (
        orig_buildings_la[orig_buildings_la.year_built.between(lookback_year, year - 1)]
        .groupby("large_area_id")["residential_units"].sum() / 7.0
    )
    # historical baseline — floor for LA cap so it can't self-deflate to zero
    la_hist_rate = (
        orig_buildings_la[orig_buildings_la.year_built.between(base_year - 6, base_year)]
        .groupby("large_area_id")["residential_units"].sum() / 7.0
    )
    # units already added this year by events/refiner before developer runs
    la_events = (
        orig_buildings_la[orig_buildings_la.year_built == year]
        .groupby("large_area_id")["residential_units"].sum()
    )

    # compute per-MCD target_units
    mcd_data = {}
    for mcdid, _ in parcels.semmcd.to_frame().groupby("semmcd"):

        mcd_orig_buildings = orig_buildings[orig_buildings.semmcd == mcdid]

        if mcdid not in target_vacancies.index:
            continue
        target_vacancy = float(target_vacancies[mcdid])

        cur_agents = int((households.semmcd == mcdid).sum())
        num_units  = int(mcd_orig_buildings.residential_units.sum())
        assert target_vacancy < 1.0

        vacancy_gap_signed = cur_agents / (1.0 - target_vacancy) - num_units
        mover_index = (mcd_orig_buildings.recent_mover_rate * mcd_orig_buildings.residential_units).sum()
        mover_index /= 10 # 10yr avg

        V_units = vacancy_gap_signed
        R_units = max(0, mover_index) * mover_decay
        target_units_raw = w_gap * V_units + w_demand * R_units

        feas_df   = orca.get_table("feasibility_" + str(mcdid)).to_frame()
        res_feas  = feas_df[feas_df["form"].isin(res_forms)]
        profitable = res_feas[res_feas["max_profit"] > 0]
        if len(profitable) > 0:
            ave_unit_sz = parcels.ave_unit_size.reindex(profitable.index).fillna(1000)
            feasible_units = int(
                (profitable["residential_sqft"] / ave_unit_sz).clip(lower=0).sum()
            )
        else:
            feasible_units = 0
        # all_feasible: includes suboptimal proposals (keep_suboptimal=True)
        # used by historical floor to allow distressed-market builds
        if len(res_feas) > 0:
            ave_unit_sz_all = parcels.ave_unit_size.reindex(res_feas.index).fillna(1000)
            all_feasible_units = int(
                (res_feas["residential_sqft"] / ave_unit_sz_all).clip(lower=0).sum()
            )
        else:
            all_feasible_units = 0

        target_units = int(np.clip(target_units_raw, 0, feasible_units))

        mcd_data[mcdid] = {
            "target_units": target_units, "feasible_units": feasible_units,
            "all_feasible_units": all_feasible_units,
            "cur_agents": cur_agents,
            "num_units": num_units, "vacancy_gap": int(vacancy_gap_signed),
            "mover_index": mover_index, "V_units": V_units,
            "R_units": R_units,
            "target_raw": target_units_raw, "la_scale": 1.0,
        }

    # historical minimum floor — before LA alignment so LA cap is the hard ceiling
    n_floored = 0
    for mcdid, d in mcd_data.items():
        hist_rate = mcd_hist_rate.get(mcdid, 0)
        if hist_rate < hist_floor_min_rate:
            continue
        floor = int(hist_rate * hist_floor_factor)
        if d["target_units"] < floor:
            prev = d["target_units"]
            # use all_feasible_units (incl. suboptimal) so distressed markets aren't
            # blocked by the profitable-only cap
            d["target_units"] = min(floor, d["all_feasible_units"])
            if d["target_units"] > prev:
                n_floored += 1
    if n_floored:
        print(f"  Historical floor applied to {n_floored} MCDs "
              f"(factor={hist_floor_factor}, min_rate={hist_floor_min_rate})")

    # LA alignment — hard ceiling applied after hist_floor
    for la_id, la_mcd_ids in mcd_to_la.groupby(mcd_to_la).groups.items():
        active = [m for m in la_mcd_ids if m in mcd_data]
        if not active:
            continue
        la_sum    = sum(mcd_data[m]["target_units"] for m in active)
        # floor la_rate at hist_floor_factor × historical baseline (Fix 2: prevents cap collapse)
        la_rate   = max(float(la_sim_rate.get(la_id, 0)),
                        float(la_hist_rate.get(la_id, 0)) * hist_floor_factor)
        # subtract units already built by events/refiner this year (Fix 1: no double-count)
        la_ev     = float(la_events.get(la_id, 0))
        la_allowed = max(0, la_rate * la_max_ratio - la_ev)
        if la_sum <= 0:
            continue
        if la_allowed <= 0:
            # events already met or exceeded LA cap — zero out all dev targets
            for m in active:
                mcd_data[m]["target_units"] = 0
                mcd_data[m]["la_scale"] = 0.0
            print("  LA {}: events={:.0f} ≥ cap={:.0f} → dev=0 (all MCD targets zeroed)".format(
                la_id, la_ev, la_rate * la_max_ratio))
            continue
        scale = float(min(la_allowed / la_sum, la_max_ratio))
        if abs(scale - 1.0) < 1e-4:
            continue
        print("  LA {}: raw_sum={:,} la_rate={:.0f} la_ev={:.0f} → scale={:.3f}".format(
            la_id, la_sum, la_rate, la_ev, scale))
        for m in active:
            d = mcd_data[m]
            d["target_units"] = int(np.clip(d["target_units"] * scale, 0, d["feasible_units"]))
            d["la_scale"] = round(scale, 4)

    # snapshot units before developer loop to isolate developer additions from refiner
    _units_before = orca.get_table("buildings").to_frame(["year_built", "residential_units"])
    _units_before = int(_units_before[_units_before["year_built"] == year]["residential_units"].sum())

    # start building
    for mcdid, d in mcd_data.items():
        target_units  = d["target_units"]
        feasible_units = d["feasible_units"]

        print(
            "developing residential for MCD {} | "
            "agents={:,} units={:,} vac_gap={:+,} | "
            "V={:+.0f} R={:.0f} raw={:.0f} | "
            "feasible={:,} la_scale={:.3f} target={:,}\n".format(
                mcdid,
                d["cur_agents"], d["num_units"], d["vacancy_gap"],
                d["V_units"], d["R_units"], d["target_raw"],
                feasible_units, d["la_scale"], target_units,
            )
        )

        units_added, parcels_idx_to_update = run_developer(
            target_units,
            mcdid,
            res_forms,
            orca.get_table("buildings"),
            "residential_units",
            parcels.parcel_size,
            parcels.ave_unit_size,
            parcels.total_units,
            "res_developer.yaml",
            add_more_columns_callback=add_extra_columns_res,
            custom_selection_func=custom_sel_func,
        )

        _calculate_pct_undev(parcels, parcels_idx_to_update, year)

        debug_res_developer = pd.concat(
            [debug_res_developer, pd.DataFrame([{
                "year":          year,
                "mcd":           mcdid,
                "cur_agents":    d["cur_agents"],
                "num_units":     d["num_units"],
                "vacancy_gap":   d["vacancy_gap"],
                "mover_index":   round(d["mover_index"], 3),
                "V_units":       round(d["V_units"], 1),
                "R_units":       round(d["R_units"], 1),
                "target_raw":    round(d["target_raw"], 1),
                "la_scale":      d["la_scale"],
                "feasible_units": feasible_units,
                "target_units":  target_units,
                "units_added":   units_added,
            }])],
            ignore_index=True,
        )
        if units_added < target_units:
            print(
                " ***  Not enough housing units built for MCD %s, target: %s, built: %s"
                % (mcdid, target_units, int(units_added))
            )

    # ── annual log ────────────────────────────────────────────────────────────
    nb = orca.get_table("buildings").to_frame(["year_built", "residential_units", "building_type_id", "parcel_id"])
    nb = nb[nb["year_built"] == year].copy()
    nb["large_area_id"] = nb["parcel_id"].map(pcl_la["large_area_id"])

    # reg_7yr = rolling 7yr average of all prior-year builds (developer + events)
    reg_7yr  = orig_buildings[
        (orig_buildings.year_built >= lookback_year) & (orig_buildings.year_built < year)
    ].residential_units.sum() / 7.0
    la_total = nb.groupby("large_area_id")["residential_units"].sum()
    # la_events already computed above (units added by events/refiner before developer ran)
    la_dev   = (la_total.subtract(la_events, fill_value=0)).clip(lower=0)
    reg_dev  = int(la_dev.sum())
    run_name = os.path.basename(orca.get_injectable("data_out_dir")) if orca.is_injectable("data_out_dir") else "test"
    utils.log_res_developer_year(year, reg_7yr, reg_dev, la_hist_rate, la_dev, la_events, nb, run_name)

    # log the target and result in this year's run
    orca.add_table("debug_res_developer", debug_res_developer)


@orca.step()
def non_residential_developer(jobs, parcels, target_vacancies, nonres_forms):
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
    year = orca.get_injectable("year")
    # get target vacancies
    target_vacancies = target_vacancies.to_frame()
    target_vacancies = target_vacancies[
        target_vacancies.year == year
    ]

    # get original buildings table
    orig_buildings = orca.get_table("buildings").to_frame(
        ["job_spaces", "large_area_id", "building_type_id"]
    )

    orig_jobs = jobs.to_frame(['building_id', 'home_based_status', 'large_area_id'])
    orig_jobs = orig_jobs[orig_jobs.home_based_status == 0]

    # loop through large area
    for lid, _ in parcels.large_area_id.to_frame().groupby("large_area_id"):
        # get large area buildings
        la_orig_buildings = orig_buildings[orig_buildings.large_area_id == lid]

        # get current large area vacancy target
        target_vacancy = float(
            target_vacancies[
                target_vacancies.large_area_id == lid
            ].non_res_target_vacancy_rate.iloc[0]
        )

        # loop through non-residential building forms (1:1 with building_type_id)
        for form in nonres_forms:
            form_btype_ids = orca.get_injectable("form_to_btype")[form]
            form_blds = la_orig_buildings[la_orig_buildings.building_type_id.isin(form_btype_ids)]
            # number of non-homebased jobs in the large area
            num_agents = (
                    (orig_jobs.large_area_id == lid) & 
                    (orig_jobs.building_id.isin(form_blds.index))
                ).sum()
            # number of total job spaces for LA
            num_units = form_blds.job_spaces.sum()

            print(f"Developing {form} spaces for large area {lid}:")
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

            # run nonres developer step
            spaces_added, parcels_idx_to_update = run_developer(
                target_units,
                lid,
                [form],
                orca.get_table("buildings"),
                "job_spaces",
                parcels.parcel_size,
                parcels.ave_unit_size,
                parcels.total_job_spaces,
                "nonres_developer.yaml",
                add_more_columns_callback=add_extra_columns_nonres,
            )

            _calculate_pct_undev(parcels, parcels_idx_to_update, year)


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
        "configs/available_networks_2050.yaml", "r"
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
        st = pd.HDFStore(input_paths.NETWORKS_2050_H5, "r")
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
    idx_invalid_building_id = np.isin(j.building_id, b.index.values) == False
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
    idx_invalid_building_id = np.isin(h.building_id, b.index.values) == False
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
    Last used during RDF2050

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
    bb.loc[hhs_by_pseudo_b.index, "residential_units"] = hhs_by_pseudo_b.astype(bb["residential_units"].dtype)

    print("Dropped %s hh from current pseudo buildings." % k)

    # update households and buildings
    orca.add_table("households", hh[households.local_columns]) # remove extra columns
    orca.add_table("buildings", bb)


@orca.step()
def refine_housing_units(households, buildings, mcd_total):
    """ Refine housing units before mcd_hu_sampling to allow it matching mcd_total or 
    total unplaced households depends on which one is larger

    Args:
        households (DataFrame Wrapper): households
        buildings (DataFrame Wrapper): buildings
        mcd_total (DataFrame Wrapper): mcd_total
    """
    year = orca.get_injectable("year")
    b = buildings.to_frame(
        buildings.local_columns + [
            "hu_filter", "sp_filter", "semmcd", 
            "large_area_id", "vacant_residential_units"
        ]
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
        add_hu = int(row["diff"] * 1.2)
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

    # TODO: ensure LA has enough HU for unplaced HH
    la_ids = b.large_area_id.unique()
    h = households.local
    for la_id in la_ids:
        # getting placeable empty housing units
        la_empty_units = b[(b.large_area_id == la_id) & (b.hu_filter == 0) & (
            b.sp_filter >= 0)].vacant_residential_units.sum()
        # getting unplaced households count
        la_unplaced_hh = h[(h.large_area_id == la_id) & (h.building_id == -1)].shape[0]
        print( "%s: la_empty_units %s la_unplaced_hh %s" % (la_id, la_empty_units, la_unplaced_hh))
        if la_empty_units < la_unplaced_hh: 
            # not enough la_empty_units for unplaced hhs
            # sample LA housing units to match
            diff = la_unplaced_hh - la_empty_units
            local_units = housing_units.loc[
                (housing_units.building_type_id.isin([81, 82, 83]))
                & (housing_units.large_area_id == la_id)
            ]
            # filter out hu_filter and sp_filter
            local_units = local_units[local_units["hu_filter"] == 0]
            local_units = local_units[local_units["sp_filter"] >= 0]
            print( "%s missing %s HU: total housing units of %s" % (la_id, diff, local_units.sum()))
            new_units = local_units.sample(
                int(diff), replace=False, random_state=1).index.value_counts()
            b.loc[new_units.index, "residential_units"] += new_units
            print(
                "Adding %s units to large_area %s, actually added %s"
                % (diff, la_id, new_units.sum())
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
