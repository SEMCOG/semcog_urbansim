

import os
import copy
import time
import yaml
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import torch
from forecast_estimation.models.LCM_torch import LCM_NN
from forecast_estimation.utils import (std_scaler_transform, robust_scaler_transform,
                                       min_max_scaler_transform, apply_scaler_state)

import orca
import utils
from urbansim.utils import misc
from urbansim.models import dcm
from urbansim.models import util
from urbansim.urbanchoice import interaction
from urbansim.models import MNLDiscreteChoiceModel
from urbansim_templates.models import LargeMultinomialLogitStep
from urbansim.models.util import (apply_filter_query, columns_in_filters, 
        columns_in_formula)



def random_choices(model, choosers, alternatives):
    """
    Simulate choices using random choice, weighted by probability
    but not capacity constrained.
    Parameters
    ----------
    model : SimulationChoiceModel
        Fitted model object.
    choosers : pandas.DataFrame
        DataFrame of choosers.
    alternatives : pandas.DataFrame
        DataFrame of alternatives.
    Returns
    -------
    choices : pandas.Series
        Mapping of chooser ID to alternative ID.
    """
    probabilities = model.calculate_probabilities(choosers, alternatives)
    # per-(segment, year) stream (model.name is the segment id) so a change in
    # one LCM segment doesn't perturb another's choices
    rng = utils.step_rng("lcm_random_choice", getattr(model, "name", ""))
    choices = rng.choice(
        probabilities.index, size=len(choosers),
        replace=True, p=probabilities.values)
    return pd.Series(choices, index=choosers.index)


def unit_choices(model, choosers, alternatives):
    """
    Simulate choices using unit choice.  Alternatives table is expanded
    to be of length alternatives.vacant_variables, then choices are simulated
    from among the universe of vacant units, respecting alternative capacity.
    Parameters
    ----------
    model : SimulationChoiceModel
        Fitted model object.
    choosers : pandas.DataFrame
        DataFrame of choosers.
    alternatives : pandas.DataFrame
        DataFrame of alternatives.
    Returns
    -------
    choices : pandas.Series
        Mapping of chooser ID to alternative ID.
    """
    supply_variable, vacant_variable = (model.supply_variable,
                                        model.vacant_variable)

    available_units = alternatives[supply_variable]
    vacant_units = alternatives[vacant_variable]
    # must have positive index
    vacant_units = vacant_units[vacant_units.index.values >= 0]

    print("There are {} total available units"
          .format(available_units.sum()),
          "    and {} total choosers"
          .format(len(choosers)),
          "    but there are {} overfull alternatives"
          .format(len(vacant_units[vacant_units < 0])))

    vacant_units = vacant_units[vacant_units > 0]

    indexes = np.repeat(vacant_units.index.values,
                        vacant_units.values.astype('int'))
    isin = pd.Series(indexes).isin(alternatives.index)
    missing = len(isin[isin == False])  # noqa
    indexes = indexes[isin.values]
    units = alternatives.loc[indexes].reset_index()

    print("    for a total of {} temporarily empty units"
          .format(vacant_units.sum()),
          "    in {} alternatives total in the region"
          .format(len(vacant_units)))

    if missing > 0:
        print(
            "WARNING: {} indexes aren't found in the locations df -"
            .format(missing),
            "    this is usually because of a few records that don't join ",
            "    correctly between the locations df and the aggregations",
            "tables")

    print("There are {} total movers for this LCM".format(len(choosers)))

    if len(choosers) > vacant_units.sum():
        print("WARNING: Not enough locations for movers",
              "reducing locations to size of movers for performance gain")
        choosers = choosers.head(int(vacant_units.sum()))

    choices = model.predict(choosers, units, debug=True)

    def identify_duplicate_choices(choices):
        choice_counts = choices.value_counts()
        return choice_counts[choice_counts > 1].index.values

    if model.choice_mode == 'individual':
        print('Choice mode is individual, so utilizing lottery choices.')

        chosen_multiple_times = identify_duplicate_choices(choices)

        while len(chosen_multiple_times) > 0:
            duplicate_choices = choices[choices.isin(chosen_multiple_times)]

            # Identify the choosers who keep their choice, and those who must
            # choose again.
            keep_choice = duplicate_choices.drop_duplicates()
            rechoose = duplicate_choices[~duplicate_choices.index.isin(
                                                           keep_choice.index)]

            # Subset choices, units, and choosers to account for occupied
            # units and choosers who need to choose again.
            choices = choices.drop(rechoose.index)
            units_remaining = units.drop(choices.values)
            choosers = choosers.drop(choices.index, errors='ignore')

            # Agents choose again.
            next_choices = model.predict(choosers, units_remaining)
            choices = pd.concat([choices, next_choices])
            chosen_multiple_times = identify_duplicate_choices(choices)

    return pd.Series(units.loc[choices.values][model.choice_column].values,
                     index=choices.index)


def register_config_injectable_from_yaml(injectable_name, yaml_file):
    """
    Create orca function for YAML-based config injectables.
    """
    @orca.injectable(injectable_name, cache=True)
    def func():
        with open(os.path.join(misc.configs_dir(), yaml_file)) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            return config
    return func


def register_elcm_model_step(model_name, alt_capacity='vacant_job_spaces', elcm_calibration_config=None):
    @orca.step(model_name)
    def choice_model_simulate(emp_location_choice_models, job_btype_baseyear_prob_matrix):
        model = emp_location_choice_models[model_name]

        # Parse LA and sector info from model name
        model_path = orca.get_injectable('elcm_model_path')
        model_desc_path = os.path.join(model_path, 'model_description.yaml')
        with open(model_desc_path, 'r') as f:
            model_desc = yaml.load(f, Loader=yaml.FullLoader)

        la_id = model_name.split('_')[2][2:]
        home_based = model_name.split('_')[3] == 'homebased'
        job_sector = int(model_name.split('.')[0].split('_')[4][6:])

        filter_text = ''
        for cat_name, categories in model_desc['job_categories'].items():
            for cat in categories:
                if cat in model_name.split('.')[0].split('_'):
                    filter_text += '&(%s_%s==1)' % (cat_name, cat)
                    break

        # === FILTERS ===
        alts_pre_filter = chooser_pre_filter = f"(large_area_id=={la_id})"
        chooser_filter = f"(building_id==-1){filter_text}"
        alt_filter = "(sp_filter>=0)"
        if not home_based:
            alt_filter += f'&(non_residential_sqft>0)&({alt_capacity}>0)'
        else:
            alt_filter += '&(residential_units>0)'

        # === GET DATA ===
        choosers = orca.get_table('jobs')
        alts = orca.get_table('buildings')

        variable_cols = model.variables
        choosers_filter_cols = columns_in_filters(chooser_filter) + columns_in_filters(chooser_pre_filter)
        alts_filter_cols = columns_in_filters(alt_filter) + columns_in_filters(alts_pre_filter)

        choosers_df = choosers.to_frame(choosers_filter_cols)
        choosers_df = choosers_df.query(chooser_pre_filter)
        final_choosers_df = choosers_df.query(chooser_filter)
        n = len(final_choosers_df)
        if n == 0:
            return

        # Compute building age adjustment
        bld_age_var = 'building_age'
        taz_emp_ratio_var = f'taz_empratio_{job_sector}'
        space_col = 'job_spaces'

        _t_alts = time.perf_counter()
        alts_df = alts.to_frame(list(set(
            variable_cols + alts_filter_cols + [alt_capacity, space_col, 'building_type_id', 'stories', bld_age_var, taz_emp_ratio_var]
        )))
        print(f"[alts-timing] ELCM la{la_id} sec{job_sector} "
              f"alts_to_frame={(time.perf_counter() - _t_alts) * 1000:.0f}ms "
              f"rows={len(alts_df)} cols={alts_df.shape[1]}", flush=True)

        # Set bld_age_var to 0 for 10+ story offices
        mask_office_10plus = (alts_df['building_type_id'] == 23) & (alts_df['stories'] >= 10)
        alts_df.loc[mask_office_10plus, bld_age_var] = 0

        alts_idx = alts_df.query(alts_pre_filter).index
        final_alts_df = alts_df.loc[alts_idx].query(alt_filter)

        final_alts_df = final_alts_df[list(set(
            variable_cols + [alt_capacity, space_col, bld_age_var, 'building_type_id', taz_emp_ratio_var]
        ))]

        # Compute vacancy rate
        vacancy_rate = np.where(
            final_alts_df[space_col] == 0,
            0.0,
            final_alts_df[alt_capacity] / final_alts_df[space_col]
        )
        final_alts_df.loc[:, 'vacancy_rate'] = np.clip(vacancy_rate, 0.0, 1.0)

        # Assign vacancy weight
        final_alts_df.loc[:, 'vacancy_weight'] = 1.0
        mask_age_10plus = final_alts_df[bld_age_var] >= 10
        final_alts_df.loc[mask_age_10plus, 'vacancy_weight'] = 1 - final_alts_df.loc[mask_age_10plus, 'vacancy_rate']

        job_btype_matrix = job_btype_baseyear_prob_matrix[job_sector]
        final_alts_df['job_btype_ratio'] = final_alts_df['building_type_id'].map(job_btype_matrix).fillna(0.0)

        if not home_based:
            # OPTIMIZATION: capacity-weighted sampling WITHOUT materializing the
            # full capacity-expanded feature matrix. np.repeat on the index alone
            # is cheap (just an array of building_ids); we sample M slots from it
            # and only then build the M feature rows via .loc. Sampling positions
            # from a length-N Series with random_state=0 picks the same positions
            # as sampling the length-N expanded frame did (sample selects by
            # position, not data), so the chosen M rows match the old
            # full-expansion path — without the ~500 MB build. replace=False
            # keeps each building capped at its capacity (it has that many slots).
            repeated_idx = pd.Series(
                np.repeat(final_alts_df.index.to_numpy(),
                          final_alts_df[alt_capacity].to_numpy().astype(np.int64))
            )
            _n_full = len(repeated_idx)
            M = min(_n_full, n * 20)
            sampled_idx = repeated_idx.sample(M, replace=False, random_state=0).to_numpy()
            predict_X_df = final_alts_df.loc[sampled_idx, variable_cols]
        else:
            predict_X_df = final_alts_df.loc[final_alts_df.index, variable_cols]
            _n_full = len(predict_X_df)
            M = min(_n_full, n * 20)
            predict_X_df = predict_X_df.sample(M, replace=False, random_state=0)

        # clean + scale only the M sampled rows (was previously done on the full
        # expanded matrix before sampling)
        predict_X_df = predict_X_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        _t_sc = time.perf_counter()
        scaler = RobustScaler()
        predict_X_df = pd.DataFrame(
            scaler.fit_transform(predict_X_df),
            columns=predict_X_df.columns,
            index=predict_X_df.index
        )
        predict_X_df = np.clip(predict_X_df, -5, 5)
        _t_fw = time.perf_counter()
        pred = model.predict(predict_X_df).detach().cpu().numpy().flatten()
        print(f"[lcm-timing] ELCM la{la_id} sec{job_sector} full={_n_full} M={M} "
              f"scale={(_t_fw - _t_sc) * 1000:.0f}ms fwd={(time.perf_counter() - _t_fw) * 1000:.0f}ms",
              flush=True)

        # === CALIBRATION ===
        calibration = elcm_calibration_config
        method = calibration.get("calibration_method", "multiplicative")
        weights = calibration.get("weights", {})
        use_taz_cluster = calibration.get("use_taz_cluster_by_sector", {}).get(job_sector, False)
        override_key = f"la_{la_id}_sector_{job_sector}"
        if override_key in calibration.get("overrides", {}):
            method = calibration["overrides"][override_key].get("calibration_method", method)
            weights = calibration["overrides"][override_key].get("weights", weights)

        # Default weights fallback
        weight_building_age = weights.get("building_age", 1.0)
        weight_vacancy = weights.get("vacancy", 1.0)
        weight_btype = weights.get("btype_matrix", 1.0)
        weight_taz = weights.get("taz_cluster", 1.0)

        # Apply individual weight components
        building_age_weights = final_alts_df.loc[predict_X_df.index, bld_age_var]
        building_age_weights = np.digitize(building_age_weights, [16, 31])
        building_age_weights = np.array([5.0, 3.0, 1.0])[building_age_weights]

        vacancy_weights = final_alts_df.loc[predict_X_df.index, 'vacancy_weight'].to_numpy()
        job_btype_arr = final_alts_df.loc[predict_X_df.index, 'job_btype_ratio'].to_numpy()

        if use_taz_cluster:
            taz_arr = final_alts_df.loc[predict_X_df.index, taz_emp_ratio_var].to_numpy()
            min_val, max_val = taz_arr.min(), taz_arr.max()
            taz_cluster_arr = np.ones_like(taz_arr) if min_val == max_val else (taz_arr - min_val) / (max_val - min_val)
        else:
            taz_cluster_arr = np.ones(len(predict_X_df))

        # Final calibration
        if method == 'log_weighted':
            total = weight_building_age + weight_vacancy + weight_btype + weight_taz
            pred_weighted = pred * np.exp(
                (weight_building_age / total) * np.log(building_age_weights + 1e-6) +
                (weight_vacancy / total) * np.log(vacancy_weights + 1e-6) +
                (weight_btype / total) * np.log(job_btype_arr + 1e-6) +
                (weight_taz / total) * np.log(taz_cluster_arr + 1e-6)
            )
        else:
            # use regular multiplication
            pred_weighted = pred * building_age_weights * vacancy_weights * job_btype_arr * taz_cluster_arr

        picked_idx = np.argsort(pred_weighted)[-n:]
        picked_bid = predict_X_df.iloc[picked_idx].index

        # Assign buildings
        choosers_df.loc[final_choosers_df.index, 'building_id'] = picked_bid.values
        orca.get_table('jobs').update_col_from_series('building_id', choosers_df.loc[final_choosers_df.index, 'building_id'], cast=True)

        # Update capacity if applicable
        if alt_capacity in alts.local_columns:
            picked_hu = picked_bid.value_counts()
            new_capacity = alts_df.loc[picked_hu.index][alt_capacity] - picked_hu
            if (new_capacity < 0).any():
                raise ValueError("Negative capacity detected.")
            orca.get_table('buildings').update_col_from_series(alt_capacity, new_capacity, cast=True)

        print(f"Placed {len(picked_bid)} jobs.")

    return choice_model_simulate

def register_hlcm_model_step(model_name, alt_capacity='residential_units'):

    # TODO: Update simulate steps with lcm nn model
    @orca.step(model_name)
    def choice_model_simulate(hh_location_choice_models):
        model = hh_location_choice_models[model_name]

        model_path = orca.get_injectable('hlcm_model_path')
        model_desc_path = os.path.join(model_path, 'model_description.yaml')
        with open(model_desc_path, 'r') as f:
            model_desc = yaml.load(f, Loader=yaml.FullLoader)

        # chooser segment
        la_id = model_name.split('_')[2][2:]
        filter_text = ''
        tract_segment_type_var = 'tract_hh_type_ratio'
        for cat_name, categories in model_desc['hh_categories'].items():
            for cat in categories:
                if cat in model_name:
                    filter_text += '&(%s_%s==1)' % (cat_name, cat)
                    tract_segment_type_var += '_%s_%s' % (cat_name, cat)
                    break

        # hh_size = model_name.split('_')[3]
        # ownership = model_name.split('_')[4]
        # aoh = model_name.split('_')[5].split('.')[0]
        
        # pre filter
        alts_pre_filter = chooser_pre_filter = "(large_area_id==%s)" % (la_id)

        # filter for picking hh with no building_id assigned
        # chooser_filter = "(building_id==-1)&(hh_size_%s==1)&(ownership_%s==1)&(aoh_%s==1)" % (hh_size, ownership, aoh)
        chooser_filter = "(building_id==-1)" + filter_text

        # filter alternatives
        alt_filter = "(residential_units>0) & (%s>0) & (hu_filter==0) & (sp_filter>=0)" % (alt_capacity)
            
        # load variables from model
        variable_cols = model.variables

        # filter for choosers and alternatives
        choosers_filter_cols = columns_in_filters(chooser_filter) + columns_in_filters(chooser_pre_filter)
        alts_filter_cols = columns_in_filters(alt_filter) + columns_in_filters(alts_pre_filter)

        # choosers
        choosers = orca.get_table('households')
        choosers_df = choosers.to_frame(choosers_filter_cols)
        # query using chooser_pre_filter to match whats used in estimation
        choosers_df = choosers_df.query(chooser_pre_filter)
        # std choosers columns
        # filter using chooser_filter
        final_choosers_df = choosers_df.query(chooser_filter)
        n = len(final_choosers_df)

        # return if not needed
        if n == 0:
            return

        # alternatives
        alts = orca.get_table('buildings')

        # all variables should ben available
        assert all([True if col in alts.columns else False for col in variable_cols])

        formula_alts_col = list(set(variable_cols))
        _t_alts = time.perf_counter()
        alts_df = alts.to_frame(list(set(formula_alts_col+alts_filter_cols+[alt_capacity, tract_segment_type_var])))
        print(f"[alts-timing] HLCM la{la_id} "
              f"alts_to_frame={(time.perf_counter() - _t_alts) * 1000:.0f}ms "
              f"rows={len(alts_df)} cols={alts_df.shape[1]}", flush=True)

        # query using alts_pre_filter to match whats used in estimation
        alts_idx = alts_df.query(alts_pre_filter).index
        
        # alts_col_df alts columns
        std_cols = [col for col in formula_alts_col if col != alt_capacity]
        alts_col_df = alts_df.loc[alts_idx, std_cols]

        # derive scaler from model_name, default to 'std' if no 8th segment
        name_parts = model_name.split('_')
        try:
            scaler = name_parts[7]
            # strip trailing ".pt" if present
            if scaler.endswith('.pt'):
                scaler = scaler[:-3]
        except IndexError:
            scaler = 'std'

        # retrive scaler trained during estimation for consistency
        scaler_state = getattr(model, 'scaler_state', None)
        if scaler_state is not None:
            alts_col_df = apply_scaler_state(alts_col_df, scaler_state)
        elif scaler == 'std':
            alts_col_df = std_scaler_transform(alts_col_df)
        elif scaler == 'robust':
            alts_col_df = robust_scaler_transform(alts_col_df)
        elif scaler == 'minmax':
            alts_col_df = min_max_scaler_transform(alts_col_df)
        else:
            # std by default
            alts_col_df = std_scaler_transform(alts_col_df)

        # fill them back to alts_df (ensure float so pandas 3 accepts scaled values)
        for _c in std_cols:
            if alts_df[_c].dtype != float:
                alts_df[_c] = alts_df[_c].astype(float)
        alts_df.loc[alts_idx, std_cols] = alts_col_df

        # filter using alt_filter
        final_alts_df = alts_df.loc[alts_idx].query(alt_filter)

        # construct predict DF with capacity and get result
        final_alts_df = final_alts_df[list(set(formula_alts_col + [alt_capacity, tract_segment_type_var]))]
        predict_X_df = final_alts_df.loc[
            np.repeat(final_alts_df.index, final_alts_df[alt_capacity]),
            formula_alts_col
        ]

        # std alt_capacity variable if it's in formula_alts_col
        if alt_capacity in formula_alts_col:
            # predict_X_df[alt_capacity] = scaler.fit_transform(predict_X_df[alt_capacity].to_numpy().reshape(-1,1))
            alt_capacity_arr = predict_X_df[alt_capacity].to_numpy().reshape(-1,1)
            if scaler == 'std':
                predict_X_df[alt_capacity] = std_scaler_transform(alt_capacity_arr)
            elif scaler == 'robust':
                predict_X_df[alt_capacity] = robust_scaler_transform(alt_capacity_arr)
            elif scaler == 'minmax':
                predict_X_df[alt_capacity] = min_max_scaler_transform(alt_capacity_arr)
            else:
                # std by default
                predict_X_df[alt_capacity] = std_scaler_transform(alt_capacity_arr)

        # clip transform after scaling
        predict_X_df = np.clip(predict_X_df.fillna(0.0), -5, 5)

        # sample predict_X_df to 1:5 preventing hlcm segment order issue
        M = min(len(predict_X_df), n * 5) # HU pool count
        predict_X_df = predict_X_df.sample(M, replace=False, random_state=0)

        # run predict
        pred = model.predict(predict_X_df).detach().cpu().numpy().flatten()

        # === CALIBRATION ===
        USE_TRACT_CALIBRATOR_MODEL = True
        if USE_TRACT_CALIBRATOR_MODEL:
            # Build training targets from observed & base ratios
            base_ratios_df = orca.get_table("tract_hh_type_base_ratios").to_frame()
            current_ratios = final_alts_df.loc[predict_X_df.index, tract_segment_type_var].to_numpy()

            bld_to_tract = alts.to_frame(['tract_id'])['tract_id']
            tract_ids = bld_to_tract.reindex(predict_X_df.index).to_numpy()
            base_ratios = base_ratios_df[tract_segment_type_var].reindex(tract_ids).to_numpy()

            # Compute y_train = base / current (avoid divide-by-zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                y_train = np.divide(base_ratios, current_ratios)
                y_train = np.nan_to_num(y_train, nan=1.0, posinf=2.0, neginf=0.5)

            # Load and prepare Census Tracts features
            # TODO: load current all Tracts available in alt_df
            tracts_df = orca.get_table('census_tracts').to_frame()
            # Handle missing tract_id = -1
            if -1 not in tracts_df.index:
                tracts_df.loc[-1] = tracts_df.mean(numeric_only=True)

            # Get Tracts features for buildings used
            used_tract_ids = pd.Series(tract_ids).dropna().unique()
            # Filter to Tracts used and select only numeric float columns
            valid_columns = tracts_df.select_dtypes(include=['float', 'float32', 'float64']).columns
            # some exclusions
            valid_columns = [col for col in valid_columns if col not in ['tazce10_n']]

            X_train = tracts_df.loc[tracts_df.index.intersection(used_tract_ids), valid_columns].fillna(0.0)
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.select_dtypes(include='number').columns
            )
            # Feature Selection, remove low variance columns
            low_variance_cols = X_train_scaled.var()[X_train_scaled.var() < 1e-5].index
            X_train_scaled.drop(columns=low_variance_cols, inplace=True)

            # Aggregate y_train to Tract level
            tract_weights = pd.Series(y_train, index=tract_ids).groupby(tract_ids).mean()
            y_tract = tract_weights.reindex(X_train_scaled.index).fillna(1.0)

            # === Pre-train baseline error (if using a naive model like predicting mean) ===
            naive_pred = np.full_like(y_tract.values, fill_value=y_tract.mean())
            mse_naive = mean_squared_error(y_tract.values, naive_pred)
            mae_naive = mean_absolute_error(y_tract.values, naive_pred)
            print(f"[Calibrator] Pre-train baseline error: MSE={mse_naive:.4f}, MAE={mae_naive:.4f}")

            # Train calibrator model (choose one)
            # XGBoost
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,      # Slow learning helps reduce overfitting
                max_depth=3,             # Prevents over-complex trees
                subsample=0.7,           # Row subsampling
                colsample_bytree=0.7,    # Feature subsampling
                reg_alpha=1.0,           # L1 regularization (sparse solutions)
                reg_lambda=10.0,         # Stronger L2 regularization
                min_child_weight=10,     # Avoid splits on small samples
                random_state=42,
                verbosity=1,
                tree_method='hist',
                device = "cuda"
            )
            model.fit(X_train_scaled.values, y_tract.values)

            # === Post-train prediction error ===
            y_pred = model.predict(X_train_scaled.values)
            mse_model = mean_squared_error(y_tract.values, y_pred)
            mae_model = mean_absolute_error(y_tract.values, y_pred)
            print(f"[Calibrator] Post-train error:        MSE={mse_model:.4f}, MAE={mae_model:.4f}")

            # Predict adjustment weights
            tract_predicted_weights = pd.Series(y_pred, index=X_train_scaled.index)
            tract_predicted_weights = tract_predicted_weights.clip(lower=0.5, upper=2.0)

            # Map to HU-level rows in predict_X_df
            tract_segment_adj_arr = tract_predicted_weights.reindex(tract_ids).fillna(1.0).to_numpy()

        else:
            tract_segment_adj_arr = np.ones(len(predict_X_df))

        # Apply individual weight components
        # default to multiplicative calibration
        pred_weighted = pred * tract_segment_adj_arr 
        
        picked_idx = np.argsort(pred_weighted)[-n:]
        picked_bid = predict_X_df.iloc[picked_idx].index

        # update building_id
        choosers_df.loc[final_choosers_df.index, 'building_id'] = picked_bid.values.astype(
            choosers_df['building_id'].dtype)

        print("Placed %s households." % len(picked_bid))

        # update households table
        orca.get_table('households').update_col_from_series(
            'building_id', choosers_df.loc[final_choosers_df.index, 'building_id'], cast=True)

        # if alt_capacity exists in local_columns, updates it
        if alt_capacity in alts.local_columns:
            # Update alts table to reduce remaining capacity
            picked_hu = picked_bid.value_counts()
            new_capacity = alts_df.loc[picked_hu.index][alt_capacity] - picked_hu
            if (new_capacity < 0).any():
                raise ValueError("Encounter negative value while calculating new building capacity")
            orca.get_table('buildings').update_col_from_series(alt_capacity, new_capacity, cast=True)

    return choice_model_simulate


class SimulationChoiceModel(MNLDiscreteChoiceModel):
    """
    A discrete choice model with parameters needed for simulation.
    Initialize with MNLDiscreteChoiceModel's init parameters or with from_yaml,
    then add simulation parameters with set_simulation_params().

    """
    def set_simulation_params(self, name, supply_variable, vacant_variable,
                              choosers, alternatives, choice_column=None,
                              summary_alts_xref=None, merge_tables=None,
                              agent_units=None):
        """
        Add simulation parameters as additional attributes.
        Parameters
        ----------
        name : str
            Name of the model.
        supply_variable : str
            The name of the column in the alternatives table indicating number
            of available spaces, vacant or not, that can be occupied by
            choosers.
        vacant_variable : str
            The name of the column in the alternatives table indicating number
            of vacant spaces that can be occupied by choosers.
        choosers : str
            Name of the choosers table.
        alternatives : str
            Name of the alternatives table.
        summary_alts_xref : dict or pd.Series, optional
            Mapping of alternative index to summary alternative id.  For use
            in evaluating a model with many alternatives.
        merge_tables : list of str, optional
            List of additional tables to be broadcast onto the alternatives
            table.
        agent_units : str, optional
            Name of the column in the choosers table that designates how
            much supply is occupied by each chooser.
        Returns
        -------
        None
        """
        self.name = name
        self.supply_variable = supply_variable
        self.vacant_variable = vacant_variable
        self.choosers = choosers
        self.alternatives = alternatives
        self.summary_alts_xref = summary_alts_xref
        self.merge_tables = merge_tables
        self.agent_units = agent_units
        self.choice_column = choice_column if choice_column is not None \
            else self.choice_column

    def simulate(self, choice_function=None, save_probabilities=False,
                 **kwargs):
        """
        Computing choices, with arbitrary function for handling simulation
        strategy.
        Parameters
        ----------
        choice_function : function
            Function defining how to simulate choices based on fitted model.
            Function must accept the following 3 arguments:  model object,
            choosers DataFrame, and alternatives DataFrame. Additional optional
            keyword args can be utilized by function if needed (kwargs).
        save_probabilities : bool
            If true, will save the calculated probabilities underlying the
            simulation as an orca injectable with name
            'probabilities_modelname_itervar'.
        Returns
        -------
        choices : pandas.Series
            Mapping of chooser ID to alternative ID. Some choosers
            will map to a nan value when there are not enough alternatives
            for all the choosers.
        """
        choosers, alternatives = self.calculate_model_variables()

        choosers, alternatives = self.apply_predict_filters(
                                 choosers, alternatives)

        # By convention, choosers are denoted by a -1 value
        # in the choice column
        choosers = choosers[choosers[self.choice_column] == -1]
        print("{} agents are making a choice.".format(len(choosers)))

        if choice_function:
            choices = choice_function(self, choosers, alternatives, **kwargs)
        else:
            choices = self.predict(choosers, alternatives, debug=True)

        if save_probabilities:
            if not self.sim_pdf:
                probabilities = self.calculate_probabilities(choosers,
                                                             alternatives)
            else:
                probabilities = self.sim_pdf.reset_index().set_index(
                    'alternative_id')[0]
            orca.add_injectable('probabilities_{}_{}'.format(
                self.name, orca.get_injectable('iter_var')),
                probabilities)

        return choices

    def fit_model(self):
        """
        Estimate model based on existing parameters
        Returns
        -------
        None
        """
        choosers, alternatives = self.calculate_model_variables()
        self.fit(choosers, alternatives, choosers[self.choice_column])
        return self.log_likelihoods, self.fit_parameters

    def calculate_probabilities(self, choosers, alternatives):
        """
        Calculate model probabilities.
        Parameters
        ----------
        choosers : pandas.DataFrame
            DataFrame of choosers.
        alternatives : pandas.DataFrame
            DataFrame of alternatives.
        Returns
        -------
        probabilities : pandas.Series
            Mapping of alternative ID to probabilities.
        """
        probabilities = self.probabilities(choosers, alternatives)
        probabilities = probabilities.reset_index().set_index(
            'alternative_id')[0]  # remove chooser_id col from idx
        return probabilities

    def calculate_model_variables(self):
        """
        Calculate variables needed to simulate the model, and returns
        DataFrames of simulation-ready tables with needed variables.
        Returns
        -------
        choosers : pandas.DataFrame
            DataFrame of choosers.
        alternatives : pandas.DataFrame
            DataFrame of alternatives.
        """
        columns_used = self.columns_used() + [self.choice_column]
        columns_used = columns_used + [self.agent_units] if self.agent_units else columns_used
        choosers = orca.get_table(self.choosers).to_frame(columns_used)

        supply_column_names = [col for col in
                               [self.supply_variable, self.vacant_variable]
                               if col is not None]

        columns_used.extend(supply_column_names)

        if self.merge_tables:
            mt = copy.deepcopy(self.merge_tables)
            mt.append(self.alternatives)
            all_cols = []
            for table in mt:
                all_cols.extend(orca.get_table(table).columns)
            all_cols = [col for col in all_cols if col in columns_used]
            alternatives = orca.merge_tables(target=self.alternatives,
                                             tables=mt, columns=all_cols)
        else:
            alternatives = orca.get_table(self.alternatives).to_frame(
                columns_used + supply_column_names)
        return choosers, alternatives

    def score(self, scoring_function=accuracy_score, choosers=None,
              alternatives=None, aggregate=False, apply_filter=True,
              choice_function=random_choices):
        """
        Calculate score for model.  Defaults to accuracy score, but other
        scoring functions can be provided.  Computed on all choosers/
        alternatives by default, but can also be computed on user-supplied
        test datasets.  If model has a summary_alts_xref, then score
        calculated after mapping to summary ids.
        Parameters
        ----------
        scoring_function : function, default sklearn.metrics.accuracy_score
            Function defining how to score model predictions. Function must
            accept the following 2 arguments:  pd.Series of observed choices,
            pd.Series of predicted choices.
        choosers : pandas.DataFrame, optional
            DataFrame of choosers.
        alternatives : pandas.DataFrame, optional
            DataFrame of alternatives.
        aggregate : bool
            Whether to calculate score based on total count of choosers that
            made each choice, rather than based on disaggregate choices.
        apply_filter : bool
            Whether to apply the model's choosers_predict_filters prior to
            calculating score.  If supplying own test dataset, and do not want
            it further manipulated, then set to False.
        choice_function : function, option
            Function defining how to simulate choices.
        Returns
        -------
        score : float
            The model's score (accuracy score by default).
        """
        if choosers is None or alternatives is None:
            choosers, alternatives = self.calculate_model_variables()

        if apply_filter:
            if self.choosers_predict_filters:
                choosers = choosers.query(self.choosers_predict_filters)
            if self.choosers_fit_filters:
                choosers = choosers.query(self.choosers_fit_filters)

        observed_choices = choosers[self.choice_column]
        predicted_choices = choice_function(self, choosers, alternatives)

        if self.summary_alts_xref is not None:
            observed_choices = observed_choices.map(self.summary_alts_xref)
            predicted_choices = predicted_choices.map(self.summary_alts_xref)

        if aggregate:
            observed_choices = observed_choices.value_counts()
            predicted_choices = predicted_choices.value_counts()

            combined_index = list(set(list(predicted_choices.index) +
                                      list(observed_choices.index)))
            predicted_choices = predicted_choices.reindex(combined_index).fillna(0)
            observed_choices = observed_choices.reindex(combined_index).fillna(0)

        return scoring_function(observed_choices, predicted_choices)

    def summed_probabilities(self, choosers=None, alternatives=None):
        """
        Sum probabilities to the summary geography level.
        """
        if choosers is None or alternatives is None:
            choosers, alternatives = self.calculate_model_variables()

        if self.choosers_fit_filters:
            choosers = choosers.query(self.choosers_fit_filters)

        if self.choosers_predict_filters:
            choosers = choosers.query(self.choosers_predict_filters)

        choosers['summary_id'] = choosers[self.choice_column]
        choosers.summary_id = choosers.summary_id.map(self.summary_alts_xref)
        probs = self.calculate_probabilities(choosers, alternatives)
        probs = probs.reset_index().rename(columns={0: 'proba'})
        probs['summary_id'] = probs.alternative_id.map(self.summary_alts_xref)
        return probs.groupby('summary_id').proba.sum()

    def observed_distribution(self, choosers=None):
        """
        Calculate observed distribution across alternatives at the summary
        geography level.
        """
        if choosers is None:
            choosers, alternatives = self.calculate_model_variables()

        if self.choosers_fit_filters:
            choosers = choosers.query(self.choosers_fit_filters)

        if self.choosers_predict_filters:
            choosers = choosers.query(self.choosers_predict_filters)

        if 'summary_id' not in choosers.columns:
            summ_id = choosers[self.choice_column].map(self.summary_alts_xref)
            choosers['summary_id'] = summ_id

        observed_distrib = choosers.groupby('summary_id').size()
        return observed_distrib / observed_distrib.sum()

    def summed_probability_score(self, scoring_function=r2_score,
                                 choosers=None, alternatives=None,
                                 validation_data=None):
        if choosers is None or alternatives is None:
            choosers, alternatives = self.calculate_model_variables()

        if self.choosers_fit_filters:
            choosers = choosers.query(self.choosers_fit_filters)

        if self.choosers_predict_filters:
            choosers = choosers.query(self.choosers_predict_filters)

        summed_probas = self.summed_probabilities(choosers, alternatives)

        if validation_data is None:
            validation_data = self.observed_distribution(choosers)

        combined_index = list(set(list(summed_probas.index) +
                                  list(validation_data.index)))
        summed_probas = summed_probas.reindex(combined_index).fillna(0)
        validation_data = validation_data.reindex(combined_index).fillna(0)

        print(summed_probas.corr(validation_data))
        score = scoring_function(validation_data, summed_probas)
        print(score)

        residuals = summed_probas - validation_data
        return score, residuals


def get_model_category_configs(yaml_configs):
    """
    Returns dictionary where key is model category name and value is dictionary
    of model category attributes, including individual model config filename(s)
    """
    # TODO: update yaml_configs_2050.yaml
    with open(os.path.join(misc.configs_dir(), yaml_configs)) as f:
        yaml_configs = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(misc.configs_dir(), 'model_structure.yaml')) as f:
        model_category_configs = yaml.load(f, Loader=yaml.FullLoader)['models']

    for model_category, category_attributes in list(model_category_configs.items()):
        category_attributes['config_filenames'] = yaml_configs[model_category]

    return model_category_configs


def load_hlcm_model_configs_from_path(path, yaml_configs):
    # load all available model files from path and dump into yaml_configs
    nn_models = os.listdir(os.path.join(path, 'pts'))
    # with open(os.path.join(misc.configs_dir(), yaml_configs), 'r+') as f:
    with open(os.path.join(misc.configs_dir(), yaml_configs), 'r') as f:
        ym = yaml.safe_load(f)
        ym['hlcm'] = nn_models
    with open(os.path.join(misc.configs_dir(), yaml_configs), 'w') as f:
        yaml.dump(ym, f)

def load_elcm_model_configs_from_path(path, yaml_configs):
    # load all available model files from path and dump into yaml_configs
    nn_models = os.listdir(os.path.join(path, 'pts'))
    # with open(os.path.join(misc.configs_dir(), yaml_configs), 'r+') as f:
    with open(os.path.join(misc.configs_dir(), yaml_configs), 'r') as f:
        ym = yaml.safe_load(f)
        ym['elcm'] = nn_models
    with open(os.path.join(misc.configs_dir(), yaml_configs), 'w') as f:
        yaml.dump(ym, f)


def create_lcm_from_config(config_filename, model_attributes):
    """
    For a given model config filename and dictionary of model category
    attributes, instantiate a LargeMultinomialLogitStep object.

    config_filename: model name
    model_attributes: model_structure.yaml
    """
    with open(misc.config(config_filename), "r") as f:
        config_obj = yaml.load(f, Loader=yaml.FullLoader)

    model = LargeMultinomialLogitStep.from_dict(config_obj['saved_object'])
    model.choosers = model_attributes['agents_name']
    model.alternatives = model_attributes['alternatives_name']
    model.choice_column = model_attributes['alternatives_id_name']
    # is it alt_capacity in largeMNL equals vacant_variable in 2045?
    model.alt_capacity = model_attributes['vacant_variable']
    return model


# TODO: create new create_lcm function for lcm_nn
def load_torch_lcm(config_filename, model_attributes):
    """
    For a given model filename and dictionary of model category
    attributes, instantiate a torch model object.

    config_filename: model filename
    model_attributes: model_structure.yaml
    """
    device = torch.device('cpu')
    state = torch.load(config_filename, map_location=device)

    lcm = LCM_NN( 
        state['input_size'], 
        hidden_layer=state['hidden_layer'], 
        lr=state['lr'], 
        weight_decay=state['weight_decay']
    )

    lcm.load_model(config_filename)

    return lcm


def get_hlcm_valid_vars(data_path: str) -> tuple[list[str], list[str]]:
    """
    Extract valid household and building variable names from a YAML configuration file.

    Parameters:
    data_path (str): Path to the directory containing the variable validation YAML files.

    Returns:
    tuple: A tuple containing two lists of valid variable names: valid household variable names and valid building variable names.
    """
    var_validation_list = [
        os.path.join(data_path, f)
        for f in os.listdir(data_path)
        if ("variable_validation" in f) and (f[-5:] == ".yaml")
    ]
    var_validation_last = max(var_validation_list, key=os.path.getctime)
    
    with open(var_validation_last, "r") as f:
        vars_config = yaml.load(f, Loader=yaml.FullLoader)
    
    valid_b_vars = vars_config["buildings"]["valid variables"]
    valid_hh_vars = vars_config["households"]["valid variables"]
    return valid_hh_vars, valid_b_vars

def get_elcm_valid_vars(data_path: str) -> tuple[list[str], list[str]]:
    """
    Extract valid household and building variable names from a YAML configuration file.

    Parameters:
    data_path (str): Path to the directory containing the variable validation YAML files.

    Returns:
    tuple: A tuple containing two lists of valid variable names: valid household variable names and valid building variable names.
    """
    var_validation_list = [
        (data_path + "/" + f)
        for f in os.listdir(data_path)
        if ("variable_validation" in f) & (f[-5:] == ".yaml")
    ]
    var_validation_last = max(var_validation_list, key=os.path.getctime)
    with open(var_validation_last, "r") as f:
        vars_config = yaml.load(f, Loader=yaml.FullLoader)
    valid_job_vars = vars_config["jobs"]["valid variables"]
    valid_b_vars = vars_config["buildings"]["valid variables"]
    return valid_job_vars, valid_b_vars

def load_hlcm_df(households: orca.DataFrameWrapper, buildings: orca.DataFrameWrapper, hh_var: list[str], b_var: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and return household and building DataFrames.

    Parameters:
    households (orca.DataFrameWrapper): Orca DataFrameWrapper for households.
    buildings (orca.DataFrameWrapper): Orca DataFrameWrapper for buildings.
    hh_var (list[str]): Names of the household variables to load.
    b_var (list[str]): Names of the building variables to load.

    Returns:
    tuple: A tuple containing two DataFrames: the household DataFrame and the building DataFrame.

    Example:
    >>> household_df, building_df = load_hlcm_df(households, buildings, ['persons'], ['parcel_id'])
    """
    hh = households.to_frame(hh_var)
    b = buildings.to_frame(b_var)
    return hh, b

def load_elcm_df(jobs: orca.DataFrameWrapper, buildings: orca.DataFrameWrapper, job_var: list[str], b_var: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and return job and building DataFrames.

    Parameters:
    jobs (orca.DataFrameWrapper): Jobs data DataFrame.
    buildings (orca.DataFrameWrapper): Building data DataFrame.
    job_var (list[str]): Names of the job variables to load.
    b_var (list[str]): Names of the building variables to load.

    Returns:
    tuple: A tuple containing two DataFrames: the job DataFrame and the building DataFrame.

    Example:
    >>> job_df, building_df = load_elcm_df(['employment'], ['parcel_id'])
    """
    job_df = jobs.to_frame(job_var)
    b_df = buildings.to_frame(b_var)
    return job_df, b_df

def columns_in_vars(vars: list[str], valid_agent_vars: list[str], valid_b_vars: list[str]) -> tuple[list[str], list[str]]:
    """
    Categorize variables into agents and building columns.

    This function takes a list of variable names and categorizes them into
    agents columns and building columns based on the presence of a colon
    separator or matching valid variable names.

    Parameters:
    vars (list[str]): List of variable names to categorize.
    valid_agent_vars (list[str]): List of valid agents variable names
    valid_b_vars (list[str]): List of valid building variable names

    Returns:
    tuple: A tuple containing two lists of strings: agents column names and building column names.
    """
    agent_columns, b_columns = [], []
    for varname in vars:
        if ':' in varname:
            agent_col, b_col = map(str.strip, varname.split(':'))
            if agent_col in valid_agent_vars:
                agent_columns.append(agent_col)
            if b_col in valid_b_vars:
                b_columns.append(b_col)
        elif varname in valid_agent_vars:
            agent_columns.append(varname)
        elif varname in valid_b_vars:
            b_columns.append(varname)
        else:
            print(varname, " not found in both agents and buildings table")
    return agent_columns, b_columns

def get_interaction_vars(df: pd.DataFrame, varname: str) -> np.ndarray:
    """
    Get interaction variables from variable name.

    This function calculates interaction variables based on the provided variable name
    within the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the variables.
        varname (str): The name of the interaction variable.

    Returns:
        np.ndarray: A NumPy array containing the calculated interaction variables.

    Example:
    >>> data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    >>> df = pd.DataFrame(data)
    >>> interaction_array = get_interaction_vars(df, 'A:B')
    """
    if ":" in varname:
        var1, var2 = map(str.strip, varname.split(":"))
        return (df[var1] * df[var2]).values.reshape(-1, 1)
    else:
        return df[varname].values.reshape(-1, 1)

def load_hlcm_dataset(valid_hh_vars, valid_b_vars, var_pool_table_path, hh_filter_columns, b_filter_columns, use_cache=False):
    """
    Load and preprocess dataset variables

    This function loads and preprocesses the dataset variables needed for estimation. It extracts the set of valid variables
    from the variable pool table, loads the necessary variables from the 'buildings' and 'households' Orca tables,
    and caches the resulting household and building DataFrames for later use.

    Parameters:
    valid_hh_vars (list): A list of valid household variable names.
    valid_b_vars (list): A list of valid building variable names.

    Returns:
    tuple: A tuple containing the following elements:
        - hh_region (pd.DataFrame): A DataFrame containing loaded household data.
        - b_region (pd.DataFrame): A DataFrame containing loaded building data.
        - vars_to_use (np.ndarray): An array of variable names used for modeling.
    """
    # Load the variable pool table and extract valid variable names
    used_vars = pd.read_excel(var_pool_table_path, sheet_name=2)
    v1 = used_vars[~used_vars["new variables 1"].isna()]["new variables 1"].unique()
    v2 = used_vars[~used_vars["new variables 2"].isna()]["new variables 2"].unique()
    vars_to_use = np.array(list(set(v1.tolist()).union(v2.tolist())))

    # Choose whether to reload data or use cached data
    if not use_cache:
        # from notebooks.models_test import *
        import models
        buildings = orca.get_table("buildings")
        households = orca.get_table("households")

        # set year to 2020 and run build network and neigh vars
        orca.add_injectable('year', 2020)
        orca.run(["build_networks_2050"])
        orca.run(["neighborhood_vars"])

        # set year to 2050 and run mcd_hu_sampling
        orca.add_injectable('year', 2050)
        orca.run(["mcd_hu_sampling"])

        # Get valid variables for modeling and load corresponding data
        hh_columns, b_columns = columns_in_vars(vars_to_use, valid_hh_vars, valid_b_vars)
        hh_var = hh_columns + hh_filter_columns
        b_var = b_columns + b_filter_columns
        hh_region, b_region = load_hlcm_df(households, buildings, hh_var, b_var)

        # Cache the loaded DataFrames as CSV files
        hh_region.to_csv('data/hh.csv')
        b_region.to_csv('data/b_hlcm.csv')
    else:
        hh_region, b_region = load_cache_hh_b('data/hh.csv', 'data/b_hlcm.csv')
    return hh_region, b_region, vars_to_use

def load_elcm_dataset(valid_job_vars, valid_b_vars, var_pool_table_path, job_filter_columns, b_filter_columns, use_cache=False):
    """
    Load and preprocess job and building datasets for ELCM estimation.

    This function loads the job and building datasets, extracts valid variable names, and preprocesses the data
    for estimation using the ELCM (Employment Location Choice Model).

    Parameters:
    valid_job_vars (list[str]): Valid job variables for modeling.
    valid_b_vars (list[str]): Valid building variables for modeling.
    var_pool_table_path (str): Path to the variable pool table Excel file.
    job_filter_columns (list[str]): Job filter columns to exclude from the loaded data.
    b_filter_columns (list[str]): Building filter columns to exclude from the loaded data.
    use_cache (Boolean): Use cache(True) or reload(False)

    Returns:
    tuple: A tuple containing job_region DataFrame, building_region DataFrame, and vars_to_use array.

    Example:
    >>> job_region, building_region, vars_to_use = load_elcm_dataset(valid_job_vars, valid_b_vars,
    ...                                                              var_pool_table_path, job_filter_columns,
    ...                                                              b_filter_columns)
    """
    # Load the variable pool table and extract valid variable names
    used_vars = pd.read_excel(var_pool_table_path, sheet_name=1)
    v1 = used_vars[~used_vars["variables 1"].isna()]["variables 1"].unique()
    v2 = used_vars[~used_vars["Variables 2"].isna()]["Variables 2"].unique()
    vars_to_use = np.array(list(set(v1.tolist()).union(v2.tolist())))

    # Choose whether to reload data or use cached data
    if not use_cache:
        # from notebooks.models_test import *
        import models
        buildings = orca.get_table("buildings")
        jobs = orca.get_table("jobs")

        # set year to 2020 and run build network and neigh vars
        orca.add_injectable('year', 2020)
        orca.run(["build_networks_2050"])
        orca.run(["neighborhood_vars"])

        # set year to 2050 and run mcd_hu_sampling
        orca.add_injectable('year', 2050)
        orca.run(["mcd_hu_sampling"])

        # Get valid variables for modeling and load corresponding data
        job_columns, b_columns = columns_in_vars(vars_to_use, valid_job_vars, valid_b_vars)

        job_var = job_columns + job_filter_columns
        b_var = b_columns + b_filter_columns
        job_region, b_region = load_elcm_df(jobs, buildings, job_var, b_var)

        # Cache the loaded DataFrames as CSV files
        job_region.to_csv('data/jobs.csv')
        b_region.to_csv('data/b_elcm.csv')
    else:
        job_region, b_region = load_cache_hh_b('data/jobs.csv', 'data/b_elcm.csv')
    return job_region, b_region, vars_to_use

def load_cache_hh_b(hh_csv_path: str, b_csv_path: str):
    """
    Load household and building data from CSV files and register them as tables.

    Parameters:
    hh_csv_path (str): Path to the household CSV file.
    b_csv_path (str): Path to the building CSV file.
    """
    try:
        hh_region = pd.read_csv(hh_csv_path, index_col=0)
        b_region = pd.read_csv(b_csv_path, index_col=0)
    except FileNotFoundError:
        print("CSV file not found. Please provide correct file paths.")
        return 
    
    orca.add_table('households', hh_region)
    orca.add_table('buildings', b_region)
    return hh_region, b_region

def get_hlcm_segment():
    """Generate hlcm segments variables based on model description hlcm_model_path.
    orca.get_injectable('hlcm_model_path')

    Returns:
        [(hh_cat1, hh_cat_2, hh_cat_3), ...]: products of all household categories
    """
    model_path = orca.get_injectable('hlcm_model_path')
    model_desc_path = os.path.join(model_path, 'model_description.yaml')
    with open(model_desc_path, 'r') as f:
        model_desc = yaml.load(f, Loader=yaml.FullLoader)

    # make segments a list of like (children_type, ownership, aoh)
    # Extract category keys and build prefixed values
    category_keys = list(model_desc['hh_categories'].keys())
    category_values = [
        [f"{cat_name}_{cat_val}" for cat_val in model_desc['hh_categories'][cat_name]]
        for cat_name in category_keys
    ]
    # Cartesian product of all category combinations
    segments = list(itertools.product(*category_values))
    return segments 