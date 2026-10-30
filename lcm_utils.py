

import zlib
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


def pick_slots(scores, n, method='slot_sampling', seed=0, log_scores=None):
    """Choose n slots out of the scored candidates.

    Parameters:
        scores (np.ndarray): Score per candidate slot, used for 'top_n'.
        n (int): Slots to fill.
        method (str): 'slot_sampling' draws slots without replacement with
            probability proportional to exp(log_scores); 'top_n' takes the n
            highest by `scores`, which is what this did before 2026-09-09.
            'sample' is accepted as the former name for slot_sampling, so model
            directories written before the rename still load.
        seed (int): Seeds the draw, so a run is reproducible.
        log_scores (np.ndarray, optional): Scores on a log scale, for sampling.
            Falls back to log(scores).

    Returns:
        np.ndarray: Positions into `scores`.
    """
    n = int(min(n, len(scores)))
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    if method == 'top_n':
        return np.argsort(scores)[-n:]
    if method not in ('slot_sampling', 'sample'):
        raise ValueError("placement must be 'slot_sampling' or 'top_n', got %r"
                         % method)

    if log_scores is None:
        w = np.clip(np.nan_to_num(np.asarray(scores, dtype='float64'), nan=0.0), 0.0, None)
        if w.sum() <= 0:
            return np.argsort(scores)[-n:]
        with np.errstate(divide='ignore'):
            keys = np.log(w)
    else:
        keys = np.nan_to_num(np.asarray(log_scores, dtype='float64'),
                             nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    rng = np.random.default_rng(seed)
    finite = np.isfinite(keys)
    if not finite.any():
        return np.argsort(scores)[-n:]
    keys = np.where(finite, keys + rng.gumbel(size=keys.shape), -np.inf)
    return np.argpartition(keys, -n)[-n:]


def _elcm_calibrate(calibration, pred, utility, predict_X_df, final_alts_df,
                    la_id, job_sector, bld_age_var, taz_emp_ratio_var):
    """Apply the hand-tuned multipliers to a segment's scores.

    Returns both scales, because ranking uses the probability and slot sampling
    uses the log. Off by default -- see the note at the call site.

    Parameters:
        calibration (dict): The `calibration` block from model_structure.yaml.
        pred (np.ndarray): sigmoid(utility) per candidate slot.
        utility (np.ndarray): Raw pre-sigmoid score per candidate slot.
        predict_X_df (pd.DataFrame): The scored candidates, indexed by building.
        final_alts_df (pd.DataFrame): Alternatives carrying the weight inputs.
        la_id, job_sector: Identify the segment, for the per-segment overrides.
        bld_age_var, taz_emp_ratio_var (str): Column names.

    Returns:
        tuple[np.ndarray, np.ndarray]: (pred_weighted, logit_weighted).
    """
    method = calibration.get("calibration_method", "multiplicative")
    weights = calibration.get("weights", {})
    use_taz_cluster = calibration.get(
        "use_taz_cluster_by_sector", {}).get(job_sector, False)
    override_key = f"la_{la_id}_sector_{job_sector}"
    if override_key in calibration.get("overrides", {}):
        method = calibration["overrides"][override_key].get("calibration_method", method)
        weights = calibration["overrides"][override_key].get("weights", weights)

    w_age = weights.get("building_age", 1.0)
    w_vac = weights.get("vacancy", 1.0)
    w_btype = weights.get("btype_matrix", 1.0)
    w_taz = weights.get("taz_cluster", 1.0)

    building_age_weights = np.array([5.0, 3.0, 1.0])[np.digitize(
        final_alts_df.loc[predict_X_df.index, bld_age_var], [16, 31])]
    vacancy_weights = final_alts_df.loc[predict_X_df.index, 'vacancy_weight'].to_numpy()
    job_btype_arr = final_alts_df.loc[predict_X_df.index, 'job_btype_ratio'].to_numpy()

    if use_taz_cluster:
        taz_arr = final_alts_df.loc[predict_X_df.index, taz_emp_ratio_var].to_numpy()
        lo, hi = taz_arr.min(), taz_arr.max()
        taz_cluster_arr = np.ones_like(taz_arr) if lo == hi else (taz_arr - lo) / (hi - lo)
    else:
        taz_cluster_arr = np.ones(len(predict_X_df))

    if method == 'log_weighted':
        total = w_age + w_vac + w_btype + w_taz
        log_cal = ((w_age / total) * np.log(building_age_weights + 1e-6) +
                   (w_vac / total) * np.log(vacancy_weights + 1e-6) +
                   (w_btype / total) * np.log(job_btype_arr + 1e-6) +
                   (w_taz / total) * np.log(taz_cluster_arr + 1e-6))
        pred_weighted = pred * np.exp(log_cal)
    else:
        pred_weighted = (pred * building_age_weights * vacancy_weights
                         * job_btype_arr * taz_cluster_arr)
        with np.errstate(divide='ignore'):
            log_cal = (np.log(building_age_weights + 1e-12)
                       + np.log(vacancy_weights + 1e-12)
                       + np.log(job_btype_arr + 1e-12)
                       + np.log(taz_cluster_arr + 1e-12))
    return pred_weighted, utility + log_cal


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
            variable_cols + alts_filter_cols + [alt_capacity, space_col, 'building_type_id',
                                                'stories', bld_age_var, taz_emp_ratio_var,
                                                'residential_units']
        )))
        print(f"[alts-timing] ELCM la{la_id} sec{job_sector} "
              f"alts_to_frame={(time.perf_counter() - _t_alts) * 1000:.0f}ms "
              f"rows={len(alts_df)} cols={alts_df.shape[1]}", flush=True)

        # Set bld_age_var to 0 for 10+ story offices
        mask_office_10plus = (alts_df['building_type_id'] == 23) & (alts_df['stories'] >= 10)
        alts_df.loc[mask_office_10plus, bld_age_var] = 0

        alts_idx = alts_df.query(alts_pre_filter).index
        final_alts_df = alts_df.loc[alts_idx].query(alt_filter)

        # residential_units is kept explicitly
        final_alts_df = final_alts_df[list(set(
            variable_cols + [alt_capacity, space_col, bld_age_var, 'building_type_id',
                             taz_emp_ratio_var, 'residential_units']
        ))]

        # jobs placing rule defined in esimation, reused here
        # default to slot_sampling
        _pl = model_desc.get('placement') or {}
        placement = _pl.get('rule', 'slot_sampling')
        # k_candidates is either one number for both segment types or a mapping
        # with home_based / non_home_based keys
        _k = _pl.get('k_candidates', 100)
        k_candidates = int(_k.get('home_based' if home_based else 'non_home_based',
                                  100) if isinstance(_k, dict) else _k)
        if 'placement' in (elcm_calibration_config or {}):
            print('[placement] %s: model_structure.yaml still sets placement; it is '
                  'ignored, run.placement in the estimation config is authoritative'
                  % model_name, flush=True)

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
            repeated_idx = pd.Series(
                np.repeat(final_alts_df.index.to_numpy(),
                          final_alts_df[alt_capacity].to_numpy().astype(np.int64))
            )
            _n_full = len(repeated_idx)
            M = min(_n_full, n * k_candidates)
            sampled_idx = repeated_idx.sample(M, replace=False, random_state=0).to_numpy()
            predict_X_df = final_alts_df.loc[sampled_idx, variable_cols]
        else:
            hb_capacity = final_alts_df['residential_units'].to_numpy()
            hb_capacity = np.nan_to_num(hb_capacity, nan=0.0)
            hb_capacity = np.clip(hb_capacity, 0, None).astype(np.int64)
            repeated_idx = pd.Series(
                np.repeat(final_alts_df.index.to_numpy(), hb_capacity))
            _n_full = len(repeated_idx)
            if _n_full == 0:
                return
            M = min(_n_full, n * k_candidates)
            sampled_idx = repeated_idx.sample(M, replace=False, random_state=0).to_numpy()
            predict_X_df = final_alts_df.loc[sampled_idx, variable_cols]

        # clean + scale only the M sampled rows (was previously done on the full
        # expanded matrix before sampling)
        predict_X_df = predict_X_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        _t_sc = time.perf_counter()
        # Prefer the scaler fitted during estimation
        scaler_state = getattr(model, 'scaler_state', None)
        if scaler_state is not None:
            predict_X_df = apply_scaler_state(predict_X_df, scaler_state)
        else:
            scaler = RobustScaler()
            predict_X_df = pd.DataFrame(
                scaler.fit_transform(predict_X_df),
                columns=predict_X_df.columns,
                index=predict_X_df.index
            )
            predict_X_df = np.clip(predict_X_df, -5, 5)
        _t_fw = time.perf_counter()
        with torch.no_grad():
            _u = model.utility(
                torch.tensor(predict_X_df.values, dtype=torch.float)
            ).detach().cpu().numpy().flatten()

        # sigmoid of utility _u
        pred = 1.0 / (1.0 + np.exp(-_u))
        print(f"[lcm-timing] ELCM la{la_id} sec{job_sector} full={_n_full} M={M} "
              f"scale={(_t_fw - _t_sc) * 1000:.0f}ms fwd={(time.perf_counter() - _t_fw) * 1000:.0f}ms",
              flush=True)

        # === CALIBRATION ===
        calibration = elcm_calibration_config or {}
        if not calibration.get("enabled", False):
            pred_weighted = pred
            logit_weighted = _u
        else:
            pred_weighted, logit_weighted = _elcm_calibrate(
                calibration, pred, _u, predict_X_df, final_alts_df,
                la_id, job_sector, bld_age_var, taz_emp_ratio_var)

        _yr = orca.get_injectable('year') if orca.is_injectable('year') else 0
        picked_idx = pick_slots(pred_weighted, n, placement,
                                seed=(int(_yr or 0) * 100000
                                      + int(la_id) * 1000 + int(job_sector)),
                                log_scores=logit_weighted)
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

def _hlcm_calibration_target(base_ratios, current_ratios, zero_base_share='floor'):
    """Tract calibrator target per candidate row: base-year share / current share.

    Above 1 the tract has lost this household type since the base year. A tract
    with no base-year households of the type gets 0, the floor once clipped
    ('floor'), or 1.0, no adjustment ('neutral').
    """
    base = np.asarray(base_ratios, dtype='float64')
    with np.errstate(divide='ignore', invalid='ignore'):
        y = np.divide(base, np.asarray(current_ratios, dtype='float64'))
    y = np.nan_to_num(y, nan=1.0, posinf=2.0, neginf=0.5)
    if zero_base_share == 'neutral':
        y = np.where(np.nan_to_num(base) == 0, 1.0, y)
    elif zero_base_share != 'floor':
        raise ValueError("zero_base_share must be 'floor' or 'neutral', got %r" % (zero_base_share,))
    return y


def assign_sqft_band(sqft, large_area, cuts):
    """Unit-size band per row, from base-year cut points held in square feet."""
    v = pd.Series(sqft, dtype='float64').fillna(0.0)
    vals, la = v.to_numpy(), np.asarray(large_area)
    out = np.zeros(len(v), dtype='int16')
    for k, c in cuts.items():
        m = la == k
        if m.any() and len(c):
            out[m] = np.searchsorted(np.asarray(c, dtype='float64'), vals[m],
                                     side='right')
    return pd.Series(out, index=v.index)


def _hlcm_fill_prior(seg_var, rows, la_id, clip):
    """Score multiplier on units built during the run, 1 on existing stock.

    A new unit draws the segment mix of comparable base-year new stock, over its
    large area's average, so the prior redistributes among new units without
    changing how many are filled.
    """
    ratios = orca.get_injectable('new_unit_fill_ratios')
    if seg_var not in ratios.columns:
        return np.ones(len(rows)), 0
    band = assign_sqft_band(rows['sqft_per_unit'], la_id,
                            orca.get_injectable('sqft_band_cuts'))
    key = pd.MultiIndex.from_arrays(
        [np.full(len(rows), la_id, dtype='int64'),
         rows['building_type_id'].to_numpy(), band.to_numpy()])
    r = np.nan_to_num(ratios[seg_var].reindex(key).to_numpy(), nan=1.0)
    is_new = rows['year_built'].to_numpy() > orca.get_injectable('base_year')
    return np.clip(np.where(is_new, r, 1.0), clip[0], clip[1]), int(is_new.sum())


def _hlcm_expected_share(seg_var):
    """Tract target that follows what the tract actually built.

    Existing units hold their base-year mix. Units added during the run hold the
    tract's own base-year mix tilted by the new-stock ratio and renormalized over
    the segments
    """
    ratios = orca.get_injectable('new_unit_fill_ratios')
    base_df = orca.get_table('tract_hh_type_base_ratios').to_frame()
    cols = [c for c in ratios.columns if c in base_df.columns]
    if seg_var not in cols:
        return None
    agg = orca.get_injectable('new_unit_band_counts')
    cnt, tot = agg['new'], agg['total']
    b = base_df.reindex(cnt.index.get_level_values('tract_id'))[cols].to_numpy()
    r = ratios.reindex(cnt.index.droplevel('tract_id'))[cols].fillna(1.0).to_numpy()
    tilt = b * r
    p = tilt[:, cols.index(seg_var)] / np.clip(tilt.sum(1), 1e-12, None)
    new_exp = pd.Series(cnt.to_numpy() * np.nan_to_num(p),
                        index=cnt.index).groupby(level='tract_id').sum()
    base = base_df[seg_var]
    old_exp = agg['existing'] * base.reindex(agg['existing'].index).fillna(0.0)
    return ((new_exp.reindex(tot.index).fillna(0.0)
             + old_exp.reindex(tot.index).fillna(0.0))
            / tot.replace(0, np.nan))


def _hlcm_shrink_target(tract_targets, base_counts, k=30.0):
    """Blend each tract's calibration target toward 1 by base-year cell mass.

    A tract holding a handful of base-year households of a segment carries mostly
    sampling noise: a quarter of tract x segment cells sit at a base share of 2% or
    less, where two or three households halve the ratio and peg the weight at the
    clip floor. n / (n + k) leaves thick cells alone and sends thin ones to 1, the
    shrinkage demolition_calibration._shrunk_rates uses on demolition rates.

    Returns:
        tuple[pd.Series, float]: shrunk targets, mean blend weight.
    """
    if k <= 0:
        return tract_targets, 1.0
    n = pd.Series(base_counts, dtype='float64').reindex(tract_targets.index).fillna(0.0)
    w = n / (n + float(k))
    return 1.0 + (tract_targets - 1.0) * w, float(w.mean())


def _hlcm_tract_adjustment(tract_weights, tract_ids, clip=(0.5, 2.0), strength=1.0):
    """Per-row score multiplier from predicted tract weights.

    Weights are bounded to `clip`, then raised to `strength`, which scales the
    adjustment on the log scale: 0.5 turns a 2x boost into 1.41x, 0 switches it
    off. Rows in tracts without a prediction get 1.

    Returns:
        tuple[np.ndarray, float]: multiplier per row, share of tracts at a bound.
    """
    lo, hi = float(clip[0]), float(clip[1])
    w = pd.Series(tract_weights, dtype='float64')
    at_bound = float(((w <= lo) | (w >= hi)).mean()) if len(w) else 0.0
    w = w.clip(lower=lo, upper=hi) ** float(strength)
    return w.reindex(tract_ids).fillna(1.0).to_numpy(), at_bound


def _largest_remainder(x, total):
    """Integerize `x` so it sums to `total`, giving the spare units to the largest
    fractional parts. Rounding each tract independently drifts by tens of slots
    across 1,300 tracts."""
    x = np.asarray(x, dtype='float64')
    base = np.floor(x).astype(np.int64)
    short = int(total) - int(base.sum())
    if short > 0:
        base[np.argsort(-(x - base))[:short]] += 1
    elif short < 0:
        order = np.argsort(x - base)
        order = order[base[order] > 0][:-short]
        base[order] -= 1
    return base


def _hlcm_stratified_counts(tract_ids, log_scores, tract_weights, n):
    """Slots to draw per tract: the model's own mass, tilted by the calibrator.

    The score path adds ln(w) to every slot's key and lets the Gumbel draw settle
    it.

    Returns:
        tuple[pd.Series, dict]: draws per tract_id summing to min(n, capacity),
        and diagnostics.
    """
    t = pd.Series(tract_ids).fillna(-1)
    cap = t.value_counts()
    n = int(min(n, len(t)))
    if n <= 0:
        return pd.Series(dtype='int64'), {}
    # the model's own share of the sampling mass, max-shifted so exp cannot overflow
    ls = np.nan_to_num(np.asarray(log_scores, dtype='float64'), neginf=-np.inf)
    finite = np.isfinite(ls)
    e = np.zeros(len(ls))
    if finite.any():
        e[finite] = np.exp(ls[finite] - ls[finite].max())
    mass = pd.Series(e).groupby(t.to_numpy()).sum().reindex(cap.index).fillna(0.0)
    if mass.sum() <= 0:
        mass = cap.astype('float64')
    w = pd.Series(tract_weights, dtype='float64').reindex(cap.index).fillna(1.0)
    # water-fill: hand capped tracts their capacity, re-spread the rest by w * mass
    alloc = pd.Series(0.0, index=cap.index)
    free = cap.index
    rem = float(n)
    for _ in range(50):
        a = (w[free] * mass[free]).clip(lower=0.0)
        if a.sum() <= 0:
            a = cap[free].astype('float64')
        want = rem * a / a.sum()
        over = want > cap[free]
        if not over.any():
            alloc[free] = want
            break
        hit = free[over.to_numpy()]
        alloc[hit] = cap[hit].astype('float64')
        rem -= float(cap[hit].sum())
        free = free[~over.to_numpy()]
        if rem <= 0 or len(free) == 0:
            break
    out = pd.Series(_largest_remainder(alloc.to_numpy(), n), index=cap.index)
    # rounding can push a tract one slot past its capacity; move those to tracts
    # that still have room
    for _ in range(10):
        over = out > cap
        if not over.any():
            break
        excess = int((out[over] - cap[over]).sum())
        out[over] = cap[over]
        slack = (cap - out).clip(lower=0)
        room = slack[slack > 0]
        if excess <= 0 or room.empty:
            break
        add = _largest_remainder(excess * room / room.sum(), excess)
        out[room.index] = out[room.index] + np.minimum(add, room.to_numpy())
    diag = dict(tracts=len(cap), placed=int(out.sum()), asked=n,
                at_capacity=int((out >= cap).sum()),
                tilt=float(np.abs(np.log(w.clip(lower=1e-12))).mean()))
    return out, diag


def _pick_stratified(scores, log_scores, tract_ids, counts, method, seed=0):
    """Draw each tract's allotted slots with the model's own scores.

    The tract weight is constant inside a tract, so it cancels here and only the
    model (and the fill prior, which does vary by unit) decides which slots go.
    """
    t = pd.Series(tract_ids).fillna(-1).to_numpy()
    pos = np.arange(len(t))
    out = []
    for tid, grp in pd.Series(pos).groupby(t):
        k = int(counts.get(tid, 0))
        if k <= 0:
            continue
        idx = grp.to_numpy()
        sub = pick_slots(np.asarray(scores)[idx], k, method,
                         seed=int(seed ^ zlib.crc32(repr(tid).encode())),
                         log_scores=np.asarray(log_scores)[idx])
        out.append(idx[sub])
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def _hlcm_calibration_regressor(y_train, tract_ids):
    """Smooth the per-tract calibration target through an XGBoost fit on census
    tract features, instead of applying it directly.

    Superseded: the target is a tract's own accumulated drift from its base-year
    share, which tract attributes do not predict.

    Returns:
        pd.Series: predicted weight per tract_id.
    """
    tracts_df = orca.get_table('census_tracts').to_frame()
    # Handle missing tract_id = -1
    if -1 not in tracts_df.index:
        tracts_df.loc[-1] = tracts_df.mean(numeric_only=True)

    # Tract features for the buildings used, numeric float columns only
    used_tract_ids = pd.Series(tract_ids).dropna().unique()
    valid_columns = tracts_df.select_dtypes(include=['float', 'float32', 'float64']).columns
    valid_columns = [col for col in valid_columns if col not in ['tazce10_n']]

    X_train = tracts_df.loc[tracts_df.index.intersection(used_tract_ids), valid_columns].fillna(0.0)
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.select_dtypes(include='number').columns
    )
    low_variance_cols = X_train_scaled.var()[X_train_scaled.var() < 1e-5].index
    X_train_scaled.drop(columns=low_variance_cols, inplace=True)

    # Aggregate the per-row target to tract level
    tract_weights = pd.Series(y_train, index=tract_ids).groupby(tract_ids).mean()
    y_tract = tract_weights.reindex(X_train_scaled.index).fillna(1.0)

    naive_pred = np.full_like(y_tract.values, fill_value=y_tract.mean())
    print("[Calibrator] Pre-train baseline error: MSE=%.4f, MAE=%.4f"
          % (mean_squared_error(y_tract.values, naive_pred),
             mean_absolute_error(y_tract.values, naive_pred)))

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
        device="cuda"
    )
    model.fit(X_train_scaled.values, y_tract.values)

    y_pred = model.predict(X_train_scaled.values)
    print("[Calibrator] Post-train error:        MSE=%.4f, MAE=%.4f"
          % (mean_squared_error(y_tract.values, y_pred),
             mean_absolute_error(y_tract.values, y_pred)))
    return pd.Series(y_pred, index=X_train_scaled.index)


def register_hlcm_model_step(model_name, alt_capacity='residential_units', hlcm_calibration_config=None):

    # TODO: Update simulate steps with lcm nn model
    @orca.step(model_name)
    def choice_model_simulate(hh_location_choice_models):
        model = hh_location_choice_models[model_name]

        model_path = orca.get_injectable('hlcm_model_path')
        model_desc_path = os.path.join(model_path, 'model_description.yaml')
        with open(model_desc_path, 'r') as f:
            model_desc = yaml.load(f, Loader=yaml.FullLoader)

        # How to place households, read from the estimation run's
        # model_description.yaml so the rule the reported zone accuracy assumed is
        # the rule used here. Model directories written before 2026-09-14 carry no
        # placement block and take slot_sampling with 5 candidate slots.
        _pl = model_desc.get('placement') or {}
        placement = _pl.get('rule', 'slot_sampling')
        k_candidates = int(_pl.get('k_candidates', 5))

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
        # the calibrator's extra hold on single dimensions needs each of those
        # dimensions' tract ratios next to the joint one
        # (models.hlcm.calibration.marginal)
        marginals = _hlcm_marginals(tract_segment_type_var, hlcm_calibration_config or {})
        cal_cols = [tract_segment_type_var] + [m[1] for m in marginals]
        # pulled for the fill prior; year_built and sqft_per_unit are also model
        # features, so they are read from fill_raw below, not from alts_df
        fill_cols = ['building_type_id', 'sqft_per_unit', 'year_built']
        _t_alts = time.perf_counter()
        alts_df = alts.to_frame(list(set(formula_alts_col+alts_filter_cols+fill_cols
                                         +[alt_capacity]+cal_cols)))
        print(f"[alts-timing] HLCM la{la_id} "
              f"alts_to_frame={(time.perf_counter() - _t_alts) * 1000:.0f}ms "
              f"rows={len(alts_df)} cols={alts_df.shape[1]}", flush=True)

        # query using alts_pre_filter to match whats used in estimation
        alts_idx = alts_df.query(alts_pre_filter).index

        raw_res_units = None
        if 'residential_units' in alts_df.columns:
            raw_res_units = alts_df.loc[alts_idx, 'residential_units'].astype(float).copy()

        # year_built and sqft_per_unit are model features as well, and the
        # scaling below overwrites them in place; the fill prior needs the raw
        # values, so snapshot them here
        fill_raw = alts_df.loc[alts_idx, fill_cols].copy()

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
        final_alts_df = final_alts_df[list(set(
            formula_alts_col + [alt_capacity] + cal_cols))]
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
        M = min(len(predict_X_df), n * k_candidates)  # HU pool count
        predict_X_df = predict_X_df.sample(M, replace=False, random_state=0)

        # run predict
        # Housing-unit correction, run if enable
        if getattr(model, 'sampling_correction', False):
            units = None
            if raw_res_units is not None:
                units = raw_res_units.reindex(predict_X_df.index).to_numpy(dtype=float)
            if units is None or not np.isfinite(units).all():
                raise ValueError(
                    "%s was trained with the housing-unit correction, so "
                    "simulation must subtract ln(residential_units) from its "
                    "score -- but residential_units is unavailable for these "
                    "alternatives. Re-estimate with "
                    "run.sampled_softmax.apply_sampling_correction: false, or "
                    "make residential_units available here." % model_name)
            units = np.clip(units, 1.0, None)
            with torch.no_grad():
                _u = model.utility(
                    torch.tensor(predict_X_df.values, dtype=torch.float)
                ).detach().cpu().numpy().flatten()
            pred = 1.0 / (1.0 + np.exp(-(_u - np.log(units))))
            print("[hu-correction] %s: subtracted ln(units) from %d scored rows "
                  "(per-building -> per-unit)" % (model_name, len(pred)), flush=True)
        else:
            with torch.no_grad():
                _u = model.utility(
                    torch.tensor(predict_X_df.values, dtype=torch.float)
                ).detach().cpu().numpy().flatten()
            pred = 1.0 / (1.0 + np.exp(-_u))
        # the same score on the log scale slot sampling needs. Sampling on pred,
        # the sigmoid, flattens the distribution and measures worse than top_n.
        log_score = (_u - np.log(units)) if getattr(model, 'sampling_correction', False) else _u

        # === CALIBRATION ===
        # tract calibrator, set by models.hlcm.calibration in model_structure.yaml;
        # without that block it runs at full strength as before
        calibration = hlcm_calibration_config or {}
        cal_strength = float(calibration.get('strength', 1.0))
        if calibration.get('enabled', True) and cal_strength > 0:
            # Build training targets from observed & base ratios
            base_ratios_df = orca.get_table("tract_hh_type_base_ratios").to_frame()
            current_ratios = final_alts_df.loc[predict_X_df.index, tract_segment_type_var].to_numpy()

            bld_to_tract = alts.to_frame(['tract_id'])['tract_id']
            tract_ids = bld_to_tract.reindex(predict_X_df.index).to_numpy()
            base_ratios = base_ratios_df[tract_segment_type_var]
            if calibration.get('expected_share', True) and orca.is_injectable('new_unit_fill_ratios'):
                # hold the tract to what it built, not to its 2025 mix
                exp = _hlcm_expected_share(tract_segment_type_var)
                if exp is not None:
                    base_ratios = exp.reindex(base_ratios.index).fillna(base_ratios)
            base_ratios = base_ratios.reindex(tract_ids).to_numpy()

            # Compute y_train = base / current (avoid divide-by-zero)
            y_train = _hlcm_calibration_target(
                base_ratios, current_ratios, calibration.get('zero_base_share', 'floor'))

            # Apply base/current per tract directly, or smooth it through the
            # regressor. Direct is the default
            if calibration.get('use_regressor', False):
                tract_predicted_weights = _hlcm_calibration_regressor(y_train, tract_ids)
            else:
                tract_predicted_weights = pd.Series(y_train, index=tract_ids).groupby(tract_ids).mean()
            # a thin cell's ratio is noise; shrink it toward 1 by base-year mass
            shrink_k, shrink_w = float(calibration.get('shrink_k', 30.0)), 1.0
            if 'tract_hh_total' in base_ratios_df.columns:
                tract_predicted_weights, shrink_w = _hlcm_shrink_target(
                    tract_predicted_weights,
                    base_ratios_df[tract_segment_type_var] * base_ratios_df['tract_hh_total'],
                    k=shrink_k)
            tract_segment_adj_arr, cal_at_bound = _hlcm_tract_adjustment(
                tract_predicted_weights, tract_ids,
                clip=calibration.get('clip', (0.5, 2.0)), strength=cal_strength)

            # Extra hold on single dimensions' tract composition, each multiplied
            # into the same weight. The joint cell is 1/12 of a tract's households,
            # so its ratio is noisy and strength above ~3 regresses; a marginal
            # pools the cells sharing one value (an income tercile is 4 of the 12,
            # a children or senior split 6) and takes a harder pull. A dimension is
            # then held by its own weight as well as by the joint cell's, which is
            # the point. Several together compete for the same placements, so their
            # strengths trade off against each other.
            for m_dim, m_var, m_strength, m_clip, m_k in marginals:
                if m_var not in base_ratios_df.columns:
                    print('[Marginal] %s: %s missing from tract_hh_type_base_ratios, '
                          'skipped' % (model_name, m_var), flush=True)
                    continue
                m_base = base_ratios_df[m_var].reindex(tract_ids).to_numpy()
                m_cur = final_alts_df.loc[predict_X_df.index, m_var].to_numpy()
                m_y = _hlcm_calibration_target(
                    m_base, m_cur, calibration.get('zero_base_share', 'floor'))
                m_weights = pd.Series(m_y, index=tract_ids).groupby(tract_ids).mean()
                m_shrink = 1.0
                if 'tract_hh_total' in base_ratios_df.columns:
                    m_weights, m_shrink = _hlcm_shrink_target(
                        m_weights,
                        base_ratios_df[m_var] * base_ratios_df['tract_hh_total'],
                        k=m_k)
                m_arr, m_at_bound = _hlcm_tract_adjustment(
                    m_weights, tract_ids, clip=m_clip, strength=m_strength)
                tract_segment_adj_arr = tract_segment_adj_arr * m_arr
                print('[Marginal] %s: %s (%s) strength %.2f | weight min %.3f '
                      'median %.3f max %.3f | mean |ln w| %.4f | at bound %.1f%% | '
                      'shrink k %g mean w %.2f'
                      % (model_name, m_dim,
                         m_var.replace('tract_hh_type_ratio_', ''), m_strength,
                         m_arr.min(), np.median(m_arr), m_arr.max(),
                         np.abs(np.log(m_arr)).mean(), 100 * m_at_bound, m_k, m_shrink),
                      flush=True)
            print('[Calibrator] %s: strength %.2f | weight min %.3f median %.3f max %.3f | '
                  'mean |ln w| %.4f | tracts at clip bound %.1f%% | %d tracts, '
                  'target median %.3f spread %.3f | shrink k %g mean w %.2f'
                  % (model_name, cal_strength, tract_segment_adj_arr.min(),
                     np.median(tract_segment_adj_arr), tract_segment_adj_arr.max(),
                     np.abs(np.log(tract_segment_adj_arr)).mean(), 100 * cal_at_bound,
                     len(tract_predicted_weights), tract_predicted_weights.median(),
                     tract_predicted_weights.std(), shrink_k, shrink_w))

        else:
            tract_segment_adj_arr = np.ones(len(predict_X_df))
            tract_ids, tract_predicted_weights = None, None

        fill_arr = np.ones(len(predict_X_df))
        if calibration.get('fill_prior', True) and orca.is_injectable('new_unit_fill_ratios'):
            fill_arr, n_new = _hlcm_fill_prior(
                tract_segment_type_var,
                fill_raw.loc[predict_X_df.index], int(la_id),
                calibration.get('clip', (0.5, 2.0)))
            print('[FillPrior] %s: %d of %d candidate units built since the base '
                  'year | weight min %.3f median %.3f max %.3f'
                  % (model_name, n_new, len(fill_arr), fill_arr.min(),
                     np.median(fill_arr), fill_arr.max()))

        # Apply individual weight components
        # default to multiplicative calibration
        pred_weighted = pred * tract_segment_adj_arr * fill_arr
        # how wide the model's own scores run next to the calibrator's ln(w): the
        # ratio sets how much of an intended tilt a Gumbel draw can realize
        print('[Score] %s: log_score sd %.3f, p10-p90 %.3f over %d slots'
              % (model_name, float(np.std(log_score)),
                 float(np.subtract(*np.percentile(log_score, [90, 10]))),
                 len(log_score)))
        # the calibrator weight on the log scale: under slot sampling it scales the
        # probability of a slot being drawn rather than only reordering the ranking
        logit_weighted = (log_score + np.log(np.clip(tract_segment_adj_arr, 1e-12, None))
                          + np.log(np.clip(fill_arr, 1e-12, None)))
        # the same score without the tract factor: the stratified path applies that
        # factor to the per-tract counts, so folding it into the mass as well would
        # square it. The fill prior stays in, since it varies unit by unit and so
        # belongs to the model's own preference over slots.
        logit_model = log_score + np.log(np.clip(fill_arr, 1e-12, None))

        # year in the seed, or every simulation year draws the same random stream
        _yr = orca.get_injectable('year') if orca.is_injectable('year') else 0
        _seed = zlib.crc32(('%s|%s' % (_yr, model_name)).encode())
        # 'score': nudges every slot's odds and lets the draw settle it; 
        # 'stratified': spends the same weight on how many slots each tract gets and leaves the
        #   choice within a tract to the model
        if (calibration.get('method', 'score') == 'stratified'
                and tract_ids is not None):
            counts, cdiag = _hlcm_stratified_counts(
                tract_ids, logit_model, tract_predicted_weights, n)
            picked_idx = _pick_stratified(pred_weighted, logit_model, tract_ids,
                                          counts, placement, seed=_seed)
            print('[Stratified] %s: %d of %d slots over %d tracts | %d tracts at '
                  'capacity | mean |ln w| %.4f'
                  % (model_name, cdiag.get('placed', 0), cdiag.get('asked', 0),
                     cdiag.get('tracts', 0), cdiag.get('at_capacity', 0),
                     cdiag.get('tilt', 0.0)))
        else:
            picked_idx = pick_slots(
                pred_weighted, n, placement, seed=_seed,
                log_scores=logit_weighted)
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
        orca.run(["build_networks"])
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
        orca.run(["build_networks"])
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

def get_hlcm_categories():
    """Household categories of the HLCM model description, values prefixed by the
    category name: {'income': ['income_lowinc', ...], 'children': [...], ...}.
    get_hlcm_segment is the cartesian product of these.
    """
    model_path = orca.get_injectable('hlcm_model_path')
    with open(os.path.join(model_path, 'model_description.yaml'), 'r') as f:
        model_desc = yaml.load(f, Loader=yaml.FullLoader)
    return {cat_name: [f"{cat_name}_{cat_val}" for cat_val in cat_vals]
            for cat_name, cat_vals in model_desc['hh_categories'].items()}


def get_hlcm_segment():
    """Generate hlcm segments variables based on model description hlcm_model_path.
    orca.get_injectable('hlcm_model_path')

    Returns:
        [(hh_cat1, hh_cat_2, hh_cat_3), ...]: products of all household categories
    """
    return list(itertools.product(*get_hlcm_categories().values()))


def _hlcm_marginal_var(segment_var, dim, categories=None):
    """Tract ratio column for one dimension of a segment: the income tercile of
    tract_hh_type_ratio_children_no_children_senior_with_seniors_income_lowinc is
    tract_hh_type_ratio_income_lowinc. None when `dim` is not a household category
    of this model. The marginal pools every cell sharing that value, so it carries
    more mass - and less sampling noise - than the joint cell does.
    `categories` defaults to the model description's (get_hlcm_categories).
    """
    cats = get_hlcm_categories() if categories is None else categories
    for val in cats.get(dim, []):
        if val in segment_var:
            return 'tract_hh_type_ratio_%s' % val
    return None


def _hlcm_marginals(segment_var, calibration, categories=None):
    """The marginal holds to apply to one segment, from models.hlcm.calibration.

    Reads the `marginal` block: `dimensions` maps a household category to either a
    strength or a block of its own ({strength, clip, shrink_k}); `clip` and
    `shrink_k` beside it are the defaults, falling back to the joint weight's.
    A dimension with strength 0, or absent, is off.

    Returns [(dim, var, strength, clip, shrink_k), ...], empty when none apply.
    """
    cfg = calibration.get('marginal') or {}
    out = []
    for dim, spec in (cfg.get('dimensions') or {}).items():
        spec = spec if isinstance(spec, dict) else {'strength': spec}
        strength = float(spec.get('strength', 0) or 0)
        var = _hlcm_marginal_var(segment_var, dim, categories) if strength > 0 else None
        if not var:
            continue
        clip = spec.get('clip', cfg.get('clip', calibration.get('clip', (0.5, 2.0))))
        k = spec.get('shrink_k', cfg.get('shrink_k', calibration.get('shrink_k', 30.0)))
        out.append((dim, var, strength, tuple(clip), float(k)))
    return out
