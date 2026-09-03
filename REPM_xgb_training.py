#!/usr/bin/env python
"""
REPM XGBoost Training - Train real estate price models with XGBoost.

Run in background:
nohup python REPM_xgb_training.py > \
  runs/training_logs/repm_train_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
"""

import os
import numpy as np
import pandas as pd
import orca
import scipy
import time
import yaml
import joblib
import pickle
from tqdm import tqdm
from pathlib import Path

from utils import apply_filter_query

# Suppress warnings
warnings = __import__('warnings')
warnings.warn = lambda *args, **kwargs: None

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.dummy import DummyRegressor

# ==============================================================================
# CONFIGURATION - Edit these settings as needed
# ==============================================================================

# Output directory for trained models
REPM_XGB_PATH = "./configs/repm_xgb/"

# Grid search settings
USE_GRID_SEARCH = True  # Set to True for hyperparameter tuning (slower)
# GRID_SEARCH_BASELINE options:
#   - None: Use default grid search parameters
#   - "auto": Automatically load existing grid_search_{model_name}.yaml for each segment
#   - "/path/to/dir/": Load from specified directory (auto per segment)
#   - "/path/to/file.yaml": Use single baseline for all segments (not recommended)
GRID_SEARCH_BASELINE = "auto"  # Auto-load per-segment baselines for refined search

# ==============================================================================
# ADVANCED CONFIGURATION (usually doesn't need to be changed)
# ==============================================================================

# Feature selection thresholds
VARIANCE_THRESHOLD = 1e-6  # Remove features with near-zero variance
CORRELATION_THRESHOLD = 0.95  # Remove features with correlation > 0.95

# Simple model threshold: use Ridge regression for samples below this size
SIMPLE_MODEL_THRESHOLD = 100  # Samples below this use Ridge instead of XGBoost
MIN_SAMPLES_FOR_RIDGE = 4  # mutual-information feature selection needs >= 4 records
REGIONAL_DUMMY_HEDONIC_IDS = {96}  # Data Center: 31 regional records, not enough for a hedonic fit

# Feature selection based on sample size (reduce features when data is limited)
# Returns max features to keep based on sample size
def get_max_features(sample_size):
    """Calculate maximum features to keep based on sample size."""
    if sample_size >= 500:
        return None  # Keep all features
    elif sample_size >= 200:
        return 150  # Moderate reduction
    elif sample_size >= 100:
        return 100  # More reduction
    elif sample_size >= 50:
        return 50   # Significant reduction
    else:
        return 30   # Aggressive reduction for very small samples

# Filter columns needed for model training (never skip these)
FILTER_COLS = ["sqft_price_nonres", "sqft_price_res", "non_residential_sqft",
               "hedonic_id", "residential_units"]


def _should_skip_var(var: str) -> bool:
    """Check if variable should be excluded from training."""
    # Never skip filter columns
    if var in FILTER_COLS:
        return False

    var_lower = var.lower()

    # Bike mode share is too small for bike-nearest indicators to be a
    # defensible regional price driver.
    if var_lower.startswith('bike_nearest_'):
        return True

    # === SKIP: All parcels_ variables (duplicates of building vars) ===
    if var_lower.startswith('parcels_'):
        return True

    # === SKIP: Geographic/ID variables ===
    skip_id_geog = {
        'x', 'y', 'centroid_x', 'centroid_y',
        'geoid', 'semmcd', 'large_hh_city', 'small_hh_city',
        'nodeid_walk', 'nodeid_drv',
    }
    if var in skip_id_geog:
        return True

    # === SKIP: ID columns and one-hot dummies ===
    if var.endswith('_id') or '_id_is_' in var_lower or '_is_' in var_lower:
        return True

    # === SKIP: Target variables ===
    if 'sqft_price' in var_lower:
        return True

    # === SKIP: Price-derived and self-referential price fields ===
    # The price targets are calculated from market_value. improvement_value is
    # a closely related assessed-value component and can be generated from the
    # predicted price during simulation. The original node residential average
    # includes the focal building's target price; use the leave-one-out version
    # instead. housing_cost is derived from that same leaking average.
    if var in {
        'market_value',
        'improvement_value',
        'impr_value_per_sqft',
        'nodes_walk_residential',
        'nodes_walk_housing_cost',
        'nodes_walk_residential_price_observations',
    }:
        return True

    # === SKIP: Standardized, log-transformed, tract, zone lowercase ===
    skip_prefixes = ['st_', 'b_ln_', 'ln_', 'tract_', 'zone_']
    for prefix in skip_prefixes:
        if var_lower.startswith(prefix):
            return True

    # === SKIP: Building-level b_ prefix (except employment vars) ===
    if var_lower.startswith('b_') and not var_lower.startswith('bldg_'):
        return True

    # === SKIP: is_* flags ===
    if var_lower.startswith('is_'):
        return True

    # === SKIP: One-hot dummies and redundant flags ===
    skip_flags = {
        'mcd_model_quota', 'hu_filter', 'sp_filter', 'gq_building',
        'mean_zonal_hhsize', 'popden', 'jobs_within_30_min',
        'school_district_achievement',
    }
    if var in skip_flags:
        return True

    # === SKIP: TAZ employment ratios (redundant with bldg_empratio) ===
    if var_lower.startswith('taz_empratio_'):
        return True

    # === SKIP: Zones_ variables (keep only logsums, transit_jobs, accessibility) ===
    if var_lower.startswith('zones_'):
        keep_zones = ['logsum', 'transit_jobs', 'a_ln_emp', 'a_ln_retail']
        if not any(pattern in var_lower for pattern in keep_zones):
            return True
        return False

    # === KEEP: Everything else ===
    return False


def _load_data(buildings, valid_vars):
    """Load building data into sparse matrix, excluding unwanted variables."""
    vars_used, mat_list = [], []
    t0 = time.time()

    for i, var in enumerate(tqdm(valid_vars, desc="Loading variables", ncols=80)):
        if _should_skip_var(var) or var not in buildings.columns:
            continue

        s = buildings.to_frame(var).iloc[:, 0]
        s.replace([np.inf, -np.inf], 0, inplace=True)
        s.fillna(0, inplace=True)

        if pd.api.types.is_numeric_dtype(s):
            vars_used.append(var)
            mat_list.append(scipy.sparse.csr_matrix(s.values))

        if i % 50 == 0:
            orca.clear_columns("buildings")

    orca.clear_all()
    mat = scipy.sparse.vstack(mat_list)

    # Save variable list
    varlist_path = Path(REPM_XGB_PATH) / "variables_used.txt"
    with open(varlist_path, 'w') as f:
        f.write('\n'.join(map(str, sorted(vars_used))))

    return vars_used, mat, time.time() - t0


def _get_hedonic_segments(mat, vars_used):
    """Get all unique hedonic_id segments with their metadata."""
    filter_idx = [vars_used.index(c) for c in FILTER_COLS if c in vars_used]
    available_cols = [c for c in FILTER_COLS if c in vars_used]
    # Efficient: slice all filter rows at once and transpose
    filter_mat = mat[filter_idx, :].toarray().T
    df = pd.DataFrame(filter_mat, columns=available_cols)
    df = df[~df["hedonic_id"].isna()].astype({"hedonic_id": int})

    segments = []
    for hid in df["hedonic_id"].unique():
        is_res = hid % 100 in [81, 82, 83, 84]
        prefix, price_col, size_col = ("res_repm", "sqft_price_res", "residential_units") if is_res else \
                                       ("nonres_repm", "sqft_price_nonres", "non_residential_sqft")

        # Apply filters
        subset = df[
            (df["hedonic_id"] == hid) &
            (df[size_col] > 0) &  # sqft > 0
            (df[price_col] > 1) & # per sqft price between $1-$650
            (df[price_col] < 650)
        ]

        segments.append({
            "hedonic_id": hid,
            "is_residential": is_res,
            "prefix": prefix,
            "price_col": price_col,
            "size_col": size_col,
            "sample_size": len(subset),
            "filter_idx": filter_idx,
        })

    return segments


def _prepare_training_data(mat, segment, vars_used):
    """Prepare training data for a specific hedonic segment."""
    # Get filter column indices
    filter_idx = [vars_used.index(c) for c in FILTER_COLS if c in vars_used]
    available_cols = [c for c in FILTER_COLS if c in vars_used]

    # Efficient: slice all filter rows at once and transpose
    filter_mat = mat[filter_idx, :].toarray().T
    df = pd.DataFrame(filter_mat, columns=available_cols)

    # Apply filters
    mask = (
        (df["hedonic_id"] == segment["hedonic_id"]) &
        (df[segment["size_col"]] > 0) &
        (df[segment["price_col"]] > 1) &
        (df[segment["price_col"]] < 650)
    )
    df_filtered = df[mask]

    if len(df_filtered) == 0:
        return None, None, None, 0

    # Get features (exclude filter columns)
    feat_idx = [i for i in range(mat.shape[0]) if i not in filter_idx]
    feat_names = [vars_used[i] for i in feat_idx]
    X = mat.toarray()[:, df_filtered.index].T[:, feat_idx]
    y = np.log1p(df_filtered[segment["price_col"]].values)

    return np.nan_to_num(X, copy=False), y, feat_names, len(df_filtered)


def _get_xgb_params(sample_size):
    """Get XGBoost parameters based on sample size."""
    if sample_size <= 50:
        return {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 10,
        }
    elif sample_size <= 500:
        return {
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.05,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 20,
        }
    else:
        return {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'min_child_weight': 10,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 1.0,
            'reg_lambda': 10.0,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 30,
        }


def _compute_metrics(y_true, y_pred, y_train, y_pred_train):
    """Compute comprehensive performance metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    metrics = {
        # Test metrics (primary)
        'r2_test': float(r2_score(y_true, y_pred)),
        'rmse_test': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae_test': float(mean_absolute_error(y_true, y_pred)),
        'mape_test': float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100),
        # Training metrics (for overfitting check)
        'r2_train': float(r2_score(y_train, y_pred_train)),
        'rmse_train': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        'mae_train': float(mean_absolute_error(y_train, y_pred_train)),
    }

    # For backward compatibility with old model naming
    metrics['r2_val'] = metrics['r2_test']
    metrics['rmse_val'] = metrics['rmse_test']
    metrics['mae_val'] = metrics['mae_test']
    metrics['r2_adj_val'] = metrics['r2_test']  # Simplified

    return metrics


def _train_xgboost_model(mat, segment, vars_used, use_grid_search=False, grid_search_baseline=None):
    """Train model for a hedonic segment (XGBoost or Ridge for small samples)."""
    t0 = time.time()

    # Prepare data
    X_all, y, feat_names, sample_size = _prepare_training_data(mat, segment, vars_used)

    if sample_size == 0:
        return None

    if segment["hedonic_id"] in REGIONAL_DUMMY_HEDONIC_IDS:
        return _train_constant_dummy_model(
            X_all, y, feat_names, segment, t0, "regional_data_center"
        )

    # Very small segments cannot support feature selection, a train/test split,
    # or a fitted hedonic relationship. Keep a simulation-ready constant-price
    # fallback instead of failing the entire estimation run.
    if sample_size < MIN_SAMPLES_FOR_RIDGE:
        fallback_reason = "one_record_segment" if sample_size == 1 else "insufficient_records"
        return _train_constant_dummy_model(
            X_all, y, feat_names, segment, t0, fallback_reason
        )

    # Feature selection: remove low variance and correlated features
    X_all, feat_names = _remove_low_variance(X_all, feat_names)
    X_all, feat_names = _remove_correlated(X_all, feat_names)

    # Use Ridge regression for small samples
    use_simple_model = sample_size < SIMPLE_MODEL_THRESHOLD

    if use_simple_model:
        print(f"  Using Ridge for small sample ({sample_size} < {SIMPLE_MODEL_THRESHOLD})")
        return _train_ridge_model(X_all, y, feat_names, sample_size, segment, t0)
    else:
        return _train_xgboost_model_impl(X_all, y, feat_names, sample_size, segment,
                                         use_grid_search, grid_search_baseline, t0)


def _train_constant_dummy_model(X, y, feat_names, segment, start_time, fallback_reason):
    """Create a simulation-ready constant-price model without validation."""
    # DummyRegressor still validates feature shape at prediction time. Use a
    # stable existing building column rather than preserving every candidate.
    feature_index = feat_names.index("market_value") if "market_value" in feat_names else 0
    feature_names = [feat_names[feature_index]]
    X = X[:, [feature_index]]

    model = DummyRegressor(strategy="mean")
    model.fit(X, y)
    prediction = model.predict(X)
    unavailable = float("nan")
    metrics = {
        "r2_test": unavailable,
        "rmse_test": unavailable,
        "mae_test": unavailable,
        "mape_test": unavailable,
        "r2_train": unavailable,
        "rmse_train": unavailable,
        "mae_train": unavailable,
        "r2_val": unavailable,
        "r2_adj_val": unavailable,
        "rmse_val": unavailable,
        "mae_val": unavailable,
        "sample_size": len(y),
        "n_features": len(feature_names),
    }
    print(f"  Using constant-price Dummy fallback ({len(y)} valid records; {fallback_reason})")
    return {
        "model": model,
        "feature_names": feature_names,
        "feature_importance": {feature_names[0]: 0.0},
        "metrics": metrics,
        "segment": segment,
        "y_test": y,
        "y_pred_test": prediction,
        "training_time": time.time() - start_time,
        "using_grid_search": False,
        "using_gs_baseline": False,
        "model_type": "dummy",
        "fallback_reason": fallback_reason,
    }


def _select_features_by_importance(X, y, feat_names, sample_size, model_type='ridge'):
    """Select top features based on importance scores to reduce overfitting."""
    max_feats = get_max_features(sample_size)

    if max_feats is None or len(feat_names) <= max_feats:
        return X, feat_names, False

    print(f"  Selecting top {max_feats}/{len(feat_names)} features (sample_size={sample_size})")

    # For Ridge: use mutual information as proxy
    # For XGBoost: use a quick preliminary model to get importance
    if model_type == 'ridge':
        # For Ridge, use mutual information
        mi = mutual_info_regression(X, y, random_state=42)
        top_indices = np.argsort(mi)[-max_feats:]
    else:
        # For XGBoost, train a quick model to get feature importance
        from sklearn.model_selection import train_test_split
        X_tmp, _, y_tmp, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        tmp_model = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42, n_jobs=-1)
        tmp_model.fit(X_tmp, y_tmp, verbose=False)
        importance = tmp_model.feature_importances_
        top_indices = np.argsort(importance)[-max_feats:]

    X_selected = X[:, top_indices]
    feat_names_selected = [feat_names[i] for i in top_indices]

    return X_selected, feat_names_selected, True


def _remove_correlated_features(X, feat_names, threshold=0.95):
    """Remove highly correlated features to prevent multicollinearity."""
    if X.shape[1] <= 1:
        return X, feat_names, False

    # Compute correlation matrix
    corr_matrix = np.abs(np.corrcoef(X.T))

    # Find features to remove (keep first from each correlated pair)
    to_remove = set()
    n_features = X.shape[1]

    for i in range(n_features):
        if i in to_remove:
            continue
        for j in range(i + 1, n_features):
            if j in to_remove:
                continue
            if corr_matrix[i, j] > threshold:
                to_remove.add(j)

    if to_remove:
        keep_indices = [i for i in range(n_features) if i not in to_remove]
        X_filtered = X[:, keep_indices]
        feat_names_filtered = [feat_names[i] for i in keep_indices]
        print(f"    Removed {len(to_remove)} correlated features (threshold={threshold})")
        return X_filtered, feat_names_filtered, True

    return X, feat_names, False


def _train_ridge_model(X_all, y, feat_names, sample_size, segment, start_time):
    """Train Ridge regression model with dummy mean fallback for small samples."""
    # Feature selection based on sample size
    X_selected, feat_names_selected, selected = _select_features_by_importance(
        X_all, y, feat_names, sample_size, model_type='ridge'
    )

    if selected:
        X_all = X_selected
        feat_names = feat_names_selected

    # Remove correlated features to prevent multicollinearity
    X_no_corr, feat_names_no_corr, corr_removed = _remove_correlated_features(
        X_all, feat_names, threshold=0.95
    )

    if corr_removed:
        X_all = X_no_corr
        feat_names = feat_names_no_corr

    # Use smaller test size for small samples to have more training data
    test_size = max(0.1, min(0.3, 30.0 / sample_size))  # Adaptive test size

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=test_size, random_state=42
    )

    # Use stronger regularization for smaller samples
    alpha = max(1.0, 100.0 / sample_size)

    # Train both Ridge and Dummy models
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_train, y_train)

    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train)

    # Evaluate both models
    ridge_train_pred = ridge.predict(X_train)
    ridge_test_pred = ridge.predict(X_test)
    ridge_r2_train = r2_score(y_train, ridge_train_pred)
    ridge_r2_test = r2_score(y_test, ridge_test_pred)

    dummy_test_pred = dummy.predict(X_test)
    dummy_r2_test = r2_score(y_test, dummy_test_pred)

    # Hybrid: Choose the better model (with small tolerance for Ridge)
    # Use Ridge if it's within 0.05 of Dummy, to prefer feature-based models
    tolerance = 0.05
    if ridge_r2_test >= dummy_r2_test - tolerance:
        # Ridge wins or is close enough
        model = ridge
        model_type = 'ridge'
        y_pred_train = ridge_train_pred
        y_pred_test = ridge_test_pred
        print(f"  R²: train={ridge_r2_train:.4f}, test={ridge_r2_test:.4f} (Ridge α={alpha:.2f}, test={test_size:.0%})")
    else:
        # Dummy mean predictor wins
        model = dummy
        model_type = 'dummy'
        y_pred_train = dummy.predict(X_train)
        y_pred_test = dummy_test_pred
        print(f"  R²: train={r2_score(y_train, y_pred_train):.4f}, test={dummy_r2_test:.4f} (DUMMY mean - Ridge was {ridge_r2_test:.4f})")

    # Compute metrics
    metrics = _compute_metrics(y_test, y_pred_test, y_train, y_pred_train)
    metrics['sample_size'] = sample_size
    metrics['n_features'] = len(feat_names)

    # Feature importance (only meaningful for Ridge)
    if model_type == 'ridge':
        importance = np.abs(model.coef_)
        total_importance = importance.sum() or 1
        importance_norm = {feat_names[i]: float(importance[i] / total_importance) for i in range(len(feat_names))}
    else:
        # Dummy has no feature importance
        importance_norm = {feat: 0.0 for feat in feat_names}

    return {
        'model': model,
        'feature_names': feat_names,
        'feature_importance': importance_norm,
        'metrics': metrics,
        'segment': segment,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
        'training_time': time.time() - start_time,
        'using_grid_search': False,
        'using_gs_baseline': False,
        'model_type': model_type,
    }


def _train_xgboost_model_impl(X_all, y, feat_names, sample_size, segment,
                               use_grid_search, grid_search_baseline, start_time):
    """Train XGBoost model (internal implementation)."""
    model_name = f"{segment['prefix']}{segment['hedonic_id']}"
    gs_path = Path(REPM_XGB_PATH) / f"grid_search_{model_name}.yaml"
    using_gs = False
    using_gs_baseline = False

    # Get default parameters
    params = _get_xgb_params(sample_size)

    # Load existing grid search params if available and not doing new grid search
    if not use_grid_search and gs_path.exists():
        with open(gs_path, 'r') as f:
            gs_results = yaml.load(f, Loader=yaml.FullLoader)
            best_params = gs_results.get('best_params', {})
            if best_params:
                params.update(best_params)
                params.pop('early_stopping_rounds', None)
                using_gs = True

    # Grid search or standard training
    if use_grid_search:
        # Simplified grid search: 2^4 = 16 combinations × 3-fold CV = ~48 fits (~10x normal)
        param_grid = {
            'n_estimators': [200, 500],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.7, 0.9],
        }

        # Load baseline for refined search (supports "auto", directory, or file path)
        baseline_path = None
        if grid_search_baseline:
            if grid_search_baseline == "auto":
                baseline_path = gs_path if gs_path.exists() else None
            elif Path(grid_search_baseline).is_dir():
                baseline_path = Path(grid_search_baseline) / f"grid_search_{model_name}.yaml"
                baseline_path = baseline_path if baseline_path.exists() else None
            elif Path(grid_search_baseline).exists():
                baseline_path = Path(grid_search_baseline)

        if baseline_path:
            with open(baseline_path, 'r') as f:
                baseline = yaml.load(f, Loader=yaml.FullLoader)
                best = baseline.get('best_params', {})
                if best:
                    param_grid = {
                        'n_estimators': [
                            max(100, int(best.get('n_estimators', 200) - 100)),
                            int(best.get('n_estimators', 200) + 100)
                        ][:2],
                        'max_depth': [
                            max(3, int(best.get('max_depth', 4) - 1)),
                            min(8, int(best.get('max_depth', 4) + 1))
                        ],
                        'learning_rate': [
                            round(best.get('learning_rate', 0.05) * 0.8, 3),
                            round(best.get('learning_rate', 0.05) * 1.2, 3)
                        ],
                        'subsample': [
                            round(max(0.6, best.get('subsample', 0.8) - 0.1), 2),
                            round(min(0.95, best.get('subsample', 0.8) + 0.1), 2)
                        ],
                    }
                    using_gs_baseline = True

        model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        gs = GridSearchCV(
            model, param_grid, cv=3, scoring='r2', verbose=0, n_jobs=-1,
            return_train_score=True
        )
        gs.fit(X_all, y)

        model = gs.best_estimator_
        best_params = gs.best_params_
        cv_results = gs.cv_results_

        # Save grid search results as baseline
        grid_search_results = {
            'best_params': best_params,
            'best_score': float(gs.best_score_),
            'best_cv_mean': float(cv_results['mean_test_score'][gs.best_index_]),
            'best_cv_std': float(cv_results['std_test_score'][gs.best_index_]),
            'model_name': model_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'sample_size': sample_size,
            'n_features': len(feat_names),
            'all_results': {
                'params': cv_results['params'],
                'mean_test_score': cv_results['mean_test_score'].tolist(),
                'std_test_score': cv_results['std_test_score'].tolist(),
            }
        }

        # Save to file
        with open(gs_path, 'w') as f:
            yaml.dump(grid_search_results, f, default_flow_style=False, sort_keys=False)

        # Update params with grid search best params for final training
        params.update(best_params)

    # Feature selection based on sample size (only for limited samples)
    X_selected, feat_names_selected, selected = _select_features_by_importance(
        X_all, y, feat_names, sample_size, model_type='xgboost'
    )
    if selected:
        X_all = X_selected
        feat_names = feat_names_selected

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42
    )

    # Train model with final parameters (either from grid search, loaded GS, or default)
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Compute metrics
    metrics = _compute_metrics(y_test, y_pred_test, y_train, y_pred_train)
    metrics['sample_size'] = sample_size
    metrics['n_features'] = len(feat_names)

    # Feature importance (normalized)
    importance = model.feature_importances_
    total_importance = importance.sum() or 1
    importance_norm = {feat_names[i]: float(importance[i] / total_importance) for i in range(len(feat_names))}

    return {
        'model': model,
        'feature_names': feat_names,
        'feature_importance': importance_norm,
        'metrics': metrics,
        'segment': segment,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
        'training_time': time.time() - start_time,
        'using_grid_search': using_gs,
        'using_gs_baseline': using_gs_baseline,
        'model_type': 'xgboost',
    }


def _save_model(model_artifacts, model_name):
    """Save model (XGBoost or Ridge) and metadata using modern practices."""
    model_dir = Path(REPM_XGB_PATH) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_type = model_artifacts.get('model_type', 'xgboost')

    # 1. Save model (different formats for XGBoost vs sklearn)
    if model_type == 'xgboost':
        model_path = model_dir / "xgb_model.json"
        model_artifacts['model'].save_model(str(model_path))
    else:  # Ridge/sklearn model
        model_path = model_dir / "sklearn_model.pkl"
        joblib.dump(model_artifacts['model'], model_path)

    # 2. Save metadata (without the actual model object)
    metadata = {
        'model_name': model_name,
        'model_type': model_type,
        'hedonic_id': int(model_artifacts['segment']['hedonic_id']),
        'is_residential': bool(model_artifacts['segment']['is_residential']),
        'price_col': model_artifacts['segment']['price_col'],
        'size_col': model_artifacts['segment']['size_col'],
        'feature_names': model_artifacts['feature_names'],
        'n_features': len(model_artifacts['feature_names']),
        'feature_importance': model_artifacts['feature_importance'],
        'metrics': model_artifacts['metrics'],
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    if model_artifacts.get('fallback_reason'):
        metadata['fallback_reason'] = model_artifacts['fallback_reason']

    def format_metric(value):
        return f"{value:.4f}" if np.isfinite(value) else "not_available"

    metadata_path = model_dir / "metadata.pkl"
    joblib.dump(metadata, metadata_path)

    # 3. Save human-readable summary
    summary = {
        'model_name': model_name,
        'model_type': model_type,
        'hedonic_id': metadata['hedonic_id'],
        'type': 'residential' if metadata['is_residential'] else 'non-residential',
        'sample_size': metadata['metrics']['sample_size'],
        'n_features': metadata['n_features'],
        'performance': {
            'r2_train': format_metric(metadata['metrics']['r2_train']),
            'r2_val': format_metric(metadata['metrics']['r2_val']),
            'r2_adj_val': format_metric(metadata['metrics']['r2_adj_val']),
            'rmse_val': format_metric(metadata['metrics']['rmse_val']),
            'mae_val': format_metric(metadata['metrics']['mae_val']),
        },
        'top_features': dict(
            sorted(
                metadata['feature_importance'].items(),
                key=lambda item: item[1],
                reverse=True,
            )[:10]
        ),
    }
    if metadata.get('fallback_reason'):
        summary['fallback_reason'] = metadata['fallback_reason']

    summary_path = model_dir / "summary.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    # 4. Save minimal YAML config for backward compatibility
    config = {
        'model_type': model_type,
        'model_path': str(model_dir),
        'hedonic_id': metadata['hedonic_id'],
        'target_variable': f"np.log1p({metadata['price_col']})",
        'ytransform': 'np.exp',
        'features': metadata['feature_names'],
        'fit_rsquared': metadata['metrics']['r2_val'],
        'sample_size': metadata['metrics']['sample_size'],
    }

    config_path = model_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _remove_low_variance(X, names, threshold=None):
    """Remove features with near-zero variance."""
    if threshold is None:
        threshold = VARIANCE_THRESHOLD
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    mask = selector.get_support()
    return X[:, mask], [n for n, m in zip(names, mask) if m]


def _remove_correlated(X, names, threshold=None):
    """Remove highly correlated features."""
    if threshold is None:
        threshold = CORRELATION_THRESHOLD
    corr = np.corrcoef(X.T)
    upper = np.triu(np.ones_like(corr), k=1).astype(bool)
    high_corr = np.where((np.abs(corr) > threshold) & upper)

    to_remove = set()
    for i, j in zip(*high_corr):
        to_remove.add(j) if names[i] <= names[j] else to_remove.add(i)

    keep_idx = [i for i in range(len(names)) if i not in to_remove]
    return X[:, keep_idx], [names[i] for i in keep_idx]


def run_repm_training():
    """Run REPM XGBoost training pipeline with modern best practices."""
    overall_start = time.time()

    # Header
    print("\n" + "="*80)
    print("REPM XGBoost Training".center(80))
    print("="*80)
    print(f"Output:     {REPM_XGB_PATH}")
    print(f"Grid Search:{' Yes' if USE_GRID_SEARCH else ' No'}")
    print(f"Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    Path(REPM_XGB_PATH).mkdir(parents=True, exist_ok=True)

    # Stage 1: Load data
    print("[1/4] Loading data and building networks...")
    orca.add_injectable("repm_estimation_only", True)
    if not orca.is_injectable("data_out_dir"):
        orca.add_injectable("data_out_dir", REPM_XGB_PATH)
    import models
    orca.run(["build_networks"])
    orca.run(["neighborhood_vars"])

    buildings = orca.get_table("buildings")
    vars_used, mat, load_time = _load_data(buildings, buildings.columns)
    print(f"       Loaded {len(vars_used)} variables in {load_time:.1f}s\n")

    # Stage 2: Identify segments
    print("[2/4] Identifying hedonic segments...")
    segments = _get_hedonic_segments(mat, vars_used)
    n_res = sum(1 for s in segments if s['is_residential'])
    n_nonres = len(segments) - n_res
    print(f"       Found {len(segments)} segments ({n_res} residential, {n_nonres} non-residential)\n")

    # Stage 3: Train models
    print("[3/4] Training models...")
    print("-"*80)
    print(f"{'Model':<18} {'Samples':<10} {'Features':<10} {'R²':<8} {'RMSE':<8} {'Time':<9}")
    if USE_GRID_SEARCH:
        marker_legend = "Markers: R = Ridge, D = Dummy (mean), + = refined GS, # = full GS"
    else:
        marker_legend = "Markers: R = Ridge, D = Dummy (mean), * = using GS, (blank) = default"
    print("-"*80 + " " + marker_legend)

    results = {}
    failed_segments = []

    for i, segment in enumerate(segments, 1):
        model_name = f"{segment['prefix']}{segment['hedonic_id']}"

        # Train model
        artifacts = _train_xgboost_model(mat, segment, vars_used, USE_GRID_SEARCH, GRID_SEARCH_BASELINE)

        if artifacts is None:
            failed_segments.append(model_name)
            print(f"{model_name:<18} {'SKIPPED':<54}")
            continue

        # Save model
        _save_model(artifacts, model_name)

        # Record results
        metrics = artifacts['metrics']
        model_type = artifacts.get('model_type', 'xgboost')
        results[model_name] = {
            'status': 'success',
            'hedonic_id': int(segment['hedonic_id']),
            'is_residential': bool(segment['is_residential']),
            'sample_size': int(metrics['sample_size']),
            'n_features': int(metrics['n_features']),
            'r2_train': float(metrics['r2_train']),
            'r2_val': float(metrics['r2_val']),
            'rmse_val': float(metrics['rmse_val']),
            'mae_val': float(metrics['mae_val']),
            'training_time': float(artifacts['training_time']),
            'using_grid_search': bool(artifacts['using_grid_search']),
            'using_gs_baseline': bool(artifacts.get('using_gs_baseline', False)),
            'model_type': model_type,
        }

        # Print progress line with markers:
        # 'R' = Ridge (small sample), ' ' = default XGBoost, '*' = using GS params,
        # '+' = GS with baseline, '#' = full GS
        if model_type == 'ridge':
            gs_marker = 'R'
        elif model_type == 'dummy':
            gs_marker = 'D'
        elif USE_GRID_SEARCH:
            if artifacts.get('using_gs_baseline', False):
                gs_marker = '+'
            else:
                gs_marker = '#'
        else:
            gs_marker = '*' if artifacts['using_grid_search'] else ' '

        print(f"{model_name:<18} {metrics['sample_size']:<10} {metrics['n_features']:<10} "
              f"{metrics['r2_val']:<8.4f} {metrics['rmse_val']:<8.4f} "
              f"{artifacts['training_time']:<8.1f}{gs_marker}")

    print("-"*80)

    # Stage 4: Summary
    print("\n[4/4] Saving summary...")

    # Save training summary
    training_time_total = time.time() - overall_start
    n_using_gs = sum(1 for r in results.values() if r.get('using_grid_search', False))
    n_using_gs_baseline = sum(1 for r in results.values() if r.get('using_gs_baseline', False))
    n_xgboost = sum(1 for r in results.values() if r.get('model_type', 'xgboost') == 'xgboost')
    n_ridge = sum(1 for r in results.values() if r.get('model_type', 'xgboost') == 'ridge')
    n_dummy = sum(1 for r in results.values() if r.get('model_type', 'xgboost') == 'dummy')

    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'training_time_seconds': round(training_time_total, 2),
        'grid_search_used': USE_GRID_SEARCH,
        'grid_search_baseline': GRID_SEARCH_BASELINE,
        'using_grid_search_results': n_using_gs,
        'using_grid_search_baseline': n_using_gs_baseline,
        'n_ridge_models': n_ridge,
        'n_xgboost_models': n_xgboost,
        'total_segments': len(segments),
        'successful': len(results),
        'failed': len(failed_segments),
        'n_residential': n_res,
        'n_non_residential': n_nonres,
        'n_variables': len(vars_used),
        'results': results,
        'failed_segments': failed_segments,
    }

    summary_path = Path(REPM_XGB_PATH) / "training_summary.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY".center(80))
    print("="*80)

    print(f"\nCompleted: {len(results)}/{len(segments)} models trained")
    if n_ridge > 0 or n_dummy > 0:
        model_types_str = f"{n_xgboost} XGBoost (100+ samples)"
        if n_ridge > 0:
            model_types_str += f", {n_ridge} Ridge"
        if n_dummy > 0:
            model_types_str += f", {n_dummy} Dummy (mean fallback)"
        print(f"Model Types: {model_types_str}")
    if USE_GRID_SEARCH and n_using_gs_baseline > 0:
        print(f"Grid Search: {n_using_gs_baseline} with baseline (+), {len(results) - n_using_gs_baseline} full (#)")
    elif not USE_GRID_SEARCH and n_using_gs > 0:
        print(f"Using GS: {n_using_gs} models with grid search parameters (*)")
    if failed_segments:
        print(f"Failed:    {len(failed_segments)} models ({', '.join(failed_segments[:5])}{'...' if len(failed_segments) > 5 else ''})")

    if results:
        res_results = [r for r in results.values() if r['is_residential']]
        nonres_results = [r for r in results.values() if not r['is_residential']]

        if res_results:
            r2_res = [r['r2_val'] for r in res_results if np.isfinite(r['r2_val'])]
            print(f"\nResidential Models (n={len(res_results)}):")
            if r2_res:
                print(f"  R²:     {np.mean(r2_res):.4f} ± {np.std(r2_res):.4f}  [{np.min(r2_res):.4f}, {np.max(r2_res):.4f}]")
            else:
                print("  R²:     not available (no validated models)")

        if nonres_results:
            r2_nonres = [r['r2_val'] for r in nonres_results if np.isfinite(r['r2_val'])]
            print(f"\nNon-Residential Models (n={len(nonres_results)}):")
            if r2_nonres:
                print(f"  R²:     {np.mean(r2_nonres):.4f} ± {np.std(r2_nonres):.4f}  [{np.min(r2_nonres):.4f}, {np.max(r2_nonres):.4f}]")
            else:
                print("  R²:     not available (no validated models)")

        all_r2 = [r['r2_val'] for r in results.values() if np.isfinite(r['r2_val'])]
        print(f"\nAll Models:")
        if all_r2:
            print(f"  R²:     {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
        else:
            print("  R²:     not available (no validated models)")

        total_time = sum(r['training_time'] for r in results.values())
        print(f"  Time:   {total_time:.1f}s training, {training_time_total:.1f}s total")

    print(f"\nOutput:   {REPM_XGB_PATH}")
    print(f"Summary:  {summary_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_repm_training()
