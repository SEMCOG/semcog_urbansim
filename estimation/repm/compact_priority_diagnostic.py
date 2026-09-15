"""Fold-safe compact XGBoost diagnostics for selected non-residential REPMs."""
import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split

from estimation.repm import neighborhood_vars_cache as nvcache
from estimation.repm import xgb_training as training


EASTERN = ZoneInfo("America/Detroit")
OUTPUT_ROOT = Path("/home/da/RDF2055/d_drive/estimation/REPM")
PRIORITY_SEGMENTS = {
    11531: {"market": "manufacturing", "sizes": [25, 30, 40]},
    94: {"market": "death_care", "sizes": [30, 40, 50]},
    14731: {"market": "manufacturing", "sizes": [30, 40, 50]},
    9323: {"market": "office", "sizes": [40, 50, 60]},
}
STANDARD_SEGMENTS = {
    32: {"market": "wholesale", "sizes": [30, 40, 50]},
    9321: {"market": "retail", "sizes": [40, 50, 60]},
    9331: {"market": "manufacturing", "sizes": [30, 40, 50]},
    14721: {"market": "retail", "sizes": [40, 50, 60]},
    16123: {"market": "office", "sizes": [40, 50, 60]},
    14723: {"market": "office", "sizes": [40, 50, 60]},
    533: {"market": "warehouse", "sizes": [40, 50, 60]},
    11523: {"market": "office", "sizes": [30, 40, 50]},
    14733: {"market": "warehouse", "sizes": [30, 40, 50]},
    9333: {"market": "warehouse", "sizes": [30, 40, 50]},
    11533: {"market": "warehouse", "sizes": [30, 40, 50]},
}
FOLDS = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)


def _category_and_reason(name, market):
    """Apply the shared theory/data screen before supervised selection."""
    low = name.lower()
    demographic_tokens = (
        "race", "income", "hhs", "hhsize", "recent_mover", "vehicles",
        "zero_veh", "wfh", "commute", "years_at_residence", "age_of_head",
    )
    if any(token in low for token in demographic_tokens):
        return "excluded_proxy", "Demographic or travel-behavior proxy; not a property-market attribute.", False
    if "_bike_" in low or low.startswith("bike_"):
        return "excluded_bike", "Bike indicators are outside the approved regional specification.", False
    if low == "nodes_walk_max_industrial_far":
        return "excluded_zoning_far", "Network aggregate of zoning FAR; excluded because regional non-residential FAR is not reliable.", False
    if low.endswith("_excl_self"):
        return "peer_market", "Leave-one-out local price context; focal price is excluded.", True
    if low in {"land_to_impr_ratio", "nonres_vacancy_rate", "vacant_job_spaces", "job_spaces"}:
        return "building_market", "Building/site market condition available during simulation.", True
    if low.startswith(("market_zone_", "nodes_walk_sector", "nodes_walk_jobs", "nodes_drv_log_sum")):
        return "employment_market", "Employment-market scale or sector composition.", True
    if "industrial" in low or "nonresidential" in low:
        return "industrial_context", "Industrial activity or non-residential market context.", True
    if low.startswith(("parcel_", "zoning_future_use_")):
        return "site_land_use", "Parcel/site structure or land-use context.", True
    if low in {"year_built", "building_age", "stories", "building_sqft", "non_residential_sqft", "land_area", "parcel_sqft", "sqft_per_unit"} or low.startswith("building_age_"):
        return "physical", "Physical building or site characteristic used in hedonic practice.", True
    amenity_tokens = ("school", "childcare", "grocery", "library", "park", "hospital", "health", "pharm", "urgent")
    if any(token in low for token in amenity_tokens):
        if market in {"manufacturing", "wholesale", "warehouse"}:
            return "excluded_amenity", "Sector-irrelevant amenity proxy for industrial/production space.", False
        return "access_amenity", "Locational amenity/access measure; retained for office or institutional use.", True
    if low.startswith(("drv_", "walk_", "nodes_walk_", "nodes_drv_", "fixed_route_bus", "passenger_")):
        if market in {"manufacturing", "wholesale", "warehouse"} and not any(
            token in low for token in ("jobs", "employment", "industrial", "transit")
        ):
            return "excluded_access_proxy", "Non-employment accessibility proxy outside the industrial core.", False
        return "accessibility", "Network-based locational accessibility measure.", True
    if low.startswith(("crime_", "is_")):
        return "location_context", "Local context measure available at simulation time.", True
    return "excluded_unclassified", "No documented physical, market, land-use, or accessibility rationale.", False


def _preference(name):
    """Prefer interpretable physical/site variables when a correlation pair ties."""
    low = name.lower()
    if low in {"year_built", "building_age", "stories", "building_sqft", "non_residential_sqft", "land_area", "parcel_sqft"}:
        return 0
    if low.startswith(("parcel_", "market_zone_", "zoning_future_use_")):
        return 1
    if low.endswith("_excl_self"):
        return 2
    if low.startswith(("nodes_", "drv_", "walk_")):
        return 3
    return 4


def _remove_train_redundancy(X, names, threshold=0.95):
    selector = VarianceThreshold(threshold=1e-6)
    X = selector.fit_transform(X)
    names = [name for name, keep in zip(names, selector.get_support()) if keep]
    if len(names) < 2:
        return X, names
    corr = np.nan_to_num(np.abs(np.corrcoef(X.T)), nan=0.0)
    remove = set()
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            if left in remove or right in remove or corr[left, right] <= threshold:
                continue
            left_key = (_preference(names[left]), names[left])
            right_key = (_preference(names[right]), names[right])
            remove.add(right if left_key <= right_key else left)
    keep = [i for i in range(len(names)) if i not in remove]
    return X[:, keep], [names[i] for i in keep]


def _fit_xgb(X_train, y_train, X_test, fold_no):
    X_fit, X_es, y_fit, y_es = train_test_split(
        X_train, y_train, test_size=0.2, random_state=fold_no
    )
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=2,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=10.0,
        early_stopping_rounds=25,
        random_state=42 + fold_no,
        n_jobs=2,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    return model, model.predict(X_train), model.predict(X_test)


def _evaluate_segment(hid, segment_config, mat, vars_used, out_dir):
    segment = next(item for item in training._get_hedonic_segments(mat, vars_used) if item["hedonic_id"] == hid)
    X, y, names, sample_size = training._prepare_training_data(mat, segment, vars_used)
    review_rows = []
    eligible_indices = []
    for index, name in enumerate(names):
        category, reason, eligible = _category_and_reason(name, segment_config["market"])
        review_rows.append({"hedonic_id": hid, "market": segment_config["market"], "variable": name,
                            "category": category, "reason": reason, "eligible": eligible})
        if eligible:
            eligible_indices.append(index)
    X = X[:, eligible_indices]
    names = [names[index] for index in eligible_indices]
    specs = {"reviewed_broad": None}
    specs.update({f"compact_{size}": size for size in segment_config["sizes"]})
    rows = []
    for fold_no, (train_idx, test_idx) in enumerate(FOLDS.split(X), 1):
        X_train_base, X_test_base = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train, fold_names = _remove_train_redundancy(X_train_base, names)
        selected_base = [names.index(name) for name in fold_names]
        X_test = X_test_base[:, selected_base]
        for spec_name, size in specs.items():
            if size is None:
                X_train_spec, X_test_spec, selected_names = X_train, X_test, fold_names
            else:
                selector = SelectKBest(mutual_info_regression, k=min(size, len(fold_names)))
                X_train_spec = selector.fit_transform(X_train, y_train)
                X_test_spec = selector.transform(X_test)
                selected_names = [fold_names[i] for i in selector.get_support(indices=True)]
            model, pred_train, pred_test = _fit_xgb(X_train_spec, y_train, X_test_spec, fold_no)
            rows.append({
                "hedonic_id": hid, "market": segment_config["market"], "specification": spec_name,
                "fold": fold_no, "n_train": len(train_idx), "n_test": len(test_idx),
                "n_features": len(selected_names), "r2_train": float(r2_score(y_train, pred_train)),
                "r2_test": float(r2_score(y_test, pred_test)),
                "rmse_test_log": float(np.sqrt(mean_squared_error(y_test, pred_test))),
                "mae_test_log": float(mean_absolute_error(y_test, pred_test)),
                "xgb_best_iteration": int(model.best_iteration or 0),
                "selected_features": ";".join(selected_names),
            })
    folds = pd.DataFrame(rows)
    selection = Counter(
        name for names in folds.loc[folds.specification != "reviewed_broad", "selected_features"]
        for name in names.split(";") if name
    )
    review = pd.DataFrame(review_rows)
    review["compact_selection_count"] = review.variable.map(selection).fillna(0).astype(int)
    return sample_size, review, folds


def main(batch):
    started_at = datetime.now(EASTERN)
    started = time.time()
    segments = {"priority": PRIORITY_SEGMENTS, "standard": STANDARD_SEGMENTS}[batch]
    out_dir = OUTPUT_ROOT / f"repm_{started_at.strftime('%Y%m%d_%H%M%S')}_compact_{batch}"
    out_dir.mkdir(parents=True, exist_ok=False)
    if not nvcache.is_valid():
        raise RuntimeError("A current REPM training-matrix cache is required.")
    vars_used, mat = nvcache.load()
    all_reviews, all_folds, segment_rows = [], [], []
    for hid, config in segments.items():
        sample_size, review, folds = _evaluate_segment(hid, config, mat, vars_used, out_dir)
        all_reviews.append(review)
        all_folds.append(folds)
        segment_rows.append({"hedonic_id": hid, "market": config["market"], "sample_size": sample_size,
                             "eligible_candidates": int(review.eligible.sum()), "compact_sizes": config["sizes"]})
        print(f"Completed {hid}: {sample_size} records", flush=True)
    review = pd.concat(all_reviews, ignore_index=True)
    folds = pd.concat(all_folds, ignore_index=True)
    summary = (folds.groupby(["hedonic_id", "market", "specification"], as_index=False)
               .agg(folds=("fold", "size"), n_features=("n_features", "mean"),
                    r2_test_mean=("r2_test", "mean"), r2_test_std=("r2_test", "std"),
                    rmse_test_log_mean=("rmse_test_log", "mean"), mae_test_log_mean=("mae_test_log", "mean"),
                    r2_train_mean=("r2_train", "mean"), best_iteration_mean=("xgb_best_iteration", "mean")))
    summary["n_features"] = summary.n_features.round().astype(int)
    review.to_csv(out_dir / "candidate_variable_review.csv", index=False)
    folds.to_csv(out_dir / "compact_priority_fold_results.csv", index=False)
    summary.to_csv(out_dir / "compact_priority_summary.csv", index=False)
    report = {
        "created_at": datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "purpose": f"Fold-safe compact XGBoost comparison for {batch} non-residential REPMs",
        "input_matrix_cache": str(nvcache.CACHE_DATA_PATH),
        "cross_validation": "RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)",
        "selection": "Variance/correlation screening and mutual-information selection fit on each outer training fold only.",
        "literature_basis": {
            "industrial": "https://www.pbl.nl/en/publications/a-hedonic-price-analysis-of-the-value-of-industrial-sites",
            "commercial_hedonics": "https://ec.europa.eu/eurostat/documents/7870049/8545612/KS-FT-16-001-EN-N.pdf",
            "validation": "https://scikit-learn.org/stable/common_pitfalls.html",
        },
        "segments": segment_rows,
        "summary": summary.to_dict(orient="records"),
        "total_seconds": time.time() - started,
    }
    (out_dir / "compact_priority_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Output: {out_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", choices=["priority", "standard"], default="priority")
    main(parser.parse_args().batch)
