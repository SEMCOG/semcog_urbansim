"""
Demolition scoring model calibration.

Trains logistic regression models (residential and non-residential) to score
buildings by demolition probability.  Outputs coefficients and normalization
parameters to configs/demolition_model.yaml, which scored_demolition_events
reads at simulation time.

Training design
---------------
Period 1 — 2017-2019  (base stock: buildings standing end of 2016)
  Train set.  Avoids the 2014-2016 Detroit blight-program spike.

Period 2 — 2021-2024  (base stock: terra_buildings_2020 view)
  Validation set.  Post-pandemic steady state, most representative of
  the future demolition regime we want to model.

Features
--------
building_age        : years since year_built (2020 reference)
impr_value_per_sqft : parcel building_value / total building sqft
land_to_impr_ratio  : parcel land_value / building_value
is_exempt           : parcel tax-exempt status (proxy for delinquency/public ownership)
is_wayne            : Wayne County indicator (large_area_id = 5)

Run from semcog_urbansim/ directory:
    micromamba activate forecast
    python demolition_calibration.py
"""

import os
import yaml
import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from urbansim.utils import misc

DB_CONN = "postgresql://USER:PSWD@SERVER:PORT/DBNAME"
CONFIG_PATH = os.path.join(misc.configs_dir(), "demolition_model.yaml")

# Wayne County large_area_id
WAYNE_LA_ID = 5

# Reference year for building age calculation
AGE_REF_YEAR = 2025

RES_FEATURES    = ["building_age", "impr_value_per_sqft", "land_to_impr_ratio",
                   "is_exempt", "is_wayne"]
NONRES_FEATURES = ["building_age", "impr_value_per_sqft", "land_to_impr_ratio",
                   "is_exempt", "is_wayne"]

# ─────────────────────────────────────────────────────────────────────────────
# SQL
# ─────────────────────────────────────────────────────────────────────────────

QUERY_PERIOD1 = """
SELECT
    b.building_id,
    b.parcel_id,
    b.city_id,
    b.build_type                        AS building_type_id,
    b.year_built,
    b.housing_units                     AS residential_units,
    b.res_sqft,
    b.nonres_sqft                       AS non_residential_sqft,
    CASE
        WHEN b.demolished IS NOT NULL
             AND b.demolished >= '2017-01-01'
             AND b.demolished <  '2020-01-01'
        THEN 1 ELSE 0
    END                                 AS demolished,
    COALESCE(tc.large_area_id, 0)       AS large_area_id,
    COALESCE(ua.building_value, 0)      AS bldgimprval,
    COALESCE(ua.land_value,    0)       AS landvalue,
    COALESCE(ua.exempt,        0)       AS exempt
FROM creator.terra_buildings b
LEFT JOIN creator.urbansim_assessing  ua ON b.parcel_id = ua.parcel_id
LEFT JOIN creator.terminus_city_id   tc ON b.city_id   = tc.city_id
WHERE (b.demolished IS NULL OR b.demolished >= '2017-01-01')
  AND (b.apn)::text <> 'CONDO BUILDING'
  AND (b.housing_units > 0 OR b.nonres_sqft > 0)
"""

QUERY_PERIOD2 = """
SELECT
    b.building_id,
    b.parcel_id,
    b.city_id,
    b.build_type                        AS building_type_id,
    b.year_built,
    b.housing_units                     AS residential_units,
    b.res_sqft,
    b.nonres_sqft                       AS non_residential_sqft,
    CASE
        WHEN b.demolished IS NOT NULL
             AND b.demolished >= '2021-01-01'
             AND b.demolished <  '2025-01-01'
        THEN 1 ELSE 0
    END                                 AS demolished,
    COALESCE(tc.large_area_id, 0)       AS large_area_id,
    COALESCE(ua.building_value, 0)      AS bldgimprval,
    COALESCE(ua.land_value,    0)       AS landvalue,
    COALESCE(ua.exempt,        0)       AS exempt
FROM creator.terra_buildings b
LEFT JOIN creator.urbansim_assessing  ua ON b.parcel_id = ua.parcel_id
LEFT JOIN creator.terminus_city_id   tc ON b.city_id   = tc.city_id
WHERE (b.demolished IS NULL
       OR date_part('year', b.demolished) > 2020
       OR (date_part('year', b.demolished) = 2020
           AND date_part('month', b.demolished) >= 4))
  AND (b.apn)::text <> 'CONDO BUILDING'
  AND (b.housing_units > 0 OR b.nonres_sqft > 0)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_period(query, label):
    conn = psycopg2.connect(DB_CONN)
    df = pd.read_sql(query, conn)
    conn.close()
    df["period"] = label
    n_pos = df["demolished"].sum()
    print(f"  {label}: {len(df):>10,} buildings  |  demolished={n_pos:,} ({n_pos/len(df)*100:.2f}%)")
    return df


def engineer_features(df):
    df = df.copy()
    df["building_age"] = (AGE_REF_YEAR - df["year_built"]).clip(lower=0)
    total_sqft = (df["res_sqft"] + df["non_residential_sqft"]).clip(lower=1)
    df["impr_value_per_sqft"] = (df["bldgimprval"] / total_sqft).clip(lower=0, upper=500)
    df["land_to_impr_ratio"] = (
        df["landvalue"] / df["bldgimprval"].clip(lower=1)
    ).clip(lower=0, upper=50)
    df["is_exempt"] = (df["exempt"] > 0).astype(int)
    df["is_wayne"]  = (df["large_area_id"] == WAYNE_LA_ID).astype(int)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(train_df, val_df, features, label="demolished", name=""):
    """
    Fit logistic regression on train_df, evaluate on val_df.
    Returns (model, scaler, train_auc, val_auc).
    """
    train_df = train_df.dropna(subset=features + [label])
    val_df   = val_df.dropna(subset=features + [label])

    X_tr = train_df[features].values
    y_tr = train_df[label].values
    X_va = val_df[features].values
    y_va = val_df[label].values

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)

    model = LogisticRegression(class_weight="balanced", max_iter=500,
                               C=1.0, random_state=42, solver="lbfgs")
    model.fit(X_tr_sc, y_tr)

    train_auc = roc_auc_score(y_tr, model.predict_proba(X_tr_sc)[:, 1])
    val_auc   = roc_auc_score(y_va, model.predict_proba(X_va_sc)[:, 1])
    val_ap    = average_precision_score(y_va, model.predict_proba(X_va_sc)[:, 1])

    print(f"\n  {name}")
    print(f"    Train AUC : {train_auc:.4f}")
    print(f"    Val   AUC : {val_auc:.4f}   Avg-Precision: {val_ap:.4f}")
    print(f"    Coefficients:")
    for feat, coef in zip(features, model.coef_[0]):
        print(f"      {feat:<25s} {coef:+.4f}")
    print(f"      {'intercept':<25s} {model.intercept_[0]:+.4f}")

    return model, scaler, train_auc, val_auc


def refit_combined(train_df, val_df, features, label="demolished"):
    """Refit on all data (train + val) after evaluation, for deployment."""
    combined = pd.concat([train_df, val_df], ignore_index=True).dropna(subset=features + [label])
    X = combined[features].values
    y = combined[label].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    model = LogisticRegression(class_weight="balanced", max_iter=500,
                               C=1.0, random_state=42, solver="lbfgs")
    model.fit(X_sc, y)
    return model, scaler

# ─────────────────────────────────────────────────────────────────────────────
# Config output
# ─────────────────────────────────────────────────────────────────────────────

def _model_section(model, scaler, features):
    return {
        "is_calibrated": True,
        "feature_mean": {f: float(scaler.mean_[i])  for i, f in enumerate(features)},
        "feature_std":  {f: float(scaler.scale_[i]) for i, f in enumerate(features)},
        "intercept":    float(model.intercept_[0]),
        "coef":         {f: float(model.coef_[0][i]) for i, f in enumerate(features)},
    }


def save_config(res_model, res_scaler, nonres_model, nonres_scaler):
    # Read existing config so non-model keys (eligibility, multiplier) are preserved
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    cfg["residential"]    = _model_section(res_model,    res_scaler,    RES_FEATURES)
    cfg["nonresidential"] = _model_section(nonres_model, nonres_scaler, NONRES_FEATURES)

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved → {CONFIG_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading training data from postgres...")
    p1 = load_period(QUERY_PERIOD1, "2017-2019")
    p2 = load_period(QUERY_PERIOD2, "2021-2024")

    print("\nEngineering features...")
    p1 = engineer_features(p1)
    p2 = engineer_features(p2)

    # ── Residential ──
    print("\n=== RESIDENTIAL (build_type 81/82/83) ===")
    res_types = {81, 82, 83}
    p1_res = p1[p1["building_type_id"].isin(res_types)]
    p2_res = p2[p2["building_type_id"].isin(res_types)]
    print(f"  Train: {len(p1_res):,}  |  Val: {len(p2_res):,}")

    res_model_eval, _, _, _ = train_and_evaluate(
        p1_res, p2_res, RES_FEATURES, name="Residential (train=2017-2019, val=2021-2024)"
    )
    res_model_final, res_scaler_final = refit_combined(p1_res, p2_res, RES_FEATURES)

    # ── Non-residential ──
    print("\n=== NON-RESIDENTIAL ===")
    p1_nonres = p1[~p1["building_type_id"].isin(res_types) & (p1["non_residential_sqft"] > 0)]
    p2_nonres = p2[~p2["building_type_id"].isin(res_types) & (p2["non_residential_sqft"] > 0)]
    print(f"  Train: {len(p1_nonres):,}  |  Val: {len(p2_nonres):,}")

    nonres_model_eval, _, _, _ = train_and_evaluate(
        p1_nonres, p2_nonres, NONRES_FEATURES, name="Non-residential (train=2017-2019, val=2021-2024)"
    )
    nonres_model_final, nonres_scaler_final = refit_combined(
        p1_nonres, p2_nonres, NONRES_FEATURES
    )

    # ── Validation summary by large_area ──
    print("\n=== VALIDATION: Demolition rate by large_area (2021-2024) ===")
    p2_res["pred_score"] = res_model_final.predict_proba(
        res_scaler_final.transform(p2_res[RES_FEATURES].fillna(0).values)
    )[:, 1]
    for la_id, grp in p2_res.groupby("large_area_id"):
        actual = grp["demolished"].mean() * 100
        avg_score = grp["pred_score"].mean() * 100
        n = len(grp)
        print(f"  large_area {la_id:>3}  n={n:>7,}  actual_rate={actual:.2f}%  avg_score={avg_score:.2f}%")

    # ── Save ──
    print("\nSaving calibrated config...")
    save_config(res_model_final, res_scaler_final,
                nonres_model_final, nonres_scaler_final)
    print("Done.  Re-run the simulation to use calibrated demolition scores.")


if __name__ == "__main__":
    main()
