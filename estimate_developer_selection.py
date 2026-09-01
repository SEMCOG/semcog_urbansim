"""
Residential developer site selection — estimation.

Positive labels: parcels with year_built >= LABEL_YEAR.
Negative sample: 1:1 within same MCD.
One logistic regression per LUT (>= MIN_LUT_POS positives) + pooled fallback.
Output: configs/developer_selection_coefs.yaml
"""

import os
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

import orca
from urbansim.utils import misc

orca.add_injectable('hlcm_model_path',  '/mnt/hgfs/RDF2050/estimation/models/models_25Nov13')
orca.add_injectable('elcm_model_path',  '/mnt/hgfs/RDF2050/estimation/models/elcm_models_25May30/')
orca.add_injectable('yaml_configs',     'yaml_configs_elcm_hlcm.yaml')
orca.add_injectable('use_checkpoint',   False)
orca.add_injectable('base_year',        2020)
orca.add_injectable('final_year',       2022)
orca.add_injectable('ENABLE_SCENARIO',  False)

import assumptions, dataset, variables, models

BASE_YEAR    = 2020
LABEL_YEAR   = BASE_YEAR - 5
LOOKBACK     = 7
NEG_RATIO    = 1
MIN_LUT_POS  = 200
OUT_PATH     = os.path.join(misc.configs_dir(), "developer_selection_coefs.yaml")

FEATURES = [
    "local_vacancy",
    "development_momentum",
    "accessibility_composite",
    "recent_mover_rate",
    "log_parcel_sqft",
    "bldg_impr_land_ratio",
]


def _col(df, name):
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def _norm(s):
    s = s.fillna(s.median() if not s.isna().all() else 0.0)
    lo, hi = s.min(), s.max()
    return pd.Series(0.0, index=s.index) if hi == lo else (s - lo) / (hi - lo)


def build_parcel_features(pcl, bld, zon):
    """Return parcel-level feature DataFrame (index=parcel_id)."""
    f = pd.DataFrame(index=pcl.index)

    f["local_vacancy"] = (
        pcl["zone_id"].map(zon["percent_vacant_residential_units"])
        if "zone_id" in pcl.columns and "percent_vacant_residential_units" in zon.columns
        else np.nan
    )

    lb = BASE_YEAR - LOOKBACK
    recent = bld[(bld["year_built"] >= lb) & (bld["residential_units"] > 0)]
    if "census_bg_id" in pcl.columns:
        bg_rate = (
            recent.join(pcl[["census_bg_id"]], on="parcel_id", how="left")
            .groupby("census_bg_id")["residential_units"].sum() / LOOKBACK
        )
        f["development_momentum"] = pcl["census_bg_id"].map(bg_rate).fillna(0)
    else:
        f["development_momentum"] = np.nan

    z = pcl["zone_id"] if "zone_id" in pcl.columns else None
    f["accessibility_composite"] = (
        0.35 * _norm(z.map(_col(zon, "jobs_within_30_min"))   if z is not None else pd.Series(np.nan, index=pcl.index))
      + 0.20 * _norm(z.map(_col(zon, "transit_jobs_30min"))    if z is not None else pd.Series(np.nan, index=pcl.index))
      - 0.20 * _norm(_col(pcl, "grocery_stores_walk_near_max90"))
      - 0.15 * _norm(_col(pcl, "fixed_route_bus_walk_near_max90"))
      - 0.10 * _norm(_col(pcl, "schools_k8_walk_near_max90"))
    )

    f["recent_mover_rate"] = _col(pcl, "recent_mover_rate")
    f["log_parcel_sqft"]   = np.log1p(_col(pcl, "parcel_sqft").clip(lower=0))
    land = _col(pcl, "landvalue").clip(lower=1)
    f["bldg_impr_land_ratio"] = (_col(pcl, "bldgimprval") / land).clip(upper=20)
    f["land_use_type_id"]  = _col(pcl, "land_use_type_id")

    return f


def build_features(feat_df):
    """Subset to residential parcels, drop NaN rows, return feature matrix."""
    res_bld = orca.get_table("buildings").to_frame(["parcel_id", "residential_units"])
    ids = res_bld[res_bld["residential_units"] > 0]["parcel_id"].unique()
    X = feat_df.reindex(ids)
    n0 = len(X)
    X = X.dropna(subset=[c for c in FEATURES if c in X.columns])
    print(f"  Universe: {n0:,} parcels → {len(X):,} after dropping NaN")
    return X[FEATURES]


def build_labels(idx, bld):
    res = bld[bld["residential_units"] > 0]
    pos = res.groupby("parcel_id")["year_built"].max()
    pos = pos[pos >= LABEL_YEAR].index
    y = pd.Series(idx.isin(pos).astype(int), index=idx)
    print(f"  Pos: {y.sum():,}  Neg pool: {(y==0).sum():,}")
    return y


def sample_negatives(X, y, mcd):
    pos = X[y == 1]
    neg = X[y == 0]
    mcd_pos, mcd_neg = mcd.reindex(pos.index), mcd.reindex(neg.index)
    parts = []
    for m, grp in mcd_neg.groupby(mcd_neg).groups.items():
        n = min(len(grp), int((mcd_pos == m).sum()) * NEG_RATIO)
        if n > 0:
            parts.append(neg.loc[grp].sample(n=n, random_state=42))
    neg_s = pd.concat(parts) if parts else neg.sample(frac=0.5, random_state=42)
    Xout = pd.concat([pos, neg_s])
    yout = pd.Series([1]*len(pos) + [0]*len(neg_s), index=Xout.index)
    print(f"  Train: {len(Xout):,} ({yout.sum():,} pos / {(yout==0).sum():,} neg)")
    return Xout, yout


def train(X, y):
    sc = StandardScaler()
    Xs = sc.fit_transform(X.values)
    m  = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    m.fit(Xs, y.values)
    auc = roc_auc_score(y.values, m.predict_proba(Xs)[:, 1])
    print(f"  AUC: {auc:.4f}")
    print(classification_report(y.values, m.predict(Xs), digits=3))
    for feat, c in zip(X.columns, m.coef_[0]):
        print(f"    {feat:<40} {c:+.4f}")
    return m, sc, list(X.columns)


def _pack(m, sc, cols):
    return {
        "features":    cols,
        "coef":        m.coef_[0].tolist(),
        "intercept":   float(m.intercept_[0]),
        "scaler_mean": sc.mean_.tolist(),
        "scaler_std":  sc.scale_.tolist(),
    }


def train_by_lut(X, y, mcd, lut):
    """Train pooled fallback + one model per active LUT."""
    out = {}
    print("\n── Pooled fallback ──────────────────────────────────────────────────")
    Xp, yp = sample_negatives(X, y, mcd)
    out["fallback"] = _pack(*train(Xp[FEATURES], yp))

    active = lut[y == 1].value_counts()
    active = active[active >= MIN_LUT_POS].index.tolist()
    print(f"\nActive LUTs (>= {MIN_LUT_POS} pos): {sorted(active)}")

    for lid in sorted(active):
        mask = lut == lid
        Xl, yl, ml = X[mask], y[mask], mcd[mask]
        print(f"\n── LUT {lid} ({int(yl.sum()):,} pos / {int((yl==0).sum()):,} neg) ──")
        try:
            Xs, ys = sample_negatives(Xl, yl, ml)
            out[int(lid)] = _pack(*train(Xs[FEATURES], ys))
        except Exception as e:
            print(f"  WARN: LUT {lid} failed ({e}) — using fallback")

    return out


def save(models_dict):
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        yaml.dump({"residential": models_dict}, f, default_flow_style=False)
    print(f"\nSaved → {OUT_PATH}  (fallback + {len(models_dict)-1} per-LUT)")


if __name__ == "__main__":
    print("=" * 60)
    print("Developer Site Selection — Estimation")
    print("=" * 60)

    orca.run(["build_networks"], iter_vars=[BASE_YEAR])
    orca.run(["neighborhood_vars"],   iter_vars=[BASE_YEAR])

    pcl = orca.get_table("parcels").to_frame([
        "zone_id", "census_bg_id", "semmcd", "recent_mover_rate",
        "grocery_stores_walk_near_max90", "fixed_route_bus_walk_near_max90",
        "schools_k8_walk_near_max90", "parcel_sqft", "bldgimprval",
        "landvalue", "land_use_type_id",
    ])
    bld = orca.get_table("buildings").to_frame(["parcel_id", "year_built", "residential_units"])
    zon = orca.get_table("zones").to_frame(
        ["percent_vacant_residential_units", "jobs_within_30_min", "transit_jobs_30min"]
    )

    feat_df = build_parcel_features(pcl, bld, zon)
    X       = build_features(feat_df)
    y       = build_labels(X.index, bld)
    mcd     = pcl["semmcd"].reindex(X.index)
    lut     = feat_df["land_use_type_id"].reindex(X.index)

    save(train_by_lut(X, y, mcd, lut))
