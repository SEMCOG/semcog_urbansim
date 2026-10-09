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
import sqlalchemy
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import sparse
from urbansim.utils import misc

CONFIG_PATH = os.path.join(misc.configs_dir(), "demolition_model.yaml")

# Connection string is read at run time; never write credentials in this file
SQL_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "forecast_data_input", "config", "sql.yaml")

RES_TYPES = [81, 82, 83]

# Same names and formulas as the simulation building columns (variables_building.py)
RES_FEATURES    = ["building_age", "impr_value_per_sqft", "land_to_impr_ratio"]
NONRES_FEATURES = ["building_age"]  # value features add almost nothing (holdout AUC 0.624 vs 0.618)

# Simulation eligibility (configs/demolition_model.yaml: min_age_eligible)
MIN_AGE = 10

# ─────────────────────────────────────────────────────────────────────────────
# SQL
# ─────────────────────────────────────────────────────────────────────────────

QUERY_BUILDINGS = """
SELECT b.building_id, b.parcel_id, b.apn, b.city_id,
       b.build_type                    AS building_type_id,
       b.year_built,
       b.housing_units                 AS residential_units,
       b.res_sqft,
       b.nonres_sqft                   AS non_residential_sqft,
       b.demolished::date              AS demolished,
       COALESCE(tc.large_area_id, 0)   AS large_area_id
FROM creator.terra_buildings b
LEFT JOIN creator.terminus_city_id tc ON b.city_id = tc.city_id
WHERE (b.apn)::text <> 'CONDO BUILDING'
  AND (b.housing_units > 0 OR b.nonres_sqft > 0)
"""

# Tax year T shows the condition on Dec 31 of T-1, so 2017-2019 precede both periods
QUERY_ASSESSING = """
SELECT apn, taxyear::int AS taxyear, sev, bldgimprval, landvalue
FROM creator.terra_assessing
WHERE taxyear IN (2017, 2018, 2019)
"""

# Parcel apn: urbansim_assessing joins the assessment by the parcel's apn.
# Parcel area: used by urbansim_assessing for market_value when sev = 0.
# Centroids: used by the base data step to fill a missing year_built.
QUERY_PARCELS = """
SELECT parcel_id, apn AS parcel_apn, ST_Area(the_geom) AS parcel_area,
       ST_X(ST_Centroid(the_geom)) AS centroid_x, ST_Y(ST_Centroid(the_geom)) AS centroid_y
FROM creator.terra_parcels_2023_latest
"""

# Annual demolition volume by city: units of types 81/82/83 and non-residential
# sqft, averaged over complete years with 0 counted. 2021-2024: complete years
# after the pandemic (Detroit has no permit data for the second half of 2025).
VOLUME_PATH = os.path.join(misc.configs_dir(), "demolition_rates.csv")
VOLUME_FIRST_YEAR, VOLUME_LAST_YEAR = 2021, 2024

QUERY_VOLUME = f"""
SELECT c.city_id,
       COALESCE(SUM(b.nonres_sqft), 0)                                     AS typenonsqft,
       COALESCE(SUM(b.housing_units) FILTER (WHERE b.build_type = 81), 0) AS type81units,
       COALESCE(SUM(b.housing_units) FILTER (WHERE b.build_type = 82), 0) AS type82units,
       COALESCE(SUM(b.housing_units) FILTER (WHERE b.build_type = 83), 0) AS type83units
FROM creator.terminus_city_id c
LEFT JOIN creator.terra_buildings b
  ON b.city_id = c.city_id
 AND date_part('year', b.demolished) BETWEEN {VOLUME_FIRST_YEAR} AND {VOLUME_LAST_YEAR}
GROUP BY c.city_id
ORDER BY c.city_id
"""

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def get_engine():
    with open(SQL_YAML) as f:
        return sqlalchemy.create_engine(yaml.safe_load(f)["db_pgforecast_connection_str"])


def load_inputs(engine):
    bldg = pd.read_sql(QUERY_BUILDINGS, engine)
    bldg["demolished"] = pd.to_datetime(bldg["demolished"])
    assess = pd.read_sql(QUERY_ASSESSING, engine)
    parcels = pd.read_sql(QUERY_PARCELS, engine).set_index("parcel_id")
    return bldg, assess, parcels


def impute_year_built(bldg, parcels, before=None):
    """Same rules as the base data step (forecast_data_input/model_input/transform.py).

    urbansim_buildings changes year_built 0 to -1. A missing value gets the
    year_built of the nearest building with a known value (parcel centroids).
    A value before 1760 becomes 1760.

    before: for training data, use only buildings built before this year.
    Otherwise a demolished building can get the year of its own replacement.
    """
    yb = bldg["year_built"].where(bldg["year_built"] > 0)
    xy = parcels.reindex(bldg["parcel_id"])[["centroid_x", "centroid_y"]].to_numpy()
    known = yb.notna().to_numpy() & ~np.isnan(xy).any(axis=1)
    if before is not None:
        known &= (yb < before).to_numpy()
    missing = yb.isna().to_numpy() & ~np.isnan(xy).any(axis=1)
    _, idx = cKDTree(xy[known]).query(xy[missing], k=1)
    yb = yb.to_numpy(copy=True)
    yb[missing] = yb[known][idx]
    return pd.Series(yb, index=bldg.index).clip(lower=1760)


def processed_values(df):
    """Reproduce creator.urbansim_assessing: market_value, building_value, land_value.

    df needs sev, bldgimprval, landvalue (raw assessment), parcel_area, and the
    parcel totals bsqft and bunits of the standing buildings.
    """
    sev  = df["sev"].fillna(0)
    bv   = df["bldgimprval"].fillna(0)
    lv   = df["landvalue"].fillna(0)
    area = df["parcel_area"].fillna(0)

    market = np.select(
        [(sev == 0) & (bv > 0) & (lv > 0),
         (sev == 0) & (df["bunits"] == 0),
         (sev == 0),
         sev * 2 < bv + lv],
        [bv + lv,
         np.round(area * 0.19) * 2,
         np.round(area * 0.38) * 2,
         bv + lv],
        default=sev * 2,
    )
    building_value = np.select(
        [(bv == 0) & (df["bsqft"] > 0),
         (bv == 0),
         bv >= market],
        [np.round(market * 0.7), 0, np.round(market * 0.7)],
        default=bv,
    )
    return pd.DataFrame({"building_value": building_value,
                         "land_value": market - building_value}, index=df.index)


def compute_features(df, ref_year):
    """Simulation feature formulas (variables_building.py) for buildings at ref_year.

    df needs year_built_imputed, residential sqft, non_residential_sqft,
    building_value and land_value (processed, per parcel).
    """
    total_sqft = (df["res_sqft"].fillna(0) + df["non_residential_sqft"].fillna(0)).clip(lower=1)
    return pd.DataFrame({
        "building_age":        ref_year - df["year_built_imputed"],
        "impr_value_per_sqft": (df["building_value"] / total_sqft).clip(lower=0, upper=500),
        "land_to_impr_ratio":  (df["land_value"] / df["building_value"].clip(lower=1)).clip(lower=0, upper=50),
    }, index=df.index)


def build_period(bldg, parcels, start_year, end_year, label):
    """Training rows for one period, with every feature measured before the outcomes.

    start_year: a Series (period 1: the building's tax year) or an int (period 2).
    A building is in the population if it stands at Jan 1 of its start year and
    is at least MIN_AGE years old then. The outcome is a demolition from the
    start year through end_year.
    """
    df = bldg.copy()
    df["start_year"] = start_year
    df = df[df["start_year"].notna()].copy()
    df["start_year"] = df["start_year"].astype(int)

    start = pd.to_datetime(df["start_year"].astype(str) + "-01-01")
    df = df[df["demolished"].isna() | (df["demolished"] >= start)]

    # Parcel totals of the standing buildings (urbansim_assessing uses them)
    tot = df.groupby("parcel_id")[["residential_units", "res_sqft", "non_residential_sqft"]].sum()
    df["bunits"] = df["parcel_id"].map(tot["residential_units"])
    df["bsqft"]  = df["parcel_id"].map(tot["res_sqft"] + tot["non_residential_sqft"])
    df["parcel_area"] = df["parcel_id"].map(parcels["parcel_area"])

    df = df.join(processed_values(df))
    df = df.join(compute_features(df, df["start_year"]))
    df = df[df["building_age"] >= MIN_AGE]

    yr = df["demolished"].dt.year
    df["demolished_flag"] = (yr >= df["start_year"]) & (yr <= end_year)
    df["demolished_flag"] = df["demolished_flag"].astype(int)
    df["period"] = label
    return df


def build_training_data(bldg, assess, parcels):
    """Period 1: 2017-2019, start = earliest tax year. Period 2: 2021-2024, start = 2021."""
    bldg = bldg.copy()

    a = assess.sort_values("taxyear")
    earliest = a.drop_duplicates("apn", keep="first").set_index("apn")
    latest   = a.drop_duplicates("apn", keep="last").set_index("apn")
    cols = ["taxyear", "sev", "bldgimprval", "landvalue"]

    # The view joins by the parcel apn. A redeveloped parcel can have a new apn,
    # so fall back to the building apn when the parcel apn has no assessment.
    bldg["parcel_apn"] = bldg["parcel_id"].map(parcels["parcel_apn"])

    def join_assessment(table):
        out = bldg.join(table[cols], on="parcel_apn")
        miss = out["taxyear"].isna()
        out.loc[miss, cols] = bldg.loc[miss, ["apn"]].join(table[cols], on="apn")[cols]
        return out

    b1 = join_assessment(earliest)
    b1["year_built_imputed"] = impute_year_built(bldg, parcels, before=2017)
    p1 = build_period(b1, parcels, b1["taxyear"], 2019, "2017-2019")

    b2 = join_assessment(latest)
    b2["year_built_imputed"] = impute_year_built(bldg, parcels, before=2021)
    b2 = b2[b2["taxyear"].notna()]
    p2 = build_period(b2, parcels, 2021, 2024, "2021-2024")
    return p1, p2


def report_coverage(bldg, p1, p2):
    """Step 2 check: population, demolitions, tax year and match rate by large area."""
    for p in (p1, p2):
        assert (p["year_built"].between(1600, 2100) & (p["year_built"] >= p["start_year"])).sum() == 0, \
            "a building in the population was built after its start year"
    rows = []
    for p in (p1, p2):
        lo, hi = (2017, 2019) if p["period"].iat[0] == "2017-2019" else (2021, 2024)
        yr = bldg["demolished"].dt.year
        all_dem = bldg[yr.between(lo, hi)].groupby("large_area_id").size()
        g = p.groupby("large_area_id")
        rows.append(pd.DataFrame({
            "period": p["period"].iat[0],
            "buildings": g.size(),
            "demolished": g["demolished_flag"].sum(),
            "all_demolished": all_dem,
            "taxyears": g["taxyear"].agg(lambda s: ",".join(str(int(t)) for t in sorted(s.dropna().unique()))),
        }))
    out = pd.concat(rows).reset_index()
    out["demolished_matched_pct"] = (100 * out["demolished"] / out["all_demolished"]).round(1)
    return out

def build_volume_table(engine):
    """Average annual demolition volume by city (step 6)."""
    total = pd.read_sql(QUERY_VOLUME, engine).set_index("city_id")
    return (total / (VOLUME_LAST_YEAR - VOLUME_FIRST_YEAR + 1)).round().astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Estimation (step 4)
# ─────────────────────────────────────────────────────────────────────────────

def select_pathway(df, kind):
    """Rows and strata as scored_demolition_events samples them.

    Residential: types 81/82/83 with no non-residential sqft, stratum city x type.
    Non-residential: any building with non-residential sqft, stratum city.
    """
    if kind == "residential":
        df = df[df["building_type_id"].isin(RES_TYPES) & (df["non_residential_sqft"].fillna(0) == 0)]
        strata = df["city_id"] * 100 + df["building_type_id"]
    else:
        df = df[df["non_residential_sqft"].fillna(0) > 0]
        strata = df["city_id"]
    return df, strata.rename("stratum")


def fit_fixed_effects(df, strata, features, label="demolished_flag"):
    """Logit with one indicator per stratum and standardized features.

    Strata without a demolition carry no information about the features and
    are dropped. Returns coefficients, standard errors, and the standardization.
    """
    has_event = df[label].groupby(strata).transform("max") > 0
    df, strata = df[has_event], strata[has_event]

    mean = df[features].mean()
    std = df[features].std()
    z = ((df[features] - mean) / std).to_numpy()
    codes, _ = pd.factorize(strata)
    dummies = sparse.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)))
    X = sparse.hstack([sparse.csr_matrix(z), dummies]).tocsr()
    y = df[label].to_numpy()

    model = LogisticRegression(C=np.inf, fit_intercept=False, solver="newton-cholesky", max_iter=200)
    model.fit(X, y)

    # Standard errors from the information matrix X'WX
    p = model.predict_proba(X)[:, 1]
    info = (X.T @ X.multiply((p * (1 - p))[:, None])).toarray()
    se = np.sqrt(np.diag(np.linalg.pinv(info))[:len(features)])
    k = len(features)
    return {
        "coef": pd.Series(model.coef_[0][:k], index=features),
        "se": pd.Series(se, index=features),
        "feature_mean": mean,
        "feature_std": std,
        "n_strata": int(strata.nunique()),
        "n_events": int(y.sum()),
    }


def odds_score(df, fit):
    """Selection weight in the simulation: exp(beta . z), no intercept, no stratum term."""
    z = (df[fit["coef"].index] - fit["feature_mean"]) / fit["feature_std"]
    return np.exp(z.to_numpy() @ fit["coef"].to_numpy())


# ─────────────────────────────────────────────────────────────────────────────
# Validation (step 5)
# ─────────────────────────────────────────────────────────────────────────────

def within_stratum_auc(y, score, strata):
    """AUC inside each stratum, averaged with the number of demolitions as weight."""
    d = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(score), "g": np.asarray(strata)})
    aucs, weights = [], []
    for _, g in d.groupby("g"):
        n_pos = g["y"].sum()
        if 0 < n_pos < len(g):
            aucs.append(roc_auc_score(g["y"], g["s"]))
            weights.append(n_pos)
    return float(np.average(aucs, weights=weights))


def config_score(df, section):
    """Score of a yaml section as the simulation computes it (old: logistic, new: odds)."""
    feats = list(section["coef"])
    if section.get("score") != "odds":
        feats = [f for f in feats if f in RES_FEATURES]  # is_exempt is not in the simulation
    z = sum(section["coef"][f] * (df[f] - section["feature_mean"][f]) / section["feature_std"][f]
            for f in feats)
    if section.get("score") == "odds":
        return np.exp(z)
    return 1.0 / (1.0 + np.exp(-np.clip(z + section["intercept"], -15, 15)))


def add_raw_features(df):
    """Diagnostic: the same features from the raw assessment, without the view's rules."""
    total_sqft = (df["res_sqft"].fillna(0) + df["non_residential_sqft"].fillna(0)).clip(lower=1)
    bv = df["bldgimprval"].fillna(0)
    out = df.copy()
    out["impr_value_per_sqft"] = (bv / total_sqft).clip(lower=0, upper=500)
    out["land_to_impr_ratio"] = (df["landvalue"].fillna(0) / bv.clip(lower=1)).clip(lower=0, upper=50)
    return out


DETROIT_LA_ID = 5

# Detroit demolitions are mostly blight removal and the other areas mostly
# teardowns: the residential coefficients differ (land_to_impr_ratio: -2.4 vs +0.07).
# Each city is entirely in one area, so each sampling stratum uses one model.
MODELS = {
    "residential_detroit": ("residential",    lambda df: df["large_area_id"] == DETROIT_LA_ID, RES_FEATURES),
    "residential":         ("residential",    lambda df: df["large_area_id"] != DETROIT_LA_ID, RES_FEATURES),
    "nonresidential":      ("nonresidential", lambda df: df["large_area_id"].notna(),          NONRES_FEATURES),
}


def model_rows(df, key):
    kind, in_area, features = MODELS[key]
    d, strata = select_pathway(df, kind)
    keep = in_area(d)
    return d[keep], strata[keep], features


def validate(p1, p2, key, current_section):
    """Train on period 1, test on period 2. Returns the AUC table and the coefficients."""
    tr, tr_s, features = model_rows(p1, key)
    te, te_s, _ = model_rows(p2, key)
    y = te["demolished_flag"]

    new = fit_fixed_effects(tr, tr_s, features)
    auc = {
        "new model": within_stratum_auc(y, odds_score(te, new), te_s),
        "current yaml": within_stratum_auc(y, config_score(te, current_section), te_s),
    }
    if features != ["building_age"]:
        age = fit_fixed_effects(tr, tr_s, ["building_age"])
        raw = fit_fixed_effects(add_raw_features(tr), tr_s, features)
        auc["age only"] = within_stratum_auc(y, odds_score(te, age), te_s)
        auc["raw values (diagnostic)"] = within_stratum_auc(y, odds_score(add_raw_features(te), raw), te_s)

    coefs = {"2017-2019": new, "2021-2024": fit_fixed_effects(te, te_s, features)}
    return pd.Series(auc, name=f"{key} AUC 2021-2024"), coefs


def coef_table(fits):
    rows = []
    for label, f in fits.items():
        for feat in f["coef"].index:
            b, se = f["coef"][feat], f["se"][feat]
            rows.append({"period": label, "feature": feat, "coef": b,
                         "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se,
                         "events": f["n_events"], "strata": f["n_strata"]})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Config output
# ─────────────────────────────────────────────────────────────────────────────

def _model_section(fit):
    return {
        "is_calibrated": True,
        "score": "odds",  # weight = exp(sum coef * standardized feature), no intercept
        "feature_mean": {f: float(v) for f, v in fit["feature_mean"].items()},
        "feature_std":  {f: float(v) for f, v in fit["feature_std"].items()},
        "coef":         {f: float(v) for f, v in fit["coef"].items()},
    }


def save_config(fits):
    # Read existing config so non-model keys (eligibility, multiplier) are preserved
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    for key, fit in fits.items():
        cfg[key] = _model_section(fit)

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved → {CONFIG_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(save=True):
    print("Loading data from postgres...")
    bldg, assess, parcels = load_inputs(get_engine())
    p1, p2 = build_training_data(bldg, assess, parcels)
    print(report_coverage(bldg, p1, p2).to_string(index=False))

    with open(CONFIG_PATH) as f:
        current = yaml.safe_load(f)

    final = {}
    for key in MODELS:
        section = current.get(key, current["residential"])
        auc, fits = validate(p1, p2, key, section)
        print(f"\n=== {key.upper()} ===")
        print(auc.round(4).to_string())
        print(coef_table(fits).round(3).to_string(index=False))
        if auc["new model"] < auc["current yaml"]:
            print("  New model is worse than the current yaml on the holdout: config not saved.")
            save = False

        d, strata, features = model_rows(pd.concat([p1, p2]), key)
        strata = strata.astype(str) + "_" + d["period"]  # strata do not pool across periods
        final[key] = fit_fixed_effects(d, strata, features)
        print("Deployed (both periods):")
        print(coef_table({"both": final[key]}).round(3).to_string(index=False))

    volume = build_volume_table(get_engine())
    detroit = volume.index.to_series().between(500, 599)
    print("\n=== VOLUME per year, 2021-2024 ===")
    print(pd.DataFrame({"region": volume.sum(), "Detroit": volume[detroit].sum()}).to_string())

    if save:
        save_config(final)
        volume.to_csv(VOLUME_PATH)
        print(f"Saved → {VOLUME_PATH}")


if __name__ == "__main__":
    main()
