"""
Travel Survey Variable Preprocessing
=====================================
Reads SEMCOG Regional Travel Survey CSVs and computes block-group-level
behavioral aggregates. Results are registered as cached orca tables and
broadcast to zones, census tracts, parcels, and buildings via column definitions
in variables/.

All variables are BASE YEAR (2023 survey) and never update during simulation.
Register with cache_scope='forever'.

Survey data path is configured via the 'travel_survey_path' injectable in assumptions.py.

Variables computed (same column name at every geography level):
  transit_monthly_rate  — % persons using transit at least monthly (person.transit_freq)
  walk_choice_rate     — % of all trips made on foot (trip.mode_type)
  bike_weekly_rate     — % persons biking at least weekly (person.bike_freq)
  zero_veh_rate         — % households with 0 vehicles (hh.num_vehicles)
  avg_vehicles_per_hh   — mean vehicles per household
  wfh_rate              — % employed persons working from home 3+ days/week
  years_at_residence    — mean years at current residence
  recent_mover_rate     — % households that moved in within the last 10 years
  ev_hybrid_rate        — % vehicles that are EV, PHEV, or HEV
  median_commute_dist   — mean work-trip distance in miles

Zone and tract variables are aggregated from buildings (weighted by residential_units),
not directly from BG survey data. This weights by actual housing stock.

Zone and tract variables are aggregated from BG via a parcels-derived crosswalk.
Column names are identical across BG, zone, tract, parcel, and building tables.
"""

import os
import numpy as np
import pandas as pd
import orca

# ---------------------------------------------------------------------------
# Minimum weighted respondents per BG cell before falling back to county mean
# ---------------------------------------------------------------------------
_MIN_N = 15

# ---------------------------------------------------------------------------
# Integer codes confirmed from value_labels.csv  (value_labels.csv col: value)
# ---------------------------------------------------------------------------

# hh.is_complete
_IS_COMPLETE = 1

# hh.num_vehicles  — integer/categorical: 0–7 literal, 8 = "8 or more"
# Used as numeric directly; no remapping needed.

# person.employment  — categorical
_EMPLOYED_CODES = {1, 2, 3, 7, 8}   # full-time, part-time, self-employed, volunteer, furloughed

# person.telework_freq  — categorical  (lower value = more days)
# 1=6-7d/wk, 2=5d/wk, 3=4d/wk, 4=3d/wk  →  WFH ≥3 days/week
_WFH_FREQ_CODES = {1, 2, 3, 4}

# person.job_type  — categorical
_FULL_WFH_CODE = 3                   # "Work ONLY from home or remotely"

# person.transit_freq  — categorical (value_labels.csv confirmed)
# 1=6-7d/wk … 5=1-3d/month; 6=less than monthly; 996=Never
_TRANSIT_USE_CODES = {1, 2, 3, 4, 5}  # at least monthly

# person.bike_freq  — categorical (value_labels.csv confirmed)
# 1=6-7d/wk … 6=1d/wk; 7=1-3d/month; 8=less than monthly; 996=Never
_BIKE_WEEKLY_CODES = {1, 2, 3, 4, 5, 6}  # at least once per week

# trip.mode_type  — categorical (value_labels.csv confirmed)
_WALK_MODE_CODES = {1}              # 1=Walk

# trip.d_purpose_category  — categorical
_WORK_PURPOSE_CODES = {2, 3}        # 2=Work, 3=Work-related

# Max plausible one-way commute distance (miles)
_MAX_COMMUTE_MILES = 100

# vehicle.fuel_type  — categorical (value_labels.csv confirmed)
_EV_HYBRID_FUEL_CODES = {2, 3, 4}  # 2=Hybrid (HEV), 3=Plug-in hybrid (PHEV), 4=Electric (EV)


def _survey_path():
    """Return the configured travel survey data directory."""
    return orca.get_injectable("travel_survey_path")


def _read_csv(filename, usecols=None):
    """Read a survey CSV, returning an empty DataFrame if path is unavailable."""
    path = os.path.join(_survey_path(), filename)
    if not os.path.exists(path):
        print(f"  [travel_survey] WARNING: {path} not found — survey variables will be empty.")
        return pd.DataFrame()
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def _load_value_labels():
    """
    Load value_labels.csv as a nested dict:
        {(table, variable): {int_code: label_str}}
    Returns empty dict if file not found.
    """
    path = os.path.join(_survey_path(), "value_labels.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    labels = {}
    for _, row in df.iterrows():
        key = (row["table"], row["variable"])
        labels.setdefault(key, {})[int(row["value"])] = row["label"]
    return labels


def _load_var_dtypes():
    """
    Load variable_description.csv as dict: {(table, variable): data_type}.
    Returns empty dict if file not found.
    """
    path = os.path.join(_survey_path(), "variable_description.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, usecols=["variable", "table", "data_type"])
    return {(r["table"], r["variable"]): r["data_type"] for _, r in df.iterrows()}


def _to_int_codes(series, table, var, dtypes):
    """
    Convert a column to numeric integer codes when data_type is categorical.
    Non-numeric / sentinel values (995, 997, 998, 999) are coerced to NaN.
    For integer or numeric types, just coerce to numeric without sentinel removal.
    """
    dtype = dtypes.get((table, var), "")
    s = pd.to_numeric(series, errors="coerce")
    if "categorical" in dtype:
        # Sentinel/skip codes used across survey — treat as missing
        s = s.where(~s.isin([995, 997, 998, 999]))
    return s


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _weighted_mean(group_col, value_col, weight_col, df):
    """Weighted mean of value_col grouped by group_col."""
    num = (df[value_col] * df[weight_col]).groupby(df[group_col]).sum()
    den = df[weight_col].groupby(df[group_col]).sum()
    return (num / den).where(den > 0)


def _apply_county_fallback(bg_series, bg_to_county, n_by_bg, min_n=_MIN_N):
    """
    For BGs with fewer than min_n weighted respondents replace with county mean.
    bg_to_county: Series mapping bg_id → county_id
    n_by_bg:      Series of total weighted count per bg_id
    """
    county_mean = (
        bg_series.multiply(n_by_bg, fill_value=0)
        .groupby(bg_to_county.reindex(bg_series.index))
        .sum()
        / n_by_bg.groupby(bg_to_county.reindex(n_by_bg.index)).sum()
    )
    bg_county = bg_to_county.reindex(bg_series.index)
    fallback = bg_county.map(county_mean)
    result = bg_series.where(n_by_bg.reindex(bg_series.index, fill_value=0) >= min_n, fallback)
    return result.rename(bg_series.name)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute_bg_vars():
    """
    Read survey CSVs and compute all BG-level aggregate variables.
    Returns a DataFrame indexed by census_bg_id (int).
    Returns an empty DataFrame if survey data is unavailable.
    """
    print("  [travel_survey] Computing block-group-level survey variables...")

    # ── Load household table ─────────────────────────────────────────────────
    hh_cols = [
        "hh_id", "home_bg_2020", "home_county",
        "num_vehicles", "res_year", "hh_weight", "is_complete",
    ]
    hh = _read_csv("hh.csv", usecols=hh_cols)
    if hh.empty:
        return pd.DataFrame()

    # Filter to complete HH records in region
    hh = hh[hh["is_complete"] == _IS_COMPLETE].copy()
    hh["home_bg_2020"] = pd.to_numeric(hh["home_bg_2020"], errors="coerce")
    hh["home_county"]  = pd.to_numeric(hh["home_county"],  errors="coerce")
    hh = hh.dropna(subset=["home_bg_2020", "hh_weight"])
    hh["home_bg_2020"] = hh["home_bg_2020"].astype(np.int64)
    # Keep only Michigan households (state FIPS = 26); survey includes
    # out-of-region respondents whose BGs never match SEMCOG parcels
    hh = hh[hh["home_bg_2020"] // 10_000_000_000 == 26]

    # num_vehicles: coerce to numeric; treat non-numeric (e.g. "prefer not") as NaN
    hh["num_vehicles"] = pd.to_numeric(hh["num_vehicles"], errors="coerce")

    # res_year: years at address = 2025 - res_year; non-numeric → NaN
    hh["res_year_num"] = pd.to_numeric(hh["res_year"], errors="coerce")
    hh["years_at_address"] = (2025 - hh["res_year_num"]).clip(0, 50)
    hh["recent_mover"]     = (hh["years_at_address"] <= 10).astype(float)
    hh.loc[hh["res_year_num"].isna(), ["years_at_address", "recent_mover"]] = np.nan

    bg_to_county = hh.groupby("home_bg_2020")["home_county"].first()
    n_hh_by_bg   = hh.groupby("home_bg_2020")["hh_weight"].sum()

    # ── Load person table ────────────────────────────────────────────────────
    person_cols = [
        "hh_id", "person_weight", "is_complete",
        "employment", "telework_freq", "job_type",
        "transit_freq", "bike_freq",
    ]
    person = _read_csv("person.csv", usecols=person_cols)
    if person.empty:
        return pd.DataFrame()

    person = person[person["is_complete"] == _IS_COMPLETE].copy()
    person = person.merge(
        hh[["hh_id", "home_bg_2020", "hh_weight"]],
        on="hh_id", how="inner"
    )
    person = person.dropna(subset=["home_bg_2020", "person_weight"])
    person["home_bg_2020"] = person["home_bg_2020"].astype(np.int64)

    # Load dtypes once for integer-code coercion
    _dtypes = _load_var_dtypes()

    emp_codes = _to_int_codes(person["employment"], "person", "employment", _dtypes)
    person["is_employed"] = emp_codes.isin(_EMPLOYED_CODES).astype(float)
    person.loc[emp_codes.isna(), "is_employed"] = np.nan

    # WFH: telework_freq ≥3 days/week OR job_type = "work only from home"
    tw_codes  = _to_int_codes(person["telework_freq"], "person", "telework_freq", _dtypes)
    jt_codes  = _to_int_codes(person["job_type"],      "person", "job_type",      _dtypes)
    wfh_by_freq = tw_codes.isin(_WFH_FREQ_CODES)
    wfh_by_type = jt_codes == _FULL_WFH_CODE
    person["wfh_flag"] = (wfh_by_freq | wfh_by_type).astype(float)
    # Mark NaN only when both telework_freq AND job_type are missing
    both_missing = tw_codes.isna() & jt_codes.isna()
    person.loc[both_missing, "wfh_flag"] = np.nan
    person.loc[person["is_employed"] != 1, "wfh_flag"] = np.nan

    # ── Load trip table ───────────────────────────────────────────────────────
    trip_cols = ["hh_id", "trip_weight", "mode_type", "o_bg_2020",
                 "d_purpose_category", "distance_miles"]
    trip = _read_csv("trip.csv", usecols=trip_cols)

    # ── Load vehicle table (for EV/hybrid rate) ───────────────────────────────
    vehicle_cols = ["hh_id", "fuel_type", "hh_weight"]
    vehicle = _read_csv("vehicle.csv", usecols=vehicle_cols)

    # ── Compute BG aggregates ─────────────────────────────────────────────────

    results = {}

    # --- HH-level variables ---

    hh_veh = hh.dropna(subset=["num_vehicles"]).copy()
    hh_veh["zero_veh_flag"] = (hh_veh["num_vehicles"] == 0).astype(float)
    results["zero_veh_rate"] = _apply_county_fallback(
        _weighted_mean("home_bg_2020", "zero_veh_flag", "hh_weight", hh_veh),
        bg_to_county, n_hh_by_bg
    )
    results["avg_vehicles_per_hh"] = _apply_county_fallback(
        _weighted_mean("home_bg_2020", "num_vehicles", "hh_weight", hh_veh),
        bg_to_county, n_hh_by_bg
    )

    hh_stab = hh.dropna(subset=["years_at_address"])
    results["years_at_residence"] = _apply_county_fallback(
        _weighted_mean("home_bg_2020", "years_at_address", "hh_weight", hh_stab),
        bg_to_county, n_hh_by_bg
    )

    hh_mover = hh.dropna(subset=["recent_mover"])
    results["recent_mover_rate"] = _apply_county_fallback(
        _weighted_mean("home_bg_2020", "recent_mover", "hh_weight", hh_mover),
        bg_to_county, n_hh_by_bg
    )

    # --- Person-level variables ---

    workers = person[person["is_employed"] == 1]
    n_worker_by_bg = workers.groupby("home_bg_2020")["person_weight"].sum()

    results["wfh_rate"] = _apply_county_fallback(
        _weighted_mean("home_bg_2020", "wfh_flag", "person_weight",
                       workers.dropna(subset=["wfh_flag"])),
        bg_to_county, n_worker_by_bg
    )

    # Transit culture: at least monthly (codes 1–5)
    n_person_by_bg = person.groupby("home_bg_2020")["person_weight"].sum()
    tf_codes = _to_int_codes(person["transit_freq"], "person", "transit_freq", _dtypes)
    person["transit_use_flag"] = tf_codes.isin(_TRANSIT_USE_CODES).astype(float)
    person.loc[tf_codes.isna(), "transit_use_flag"] = np.nan
    results["transit_monthly_rate"] = _apply_county_fallback(
        _weighted_mean("home_bg_2020", "transit_use_flag", "person_weight",
                       person.dropna(subset=["transit_use_flag"])),
        bg_to_county, n_person_by_bg
    )

    # Bike culture: at least weekly (codes 1–6)
    bf_codes = _to_int_codes(person["bike_freq"], "person", "bike_freq", _dtypes)
    person["bike_weekly_flag"] = bf_codes.isin(_BIKE_WEEKLY_CODES).astype(float)
    person.loc[bf_codes.isna(), "bike_weekly_flag"] = np.nan
    results["bike_weekly_rate"] = _apply_county_fallback(
        _weighted_mean("home_bg_2020", "bike_weekly_flag", "person_weight",
                       person.dropna(subset=["bike_weekly_flag"])),
        bg_to_county, n_person_by_bg
    )


    # --- Vehicle-level variables ---

    if not vehicle.empty:
        vehicle = vehicle.merge(
            hh[["hh_id", "home_bg_2020", "home_county"]], on="hh_id", how="inner"
        )
        vehicle = vehicle.dropna(subset=["home_bg_2020"])
        vehicle["home_bg_2020"] = vehicle["home_bg_2020"].astype(np.int64)
        ftype = _to_int_codes(vehicle["fuel_type"], "vehicle", "fuel_type", _dtypes)
        vehicle["ev_hybrid_flag"] = ftype.isin(_EV_HYBRID_FUEL_CODES).astype(float)
        vehicle.loc[ftype.isna(), "ev_hybrid_flag"] = np.nan
        n_veh_by_bg = vehicle.groupby("home_bg_2020")["hh_weight"].sum()
        results["ev_hybrid_rate"] = _apply_county_fallback(
            _weighted_mean("home_bg_2020", "ev_hybrid_flag", "hh_weight",
                           vehicle.dropna(subset=["ev_hybrid_flag"])),
            bg_to_county, n_veh_by_bg
        )

    # --- Trip-level variables ---

    if not trip.empty:
        trip = trip.dropna(subset=["o_bg_2020", "trip_weight"])
        trip["o_bg_2020"] = pd.to_numeric(trip["o_bg_2020"], errors="coerce")
        trip = trip.dropna(subset=["o_bg_2020"])
        trip["o_bg_2020"] = trip["o_bg_2020"].astype(np.int64)

        n_trip_by_bg = trip.groupby("o_bg_2020")["trip_weight"].sum()

        mt_codes = _to_int_codes(trip["mode_type"], "trip", "mode_type", _dtypes)

        # Walk culture: % of trips made on foot
        trip["walk_flag"] = mt_codes.isin(_WALK_MODE_CODES).astype(float)
        trip.loc[mt_codes.isna(), "walk_flag"] = np.nan

        bg_trip_county = (
            trip[["o_bg_2020"]].drop_duplicates()
            .merge(hh[["home_bg_2020", "home_county"]].drop_duplicates(),
                   left_on="o_bg_2020", right_on="home_bg_2020", how="left")
            .drop_duplicates(subset=["o_bg_2020"])
            .set_index("o_bg_2020")["home_county"]
            .dropna()
        )

        results["walk_choice_rate"] = _apply_county_fallback(
            _weighted_mean("o_bg_2020", "walk_flag", "trip_weight",
                           trip.dropna(subset=["walk_flag"])),
            bg_trip_county, n_trip_by_bg
        ).rename("walk_choice_rate")

        # Median commute dist: mean work-trip distance (capped to remove GPS outliers)
        dp_codes = _to_int_codes(trip["d_purpose_category"], "trip", "d_purpose_category", _dtypes)
        trip["distance_miles"] = pd.to_numeric(trip["distance_miles"], errors="coerce")
        work_trips = trip[dp_codes.isin(_WORK_PURPOSE_CODES)].dropna(subset=["distance_miles"])
        work_trips = work_trips[work_trips["distance_miles"] <= _MAX_COMMUTE_MILES]
        results["median_commute_dist"] = _apply_county_fallback(
            _weighted_mean("o_bg_2020", "distance_miles", "trip_weight", work_trips),
            bg_trip_county,
            work_trips.groupby("o_bg_2020")["trip_weight"].sum()
        ).rename("median_commute_dist")

    # ── Combine into one DataFrame ───────────────────────────────────────────
    bg_df = pd.DataFrame(results)
    bg_df.index.name = "census_bg_id"
    bg_df = bg_df.fillna(bg_df.mean())   # last-resort fill with region mean
    print(f"  [travel_survey] BG vars computed: {len(bg_df)} block groups × {len(bg_df.columns)} variables")
    return bg_df


# ---------------------------------------------------------------------------
# Orca table registration
# ---------------------------------------------------------------------------

@orca.table(cache=True)
def travel_survey_bg_vars():
    """Block-group-level behavioral variables from the travel survey."""
    return _compute_bg_vars()
