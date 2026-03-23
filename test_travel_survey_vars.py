"""
Travel Survey Variables — Test & Summary Script

Tests the travel_survey_vars module and all derived orca columns, then prints
a human-readable summary of each variable's distribution at every geographic level.

Run:
    python test_travel_survey_vars.py
    python test_travel_survey_vars.py --save   # also writes summary to a text file
"""

import os
import argparse
import orca
import pandas as pd
import numpy as np

# ── Injectables required before any imports ───────────────────────────────────
orca.add_injectable('hlcm_model_path',  '/mnt/hgfs/RDF2050/estimation/models/models_25Nov13')
orca.add_injectable('elcm_model_path',  '/mnt/hgfs/RDF2050/estimation/models/elcm_models_25May30/')
orca.add_injectable('yaml_configs',     'yaml_configs_elcm_hlcm.yaml')
orca.add_injectable('use_checkpoint',   False)
orca.add_injectable('base_year',        2020)
orca.add_injectable('final_year',       2022)
orca.add_injectable('ENABLE_SCENARIO',  False)

import assumptions   # registers store, building_type_map, travel_survey_path, etc.
import dataset       # registers buildings, parcels, zones, census_tracts tables
import variables     # registers all column definitions incl. travel_survey_vars

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(val, col):
    """Format a scalar value: percent for rate/share columns, else 2 decimal places."""
    if pd.isna(val):
        return "  n/a"
    col_base = col.split("zone_")[-1].split("tract_")[-1].split("bg_")[-1] \
                  .split("p_")[-1].split("b_")[-1]
    if any(col_base.startswith(p) for p in
           ("transit_monthly", "walk_choice", "bike_weekly",
            "zero_veh", "wfh", "recent_mover", "ev_hybrid")):
        return f"{val*100:6.1f}%"
    return f"{val:8.2f}"


def _describe_series(s, col, indent="    "):
    """Return a multi-line string summarising a Series."""
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return f"{indent}(no data)\n"
    lines = []
    lines.append(f"{indent}N (non-null) : {len(s):>10,}")
    lines.append(f"{indent}mean         : {_fmt(s.mean(), col)}")
    lines.append(f"{indent}median       : {_fmt(s.median(), col)}")
    lines.append(f"{indent}std          : {_fmt(s.std(), col)}")
    lines.append(f"{indent}min          : {_fmt(s.min(), col)}")
    lines.append(f"{indent}p10          : {_fmt(s.quantile(0.10), col)}")
    lines.append(f"{indent}p25          : {_fmt(s.quantile(0.25), col)}")
    lines.append(f"{indent}p75          : {_fmt(s.quantile(0.75), col)}")
    lines.append(f"{indent}p90          : {_fmt(s.quantile(0.90), col)}")
    lines.append(f"{indent}max          : {_fmt(s.max(), col)}")
    zeros = (s == 0).sum()
    lines.append(f"{indent}zeros        : {zeros:>10,}  ({zeros/len(s)*100:.1f}%)")
    return "\n".join(lines) + "\n"


def section(title, char="="):
    width = 72
    return f"\n{char * width}\n  {title}\n{char * width}\n"


def subsection(title):
    return f"\n  {'─' * 68}\n  {title}\n  {'─' * 68}\n"


# ── Variable metadata ─────────────────────────────────────────────────────────

VARIABLE_META = {
    "transit_monthly_rate": {
        "label": "Transit Monthly Rate",
        "description": "% of persons who use transit at least monthly (transit_freq codes 1–5)",
        "source": "person.transit_freq",
        "unit": "proportion (0–1)",
    },
    "walk_choice_rate": {
        "label": "Walk Choice Rate",
        "description": "% of all trips made on foot",
        "source": "trip.mode_type (code 1=Walk), trip.trip_weight",
        "unit": "proportion (0–1)",
    },
    "bike_weekly_rate": {
        "label": "Bike Weekly Rate",
        "description": "% of persons who bike at least once a week (bike_freq codes 1–6)",
        "source": "person.bike_freq",
        "unit": "proportion (0–1)",
    },
    "zero_veh_rate": {
        "label": "Zero-Vehicle Household Rate",
        "description": "% of households with no personal vehicles",
        "source": "hh.num_vehicles",
        "unit": "proportion (0–1)",
    },
    "avg_vehicles_per_hh": {
        "label": "Average Vehicles per Household",
        "description": "Mean number of personal vehicles owned per household",
        "source": "hh.num_vehicles",
        "unit": "vehicles/HH",
    },
    "wfh_rate": {
        "label": "Work-From-Home Rate",
        "description": "% of employed residents teleworking 3+ days/week or fully remote",
        "source": "person.telework_freq, person.job_type",
        "unit": "proportion (0–1)",
    },
    "years_at_residence": {
        "label": "Years at Residence",
        "description": "Mean years residents have lived at their current address (capped at 50)",
        "source": "hh.res_year (2025 − res_year)",
        "unit": "years",
    },
    "recent_mover_rate": {
        "label": "Recent Mover Rate",
        "description": "% of households that moved to their current address within the last 10 years",
        "source": "hh.res_year",
        "unit": "proportion (0–1)",
    },
    "ev_hybrid_rate": {
        "label": "EV / Hybrid Vehicle Rate",
        "description": "% of household vehicles that are EV, PHEV, or HEV",
        "source": "vehicle.fuel_type (codes 2=HEV, 3=PHEV, 4=EV)",
        "unit": "proportion (0–1)",
    },
    "median_commute_dist": {
        "label": "Median Commute Distance",
        "description": "Mean work-trip distance in miles (capped at 100mi); zones/tracts aggregate from buildings",
        "source": "trip.distance_miles, trip.d_purpose_category in {Work, Work-related}",
        "unit": "miles",
    },
}

# Geography levels — all use same column names, no prefix
# Zones/tracts aggregate from buildings (not via survey crosswalk tables)
GEO_LEVELS = [
    ("Block Group",  "travel_survey_bg_vars", "census_bg_id"),
    ("Zone / TAZ",   "zones",                 "zone_id"),
    ("Census Tract", "census_tracts",         "tract_id"),
]

PARCEL_VARS   = list(VARIABLE_META)
BUILDING_VARS = list(VARIABLE_META)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_tests(save_path=None):
    lines = []
    log = lines.append

    log(section("TRAVEL SURVEY VARIABLE TEST & SUMMARY"))
    log(f"  Survey path : {orca.get_injectable('travel_survey_path')}\n")

    # ── 1. Survey table availability ─────────────────────────────────────────
    log(section("1. Survey Table Availability", "="))
    tbl = orca.get_table("travel_survey_bg_vars").to_frame()
    log(f"  travel_survey_bg_vars                OK  — {len(tbl):,} rows × {len(tbl.columns)} columns")
    log(f"  zones/census_tracts aggregate from buildings (not survey crosswalk tables)")

    # ── 2. Variable-by-variable summary at each geo level ────────────────────
    log(section("2. Variable Summaries by Geography", "="))

    for var_key, meta in VARIABLE_META.items():
        log(subsection(f"{meta['label']}"))
        log(f"  Description : {meta['description']}")
        log(f"  Source      : {meta['source']}")
        log(f"  Unit        : {meta['unit']}\n")

        for geo_label, tbl_name, idx_name in GEO_LEVELS:
            col = var_key
            tbl = orca.get_table(tbl_name).to_frame()
            assert col in tbl.columns, f"Missing column '{col}' in {tbl_name}"
            log(f"  [{geo_label}]  (col: {col})")
            log(_describe_series(tbl[col], col))

    # ── 3. Parcel-level spot check ────────────────────────────────────────────
    log(section("3. Parcel-Level Spot Check (sample 500k)", "="))
    log("  Checks that BG aggregates broadcast correctly to parcels.\n")

    parcels = orca.get_table("parcels")
    sample_idx = parcels.index[:min(500_000, len(parcels.index))]

    for var_key in list(VARIABLE_META.keys())[:4]:   # spot check first 4
        s = parcels[var_key].loc[sample_idx]
        n_zero = (s == 0).sum()
        n_null = s.isna().sum()
        log(f"  {var_key:<35} mean={_fmt(s.mean(), var_key)}  zeros={n_zero:,}  nulls={n_null:,}")

    # ── 4. Building-level spot check ─────────────────────────────────────────
    log(section("4. Building-Level Spot Check (sample 200k)", "="))
    log("  Checks that parcel variables propagate to buildings.\n")

    buildings = orca.get_table("buildings")
    sample_idx = buildings.index[:min(200_000, len(buildings.index))]

    for var_key in list(VARIABLE_META.keys())[:4]:
        s = buildings[var_key].loc[sample_idx]
        n_zero = (s == 0).sum()
        n_null = s.isna().sum()
        log(f"  {var_key:<35} mean={_fmt(s.mean(), var_key)}  zeros={n_zero:,}  nulls={n_null:,}")

    # ── 5. Cross-geography consistency check ─────────────────────────────────
    log(section("5. Cross-Geography Consistency Check", "="))
    log("  Region-wide means should be similar across BG, zone, and tract levels.\n")

    log(f"  {'Variable':<35} {'BG mean':>12} {'Zone mean':>12} {'Tract mean':>12}")
    log(f"  {'─'*35} {'─'*12} {'─'*12} {'─'*12}")

    bg_tbl   = orca.get_table("travel_survey_bg_vars").to_frame()
    zone_tbl = orca.get_table("zones").to_frame(list(VARIABLE_META))
    trac_tbl = orca.get_table("census_tracts").to_frame(list(VARIABLE_META))

    for var_key in VARIABLE_META:
        bg_mean   = bg_tbl[var_key].mean()   if var_key in bg_tbl.columns   else np.nan
        zone_mean = zone_tbl[var_key].mean() if var_key in zone_tbl.columns else np.nan
        trac_mean = trac_tbl[var_key].mean() if var_key in trac_tbl.columns else np.nan
        log(f"  {var_key:<35} {_fmt(bg_mean, var_key):>12} {_fmt(zone_mean, var_key):>12} {_fmt(trac_mean, var_key):>12}")

    # ── 6. Coverage report ────────────────────────────────────────────────────
    log(section("6. Coverage Report", "="))
    log("  % of entities with non-zero survey values (data availability).\n")

    for geo_label, tbl_name, idx_name in GEO_LEVELS:
        tbl = orca.get_table(tbl_name).to_frame()
        total = len(tbl)
        log(f"  {geo_label} ({total:,} entities):")
        for var_key in VARIABLE_META:
            if var_key not in tbl.columns:
                continue
            n_covered = (tbl[var_key] > 0).sum()
            log(f"    {var_key:<40} {n_covered:>7,} / {total:>7,}  ({n_covered/total*100:.1f}%)")
        log("")

    _output(lines, save_path)


def _output(lines, save_path):
    text = "\n".join(str(l) for l in lines)
    print(text)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(text)
        print(f"\nSummary saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test travel survey variable definitions")
    parser.add_argument("--save", action="store_true",
                        help="Save summary to runs/run_stdout/travel_survey_vars_summary.txt")
    args = parser.parse_args()

    save_path = "runs/run_stdout/travel_survey_vars_summary.txt" if args.save else None
    run_tests(save_path=save_path)
