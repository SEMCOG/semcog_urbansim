"""
Test residential developer pipeline for one simulated year.
Logs to runs/simulate_logs/test_developer_<timestamp>.log
"""

import os, sys, datetime, yaml
import orca
import numpy as np
import pandas as pd


class _Tee:
    def __init__(self, path):
        self._t = sys.stdout
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = open(path, "w", buffering=1)
    def write(self, s):  self._t.write(s);  self._f.write(s)
    def flush(self):     self._t.flush();   self._f.flush()
    def close(self):     self._f.close()

_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_log = os.path.join("runs", "simulate_logs", f"test_developer_{_ts}.log")
sys.stdout = _Tee(_log)
print(f"Logging to: {_log}")

orca.add_injectable('hlcm_model_path',  '/mnt/hgfs/RDF2050/estimation/models/models_25Nov13')
orca.add_injectable('elcm_model_path',  '/mnt/hgfs/RDF2050/estimation/models/elcm_models_25May30/')
orca.add_injectable('yaml_configs',     'yaml_configs_elcm_hlcm.yaml')
orca.add_injectable('use_checkpoint',   False)
orca.add_injectable('base_year',        2020)
orca.add_injectable('final_year',       2022)
orca.add_injectable('ENABLE_SCENARIO',  False)

import assumptions, dataset, variables, models

YR = 2021
pd.set_option("display.float_format", "{:,.1f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)

SEP = "=" * 70

# ── Steps 1-3 ─────────────────────────────────────────────────────────────────
for label, step in [
    ("STEP 1 — Build networks",       "build_networks_2050"),
    ("STEP 2 — Neighborhood vars",    "neighborhood_vars"),
    ("STEP 3 — Feasibility",          "feasibility"),
]:
    print(f"\n{SEP}\n{label}\n{SEP}")
    orca.run([step], iter_vars=[YR])

# ── Feasibility summary ────────────────────────────────────────────────────────
feas = orca.get_table("feasibility").to_frame()
rf   = feas[feas["form"] == "residential"].copy()
rf["profit"] = rf["building_revenue"] - rf["total_cost"]
aus = orca.get_table("parcels").to_frame(["ave_unit_size"])["ave_unit_size"]
rf["residential_units"] = (rf["residential_sqft"] / rf.index.map(aus).fillna(1000)).clip(lower=0).round()
best = rf.reset_index().sort_values("profit", ascending=False).drop_duplicates("parcel_id").set_index("parcel_id")
print(f"\nFeasibility: {len(feas):,} rows | res parcels: {len(best):,} | profitable: {(best['profit']>0).sum():,} | units: {best.loc[best['profit']>0,'residential_units'].sum():,.0f}")

# ── Site selection coefs ───────────────────────────────────────────────────────
print(f"\n{SEP}\nSITE SELECTION MODEL\n{SEP}")
coefs = orca.get_injectable("res_developer_selection_coefs")
if coefs and "residential" in coefs:
    lms = coefs["residential"]
    print(f"  Loaded: fallback + {len([k for k in lms if k != 'fallback'])} per-LUT models")
    print(f"  LUTs: {sorted(k for k in lms if k != 'fallback')}")
    for key, e in lms.items():
        print(f"\n  [{key}]")
        for feat, c in zip(e.get("features", []), e.get("coef", [])):
            print(f"    {feat:<35} {c:+.4f}")
else:
    print("  WARNING: no coefs found — profit-rank fallback")

# ── Config printout ────────────────────────────────────────────────────────────
print(f"\n{SEP}\nRES DEVELOPER CONFIG\n{SEP}")
with open("configs/res_developer.yaml") as _f:
    cfg = yaml.safe_load(_f)
for k, lbl in [
    ("target_weight_vacancy_gap",  "w_gap"),
    ("target_weight_recent_rate",  "w_recent"),
    ("target_weight_demand",       "w_demand"),
    ("b_score_exponent",           "b_score_exponent"),
    ("la_alignment_tolerance",     "la_alignment_tolerance"),
    ("min_unit_size",              "min_unit_size"),
    ("max_parcel_size",            "max_parcel_size"),
    ("drop_after_build",           "drop_after_build"),
    ("keep_suboptimal",            "keep_suboptimal"),
    ("noise_scale",                "noise_scale"),
]:
    print(f"  {lbl}: {cfg.get(k, 'NOT SET')}")
ws = sum(cfg.get(k, 0) for k in ["target_weight_vacancy_gap", "target_weight_recent_rate", "target_weight_demand"])
print(f"  weight sum: {ws:.2f}{'  (need not sum to 1 — signals in HU)' if abs(ws-1)>0.01 else ''}")

# ── Step 4: run developer ──────────────────────────────────────────────────────
print(f"\n{SEP}\nSTEP 4 — Residential developer\n{SEP}")
b0 = orca.get_table("buildings").to_frame(["parcel_id", "year_built", "residential_units", "building_type_id"])
u0 = b0["residential_units"].sum()

orca.run(["residential_developer"], iter_vars=[YR])

b1  = orca.get_table("buildings").to_frame(["parcel_id", "year_built", "residential_units", "building_type_id"])
nb  = b1[b1["year_built"] == YR]
u1  = b1["residential_units"].sum()

# ── Results ────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\nRESULTS\n{SEP}")
print(f"  Buildings added: {len(nb):,}  |  Units added: {nb['residential_units'].sum():,.0f}")
print(f"  Total before: {u0:,.0f}  →  after: {u1:,.0f}")

if len(nb) > 0:
    pcl = orca.get_table("parcels").to_frame(["semmcd", "large_area_id", "land_use_type_id", "parcel_sqft"])
    nw  = nb.join(pcl, on="parcel_id", how="left")

    def _agg(grp_col):
        return (nw.groupby(grp_col)["residential_units"]
                  .agg(units="sum", buildings="count")
                  .sort_values("units", ascending=False))

    print(f"\n── by large_area\n{_agg('large_area_id').to_string()}")

    lut_s = _agg("land_use_type_id")
    lut_n = orca.get_table("land_use_types").to_frame()
    if "name" in lut_n.columns:
        lut_s = lut_s.join(lut_n["name"], how="left")
    print(f"\n── by land_use_type\n{lut_s.to_string()}")

    print(f"\n── top 10 MCDs\n{_agg('semmcd').head(10).to_string()}")
    print(f"\n── parcel_sqft\n{nw['parcel_sqft'].describe().apply(lambda x: f'{x:,.0f}')}")
    print(f"\n── building_type\n{nw['building_type_id'].value_counts().head(10).to_string()}")
else:
    print("  No buildings added.")

# ── Target diagnostics ─────────────────────────────────────────────────────────
print(f"\n{SEP}\nTARGET DIAGNOSTICS\n{SEP}")
dbg = orca.get_table("debug_res_developer").to_frame()
if dbg.empty:
    print("  debug table empty.")
else:
    d = dbg[dbg["year"] == YR].copy()
    print(f"  MCDs: {len(d):,}  target: {d['target_units'].sum():,.0f}  added: {d['units_added'].sum():,.0f}")

    lo  = d[d["recent_rate"] < 30]
    hi  = d[d["recent_rate"] >= 30]
    c_hi = hi[hi["target_units"] >= hi["recent_rate"] * 1.09]
    c_lo = hi[hi["target_units"] <= hi["recent_rate"] * 0.91]
    print(f"\n── MCD cap: low-growth (<30): {len(lo):,} | normal: {len(hi):,} | upper-capped: {len(c_hi):,} | lower-capped: {len(c_lo):,} | in-band: {len(hi)-len(c_hi)-len(c_lo):,}")

    pcl_la  = orca.get_table("parcels").to_frame(["semmcd", "large_area_id"])
    d["la"] = d["mcd"].map(pcl_la.groupby("semmcd")["large_area_id"].first())
    lac = (d.groupby("la").agg(
        tgt=("target_units","sum"), added=("units_added","sum"),
        rate=("recent_rate","sum"), scale=("la_scale","first")
    ).round(1))
    lac["tgt_vs_rate%"] = ((lac["tgt"].astype(float) / lac["rate"].astype(float).clip(lower=1) - 1)*100).round(1)
    print(f"\n── LA alignment\n{lac.to_string()}")

    print(f"\n── Signals (region): A={d['A_units'].sum():+,.0f}  B={d['B_units'].sum():,.0f}  C={d['C_units'].sum():,.0f}  raw={d['target_raw'].sum():,.0f}  capped={d['target_units'].sum():,.0f}  built={d['units_added'].sum():,.0f}")

    bh = orca.get_table("buildings").to_frame(["year_built", "residential_units"])
    rr = bh[bh["year_built"] >= YR - 7]["residential_units"].sum() / 7.0
    print(f"\n── Regional: 7yr trend={rr:,.0f}/yr  built={d['units_added'].sum():,.0f}  ratio={d['units_added'].sum()/max(rr,1):.2f}x")
