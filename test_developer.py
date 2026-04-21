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

# ── Test year config ───────────────────────────────────────────────────────────
# Set YR=2021 for base-year test; set YR=2050 + CHECKPOINT to load late-sim state
YR              = 2030
CHECKPOINT_H5   = "runs/run1386.h5"   # HDF to load state from (set None for base year)
CHECKPOINT_YEAR = 2029                 # year snapshot to load (YR - 1)

# ── run1380 reference data (from res_developer_run1380.log) ────────────────────
RUN1380_REF = {
    2030: {
        "region": {"built": 5_753, "rate_7yr": 17_108},
        "la": {
            3:   {"built": 1_221, "rate_7yr": 2_300},
            5:   {"built":   160, "rate_7yr":   952},
            93:  {"built":   974, "rate_7yr": 2_494},
            99:  {"built":   748, "rate_7yr": 2_841},
            115: {"built":   140, "rate_7yr":   721},
            125: {"built":   717, "rate_7yr": 4_628},
            147: {"built":    26, "rate_7yr":   377},
            161: {"built": 1_767, "rate_7yr": 2_796},
        },
    },
}
pd.set_option("display.float_format", "{:,.1f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)

SEP = "=" * 70

# ── Load checkpoint state ──────────────────────────────────────────────────────
if CHECKPOINT_H5:
    print(f"\n{SEP}\nLOADING CHECKPOINT {CHECKPOINT_H5} @ year {CHECKPOINT_YEAR}\n{SEP}")
    _ck = pd.HDFStore(CHECKPOINT_H5, "r")
    for _tbl in ["buildings", "households", "jobs", "parcels", "persons"]:
        _key = f"/{CHECKPOINT_YEAR}/{_tbl}"
        if _key in _ck:
            orca.add_table(_tbl, _ck[_key])
            print(f"  loaded {_key}  ({len(_ck[_key]):,} rows)")
        else:
            print(f"  WARNING: {_key} not found — using base data")
    _ck.close()

# ── Steps 1-3 ─────────────────────────────────────────────────────────────────
for label, step in [
    ("STEP 1 — Build networks",       "build_networks_2050"),
    ("STEP 2 — Neighborhood vars",    "neighborhood_vars"),
    ("STEP 3 — Feasibility",          "feasibility"),
]:
    print(f"\n{SEP}\n{label}\n{SEP}")
    orca.run([step], iter_vars=[YR])

# ── Detroit parcel filter diagnostic ──────────────────────────────────────────
print(f"\n{SEP}\nDETROIT PARCEL FILTER DIAGNOSTIC\n{SEP}")
import variables
_pcl_all = orca.get_table("parcels").to_frame(["large_area_id", "parcel_sqft", "pct_undev", "bldgimprval", "landvalue"])
_la5 = _pcl_all[_pcl_all.large_area_id == 5]
print(f"LA5 parcels in orca: {len(_la5):,}")

# parcel_size = parcel_sqft * (1 - pct_undev/100)
_ps = _la5.parcel_sqft * (1 - _la5.pct_undev.clip(0,100)/100)
print(f"  parcel_size > 0:     {(_ps>0).sum():,}")

# highval
_hv = _la5.bldgimprval > (_la5.landvalue / 10)
print(f"  NOT highval:         {(~_hv).sum():,}")

# parcel_is_allowed_2050("residential") for LA5
_allowed_res = variables.parcel_is_allowed_2050("residential")
_la5_allowed = _allowed_res.reindex(_la5.index, fill_value=False)
print(f"  parcel_is_allowed:   {_la5_allowed.sum():,}")

# after proforma parcel_filter
_pf_mask = _la5.parcel_sqft < 1_100_000
print(f"  parcel_sqft<1.1M:    {_pf_mask.sum():,}")

# combined: allowed + parcel_filter + parcel_size>0
_all_filt = _la5_allowed & _pf_mask & (_ps > 0).reindex(_la5.index, fill_value=False)
print(f"  All filters combined:{_all_filt.sum():,}")
print()

# ── Feasibility summary ────────────────────────────────────────────────────────
feas = orca.get_table("feasibility").to_frame()
rf   = feas[feas["form"] == "residential"].copy()
rf["profit"] = rf["building_revenue"] - rf["total_cost"]
aus = orca.get_table("parcels").to_frame(["ave_unit_size", "semmcd", "large_area_id"])
rf["residential_units"] = (rf["residential_sqft"] / rf.index.map(aus["ave_unit_size"]).fillna(1000)).clip(lower=0).round()
rf["semmcd"]        = rf.index.map(aus["semmcd"])
rf["large_area_id"] = rf.index.map(aus["large_area_id"])
best = rf.reset_index().sort_values("profit", ascending=False).drop_duplicates("parcel_id").set_index("parcel_id")
n_prof = (best["profit"] > 0).sum()
u_prof = best.loc[best["profit"] > 0, "residential_units"].sum()
print(f"\nFeasibility: {len(feas):,} rows | res parcels: {len(best):,} | profitable: {n_prof:,} | units: {u_prof:,.0f}")

# profitable feasible units by LA
prof = best[best["profit"] > 0]
print(f"\n── Profitable proposals by large_area ──")
la_feas = prof.groupby("large_area_id").agg(parcels=("profit","count"), units=("residential_units","sum"), avg_profit=("profit","mean"))
print(la_feas.round(0).to_string())

# MCDs with zero feasible
mcd_feas = prof.groupby("semmcd")["residential_units"].sum()
n_zero_mcd = (orca.get_table("parcels").to_frame(["semmcd"])["semmcd"].unique().shape[0]
              - mcd_feas[mcd_feas > 0].count())
print(f"\n── MCDs with 0 profitable proposals: {n_zero_mcd:,}")

# vacancy rate vs target
print(f"\n── Actual vacancy rate by large_area ──")
bldgs = orca.get_table("buildings").to_frame(["residential_units", "large_area_id"])
hh    = orca.get_table("households").to_frame(["large_area_id"])
la_units = bldgs.groupby("large_area_id")["residential_units"].sum()
la_hh    = hh.groupby("large_area_id").size()
la_vac   = pd.DataFrame({"units": la_units, "hh": la_hh})
la_vac["vac_rate"] = (1 - la_vac["hh"] / la_vac["units"].clip(lower=1)).round(3)
print(la_vac.to_string())

# pct_undev distribution — land availability
print(f"\n── pct_undev distribution (0=fully developed, 100=greenfield) ──")
pcl_dev = orca.get_table("parcels").to_frame(["pct_undev", "large_area_id"])
print(pcl_dev["pct_undev"].describe().round(1).to_string())
print(f"\n── pct_undev == 100 (greenfield) by LA ──")
print(pcl_dev[pcl_dev["pct_undev"] == 100].groupby("large_area_id").size().to_string())

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
    ("target_weight_demand",       "w_demand"),
    ("la_max_ratio",               "la_max_ratio"),
    ("min_unit_size",              "min_unit_size"),
    ("max_parcel_size",            "max_parcel_size"),
    ("drop_after_build",           "drop_after_build"),
    ("keep_suboptimal",            "keep_suboptimal"),
    ("noise_scale",                "noise_scale"),
]:
    print(f"  {lbl}: {cfg.get(k, 'NOT SET')}")
ws = sum(cfg.get(k, 0) for k in ["target_weight_vacancy_gap", "target_weight_demand"])
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

    la_built = _agg('large_area_id')
    print(f"\n── by large_area\n{la_built.to_string()}")

    # ── Comparison vs run1380 ──────────────────────────────────────────────────
    if YR in RUN1380_REF:
        ref_la  = RUN1380_REF[YR]["la"]
        ref_reg = RUN1380_REF[YR]["region"]
        cur_reg = int(nb["residential_units"].sum())
        print(f"\n{SEP}\nCOMPARISON vs run1380 year {YR}\n{SEP}")
        hdr = f"  {'LA':>4}  {'cur_built':>10}  {'ref_built':>10}  {'diff':>8}  {'diff%':>7}  {'ref_rate7':>10}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for la_id in sorted(ref_la):
            cur = int(la_built.loc[la_id, "units"]) if la_id in la_built.index else 0
            ref = ref_la[la_id]["built"]
            diff = cur - ref
            pct  = diff / ref * 100 if ref else float("nan")
            rate = ref_la[la_id]["rate_7yr"]
            print(f"  {la_id:>4}  {cur:>10,}  {ref:>10,}  {diff:>+8,}  {pct:>+6.1f}%  {rate:>10,}")
        ref_tot = ref_reg["built"]
        diff_r  = cur_reg - ref_tot
        pct_r   = diff_r / ref_tot * 100 if ref_tot else float("nan")
        print(f"  {'TOT':>4}  {cur_reg:>10,}  {ref_tot:>10,}  {diff_r:>+8,}  {pct_r:>+6.1f}%  {ref_reg['rate_7yr']:>10,}")

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

    pcl_la  = orca.get_table("parcels").to_frame(["semmcd", "large_area_id"])
    d["la"] = d["mcd"].map(pcl_la.groupby("semmcd")["large_area_id"].first())
    lac = (d.groupby("la").agg(
        tgt=("target_units","sum"), added=("units_added","sum"),
        scale=("la_scale","first")
    ).round(1))
    print(f"\n── LA alignment\n{lac.to_string()}")

    print(f"\n── Signals (region): V={d['V_units'].sum():+,.0f}  R={d['R_units'].sum():,.0f}  raw={d['target_raw'].sum():,.0f}  capped={d['target_units'].sum():,.0f}  built={d['units_added'].sum():,.0f}")

    bh = orca.get_table("buildings").to_frame(["year_built", "residential_units"])
    rr = bh[bh["year_built"] >= YR - 7]["residential_units"].sum() / 7.0
    print(f"\n── Regional: 7yr trend={rr:,.0f}/yr  built={d['units_added'].sum():,.0f}  ratio={d['units_added'].sum()/max(rr,1):.2f}x")

    # ── Detroit / LA 5 deep dive ───────────────────────────────────────────────
    det_mcds = d[d["la"] == 5]
    if not det_mcds.empty:
        print(f"\n── Detroit (LA 5) MCD detail ──")
        cols = ["mcd", "V_units", "R_units", "target_raw",
                "target_units", "units_added"]
        cols = [c for c in cols if c in det_mcds.columns]
        print(det_mcds[cols].to_string(index=False))
        # feasibility price check for LA 5 parcels
        bldg_la5 = orca.get_table("buildings").to_frame(["large_area_id", "sqft_price_res"])
        la5_prices = bldg_la5[bldg_la5["large_area_id"] == 5]["sqft_price_res"]
        la5_prices = la5_prices[la5_prices > 0]
        if len(la5_prices):
            print(f"  LA 5 sqft_price_res: n={len(la5_prices):,}  "
                  f"mean=${la5_prices.mean():.0f}  median=${la5_prices.median():.0f}  "
                  f"p10=${la5_prices.quantile(0.1):.0f}  p90=${la5_prices.quantile(0.9):.0f}")
