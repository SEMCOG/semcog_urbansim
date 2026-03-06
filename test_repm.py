"""
REPM Test Script

Runs only the steps needed to test XGBoost REPM models:
  build_networks_2050 -> neighborhood_vars -> developer steps -> REPM

Skips: household/job transitions, LCMs, scaling, indicators.
Run: nohup python test_repm.py > runs/run_stdout/repm_test_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
"""

import orca
import os
import time
import subprocess
import yaml
import pandas as pd
import utils

# ── Config ────────────────────────────────────────────────────────────────────
base_year = 2020
final_year = 2022       # short run — 2 years is enough to test REPM

# Required by variables_building.py:get_hlcm_segment() at import time
orca.add_injectable('hlcm_model_path', '/mnt/hgfs/RDF2050/estimation/models/models_25Nov13')
orca.add_injectable('elcm_model_path', '/mnt/hgfs/RDF2050/estimation/models/elcm_models_25May30/')
orca.add_injectable('yaml_configs', 'yaml_configs_elcm_hlcm.yaml')

orca.add_injectable('base_year', base_year)
orca.add_injectable('final_year', final_year)
orca.add_injectable('ENABLE_SCENARIO', False)
orca.add_injectable('use_checkpoint', False)

# ── Setup run folder ──────────────────────────────────────────────────────────
data_out = utils.get_run_filename()
orca.add_injectable("data_out_dir", data_out.replace(".h5", ""))
print(data_out)

os.makedirs(orca.get_injectable("data_out_dir"), exist_ok=True)
with open(os.path.join(orca.get_injectable("data_out_dir"), "run_config.yaml"), "w+") as f:
    yaml.dump({
        "test_script": "test_repm.py",
        "RUN NUMBER": data_out,
        "base_year": base_year,
        "final_year": final_year,
        "git_branch_name": subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip(),
        "git_commit_id": subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
    }, f, default_flow_style=False)

# ── Imports (after injectables set) ──────────────────────────────────────────
import models
from urbansim.utils import misc, networks
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    _eastern = ZoneInfo("America/Detroit")
    def _eastern_now():
        return datetime.now(_eastern).strftime("%Y-%m-%d %H:%M:%S %Z")
except ImportError:
    import pytz
    _eastern = pytz.timezone("America/Detroit")
    def _eastern_now():
        return datetime.now(_eastern).strftime("%Y-%m-%d %H:%M:%S %Z")

start_time = time.time()
utils.run_log(f"REPM test start: {_eastern_now()}\ndata_out: {data_out}")

# ── Run ───────────────────────────────────────────────────────────────────────
orca.run(
    [
        "build_networks_2050",
        "neighborhood_vars",
        # Demolition/development needed so new buildings exist to test REPM pricing
        "scheduled_demolition_events",
        "scheduled_development_events",
        "drop_pseudo_buildings",
        "feasibility",
        "residential_developer",
        "non_residential_developer",
    ]
    + orca.get_injectable("repm_step_names"),
    iter_vars=list(range(base_year + 1, final_year + 1)),
    data_out=data_out,
    out_base_tables=["buildings", "parcels", "zones", "semmcds"],
    out_run_tables=["buildings"],
    out_interval=1,
    compress=True,
)

elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
utils.run_log(f"REPM test finished: {_eastern_now()}  Total: {elapsed}")
print(f"Finished at {_eastern_now()}. Total run time: {elapsed}")

# ── Quick sanity check ────────────────────────────────────────────────────────
import numpy as np

store = pd.HDFStore(data_out, "r")
last_year = final_year
try:
    bldgs = store[f"/{last_year}/buildings"]
    res_mask = bldgs.residential_units > 0
    nonres_mask = bldgs.non_residential_sqft > 0

    print("\n── REPM Sanity Check ──────────────────────────────")
    print(f"  Year {last_year} buildings: {len(bldgs):,}")

    for col, mask, label in [
        ("sqft_price_res",    res_mask,    "residential"),
        ("sqft_price_nonres", nonres_mask, "non-residential"),
    ]:
        if col not in bldgs.columns:
            print(f"  WARNING: {col} column missing!")
            continue
        vals = bldgs.loc[mask, col]
        zero_pct = (vals == 0).mean() * 100
        print(f"\n  {label} ({label}):")
        print(f"    count:    {len(vals):,}")
        print(f"    zeros:    {zero_pct:.1f}%  ← should be low for existing bldgs")
        print(f"    mean:     ${vals[vals > 0].mean():.2f}/sqft")
        print(f"    median:   ${vals[vals > 0].median():.2f}/sqft")
        print(f"    p95:      ${vals[vals > 0].quantile(0.95):.2f}/sqft")

    # Check newly developed buildings specifically
    base_bldgs = store["/base/buildings"]
    new_idx = bldgs.index.difference(base_bldgs.index)
    if len(new_idx):
        new_bldgs = bldgs.loc[new_idx]
        print(f"\n  New buildings (added by developer): {len(new_bldgs):,}")
        for col, mask_col, label in [
            ("sqft_price_res",    "residential_units",    "res"),
            ("sqft_price_nonres", "non_residential_sqft", "nonres"),
        ]:
            sub = new_bldgs[new_bldgs[mask_col] > 0]
            if len(sub) and col in sub.columns:
                zero_pct = (sub[col] == 0).mean() * 100
                print(f"    {col}: {len(sub):,} bldgs, {zero_pct:.1f}% zeros  ← should be 0%")
    print("────────────────────────────────────────────────────\n")
except KeyError as e:
    print(f"  Could not load year data for sanity check: {e}")
finally:
    store.close()
