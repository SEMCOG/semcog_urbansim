"""HLCM city factors: placements needed for the observed relative net change.

For each city (model geography), target net household change in the first simulated
years = the large area's simulated change + the city's observed 2020-25 growth above its
large area (applied as a rate to the 2025 households). Needed placements = target net
change + the households the transition removed there. Factor = needed / simulated
placements, limited to [lo, hi], 1 for small cities, then normalized so the
placement-weighted mean is 1 in each large area.

Calibration loop (model review, not estimation):
  1. Run the model for the calibration window (e.g. 2026-2030) with the current factors
     (models.hlcm.calibration.city_factors in configs/model_structure.yaml), or with the
     setting empty for a first iteration.
  2. python scripts/estimation/make_city_factors.py --runs <run h5>[,<run h5>...] \
         --first-year 2026 --last-year 2030 --hlcm-model <HLCM model folder name> \
         [--previous configs/hlcm_city_factors.csv] [--out configs/hlcm_city_factors.csv]
     With --previous (iteration), the new factor = previous x adjustment. Several runs (same
     factors, other seeds): end households, placements and removals are the mean of the runs.
  3. Repeat until the change is within the seed noise.
The output keeps the diagnostic columns; the model reads city_id, factor and hlcm_model.
Run from the repo root with PYTHONPATH=.
"""
import argparse
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HH_2020 = os.path.join(REPO, "configs", "households_by_city_2020.csv")
OUT = os.path.join(REPO, "configs", "hlcm_city_factors.csv")
MIN_HH = 500          # smaller cities keep factor 1 (noisy)
MIN_PLACED = 50       # and cities with few simulated placements
SKIP_LA = {5}         # Detroit: 2020-25 neighborhood change mixes in the new base placement

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("--runs", required=True, help="run h5 file(s), comma separated")
p.add_argument("--first-year", type=int, required=True)
p.add_argument("--last-year", type=int, required=True)
p.add_argument("--hlcm-model", required=True, help="HLCM model folder name the runs used")
p.add_argument("--previous", help="previous factors csv (iteration)")
p.add_argument("--out", default=OUT)
p.add_argument("--lo", type=float, default=0.4)
p.add_argument("--hi", type=float, default=3.0)
# step on the log scale per iteration. Placements respond to a factor f as about f^0.58,
# so 0.8 / 0.58 = 1.38 closes about 80% of the gap per iteration
p.add_argument("--step", type=float, default=1.38)
# robust targets: cap the relative 2020-25 change at these percentiles of all cities, then
# shrink it toward 0 by size, hh2020 / (hh2020 + K) (0 = off)
p.add_argument("--cap-pct", type=float, default=5)
p.add_argument("--shrink-k", type=float, default=2000)
a = p.parse_args()
y0, y1 = a.first_year, a.last_year
prev = pd.read_csv(a.previous, index_col="city_id").factor if a.previous else None


def read_run(path):
    with pd.HDFStore(path, "r") as s:
        hh = {}
        placed, removed = pd.Series(0.0), pd.Series(0.0)
        prev_bid = s["/base/households"].building_id
        prev_city = s["/base/buildings"].city_id
        hh[2025] = prev_city.reindex(prev_bid).value_counts()
        la = s["/base/households"].groupby(prev_city.reindex(s["/base/households"].building_id).to_numpy()).large_area_id.first()
        for y in range(2026, y1 + 1):
            h = s["/%d/households" % y].building_id
            city = s["/%d/buildings" % y].city_id
            if y >= y0:
                new_or_moved = ~h.index.isin(prev_bid.index) | (prev_bid.reindex(h.index).to_numpy() != h.to_numpy())
                placed = placed.add(city.reindex(h[new_or_moved]).value_counts(), fill_value=0)
                gone = ~prev_bid.index.isin(h.index) | (h.reindex(prev_bid.index).to_numpy() != prev_bid.to_numpy())
                removed = removed.add(prev_city.reindex(prev_bid[gone]).value_counts(), fill_value=0)
            prev_bid, prev_city = h, city
        hh[y1] = prev_city.reindex(prev_bid).value_counts()
    return pd.DataFrame({"la": la, "hh2025": hh[2025], "hh_end": hh[y1], "placed": placed, "removed": removed})


runs = [read_run(r) for r in a.runs.split(",")]
d = runs[0][["la", "hh2025"]].copy()
for c in ("hh_end", "placed", "removed"):
    d[c] = pd.concat([r[c] for r in runs], axis=1).fillna(0).mean(axis=1)
h20 = pd.read_csv(HH_2020, index_col="city_id").households
d.insert(1, "hh2020", h20.reindex(d.index))
d = d.dropna(subset=["la", "hh2025"]).fillna(0)
d = d[d.index > 0]
d["la"] = d.la.astype(int)
g = d.groupby("la")[["hh2020", "hh2025", "hh_end"]].sum()
obs_rel = (d.hh2025 / d.hh2020.replace(0, np.nan) - 1) - d.la.map(g.hh2025 / g.hh2020 - 1)
if a.cap_pct > 0:
    ref = obs_rel[~d.la.isin(SKIP_LA)].dropna()
    obs_rel = obs_rel.clip(np.percentile(ref, a.cap_pct), np.percentile(ref, 100 - a.cap_pct))
if a.shrink_k > 0:
    obs_rel = obs_rel * d.hh2020 / (d.hh2020 + a.shrink_k)
sim_la = d.la.map(g.hh_end / g.hh2025 - 1)
years_obs, years_sim = 5, y1 - 2025
d["target_net"] = d.hh2025 * (sim_la + obs_rel.fillna(0) * years_sim / years_obs)
d["sim_net"] = d.hh_end - d.hh2025
d["needed"] = (d.target_net + d.removed).clip(lower=0)
adj = (d.needed / d.placed.replace(0, np.nan)).fillna(1.0)
ok = (d.hh2025 >= MIN_HH) & (d.placed >= MIN_PLACED) & ~d.la.isin(SKIP_LA)
adj = adj.where(ok, 1.0) ** a.step
f = adj if prev is None else adj * prev.reindex(d.index).fillna(1.0)
f = f.clip(a.lo, a.hi)
# placement-weighted mean 1 in each large area (keeps the large area's overall pull)
f = f.where(~d.la.isin(SKIP_LA), 1.0)
f = f / (f * d.placed).groupby(d.la).transform("sum") * d.placed.groupby(d.la).transform("sum")
d["factor"] = f.clip(a.lo, a.hi).round(4)
d["hlcm_model"] = a.hlcm_model
d.index.name = "city_id"
d = d[["factor", "hlcm_model"] + [c for c in d.columns if c not in ("factor", "hlcm_model")]]
d.to_csv(a.out)
print("wrote %s | cities %d | factor != 1: %d | at bounds: %d | median %.3f" % (
    a.out, len(d), int((d.factor != 1).sum()), int(((d.factor <= a.lo) | (d.factor >= a.hi)).sum()),
    d.factor.median()))
print("net change error |sim - target| summed: %d households (of %d target movement)" % (
    (d.sim_net - d.target_net).abs().sum(), d.target_net.abs().sum()))
