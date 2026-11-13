"""Base-year calibration targets kept in the run h5, so a resumed run uses the base year's.

models.init_taz_hlcm_trend_by_year builds these from the households, buildings and jobs in
memory. On a resume those are the checkpoint year's, and the HLCM tract calibrator and fill
prior would pull toward the resume-year mix. A run therefore saves the base-year targets
(group /base_year_targets) and a resume reads them back
(assumptions.update_store_from_checkpoint).
"""
import pandas as pd

GROUP = "base_year_targets"
TABLES = ("tract_hh_type_base_ratios",)
INJECTABLES = ("new_unit_fill_shares", "new_unit_fill_ratios", "sqft_band_cuts",
               "job_btype_baseyear_prob_matrix")
NAMES = TABLES + INJECTABLES


def save(path, targets):
    """Write the targets (dict name -> value, NAMES) to /base_year_targets in path."""
    with pd.HDFStore(path, "a") as st:
        for k in NAMES:
            v = targets[k]
            if k == "sqft_band_cuts":
                # dict large_area_id -> cut points; a large area without cuts reads back
                # as missing, which assign_sqft_band treats the same (band 0)
                v = pd.Series({(int(la), i): float(c) for la, cuts in v.items()
                               for i, c in enumerate(cuts)}, dtype="float64")
            st["%s/%s" % (GROUP, k)] = v


def read(store):
    """Targets saved in store, or None when the store has none (a run written before them)."""
    have = {k.split("/")[-1] for k in store.keys() if k.startswith("/%s/" % GROUP)}
    if not have:
        return None
    missing = sorted(set(NAMES) - have)
    if missing:
        raise ValueError("%s: incomplete /%s, missing %s" % (getattr(store, "filename", store), GROUP, missing))
    out = {k: store["%s/%s" % (GROUP, k)] for k in NAMES}
    cuts = out["sqft_band_cuts"]
    out["sqft_band_cuts"] = {int(la): g.droplevel(0).sort_index().to_numpy()
                             for la, g in cuts.groupby(level=0)}
    return out
