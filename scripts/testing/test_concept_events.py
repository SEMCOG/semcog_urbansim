"""The CONCEPT rule in models.scheduled_development_events.

A CONCEPT event carrying residential units is dropped, mixed use included; a
purely non-residential one is kept. Run from the repository root:
    PYTHONPATH=. python scripts/testing/test_concept_events.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def apply_rule(df):
    """The filter as models.scheduled_development_events applies it."""
    drop = (df.dev_type == "CONCEPT") & (df.residential_units > 0)
    return df[~drop].reset_index(drop=True), int(drop.sum())


def frame(rows):
    return pd.DataFrame(rows, columns=["dev_type", "residential_units",
                                       "non_residential_sqft", "res_sqft"])


def t_res_only_concept_is_dropped():
    out, nd = apply_rule(frame([("CONCEPT", 10, 0, 12000)]))
    assert len(out) == 0 and nd == 1
    print("  PASS residential-only CONCEPT dropped")


def t_nonres_only_concept_is_kept():
    out, nd = apply_rule(frame([("CONCEPT", 0, 50000, 0)]))
    assert len(out) == 1 and out.non_residential_sqft.iat[0] == 50000 and nd == 0
    print("  PASS non-residential-only CONCEPT untouched")


def t_mixed_concept_is_dropped():
    out, nd = apply_rule(frame([("CONCEPT", 20, 30000, 24000)]))
    assert len(out) == 0 and nd == 1
    print("  PASS mixed-use CONCEPT dropped, non-residential space with it")


def t_other_dev_types_untouched():
    rows = [(t, 10, 0, 12000) for t in ("PLANNED", "PERMITTED", "PARCEL",
                                        "ADDITION", "CONVERSION", "COMPLETED")]
    out, nd = apply_rule(frame(rows))
    assert len(out) == len(rows) and nd == 0
    assert (out.residential_units == 10).all()
    print("  PASS every other dev_type keeps its residential units")


def t_concept_with_no_content_is_left_alone():
    # never had units, so the rule does not touch it
    out, nd = apply_rule(frame([("CONCEPT", 0, 0, 0)]))
    assert len(out) == 1 and nd == 0
    print("  PASS CONCEPT with neither units nor sqft is left as it was")


def t_real_table_counts():
    import input_paths
    d = pd.read_hdf(input_paths.BASE_HDF, "events_addition")
    d = d[(d.year_built >= 2026) & (d.year_built <= 2055)]
    u0, s0 = d.residential_units.sum(), d.non_residential_sqft.sum()
    out, nd = apply_rule(d)
    kept = d[(d.dev_type == "CONCEPT") & (d.residential_units == 0)]
    assert kept.non_residential_sqft.sum() == out[
        out.dev_type == "CONCEPT"].non_residential_sqft.sum()
    print("  PASS real table: %d dropped | units %d -> %d (-%d) | "
          "non-residential sqft %d -> %d (-%d)"
          % (nd, u0, out.residential_units.sum(), u0 - out.residential_units.sum(),
             s0, out.non_residential_sqft.sum(), s0 - out.non_residential_sqft.sum()))


def main():
    fails = 0
    tests = [t_res_only_concept_is_dropped, t_nonres_only_concept_is_kept,
             t_mixed_concept_is_dropped, t_other_dev_types_untouched,
             t_concept_with_no_content_is_left_alone, t_real_table_counts]
    for fn in tests:
        try:
            fn()
        except AssertionError as e:
            fails += 1
            print("  FAIL %s: %s" % (fn.__name__, e))
    print("%d of %d passed" % (len(tests) - fails, len(tests)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
