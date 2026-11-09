"""Exercise output_indicators.combined_summary against a finished run's workbooks.

The per-geography `*_by_indicator_for_year.xlsx` files hold exactly what
`dict_ind` holds in main(), so they stand in for it without rerunning the
indicators. Usage: python -m scripts.testing.test_combined_summary [runNNN]
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from output_indicators import (SECTOR_TO_PUBLISHED, SUMMARY_GEO_COLS, SUMMARY_AGE,
                               REGION_ID,
                               combined_summary)

RUN = sys.argv[1] if len(sys.argv) > 1 else "run1518"
BOOKS = {"cities": "cities", "zones": "zones", "schools": "schools",
         "mi_house": "mi_house", "mi_senate": "mi_senate",
         "us_congress": "us_congress"}


def load(run):
    dict_ind, years = {}, None
    for tab, stem in BOOKS.items():
        p = "runs/%s/%s_by_indicator_for_year.xlsx" % (run, stem)
        if not os.path.exists(p):
            print("skip %s (no %s)" % (tab, p))
            continue
        x = pd.ExcelFile(p)
        years = years or x.sheet_names
        dict_ind[tab] = [x.parse(y, index_col=0) for y in years]
    return dict_ind, years


def main():
    dict_ind, years = load(RUN)
    assert dict_ind, "no workbooks found for %s" % RUN
    # cities_by_year_for_indicator carries large_area_id beside city_id
    la = pd.read_excel("runs/%s/cities_by_year_for_indicator.xlsx" % RUN,
                       sheet_name="hh")
    la_of = la.set_index(la.columns[0])[la.columns[1]]

    out = combined_summary(dict_ind, years, set(), la_of)
    ok = True

    print("rows %d | indicators %d | years %s"
          % (len(out), out.index.get_level_values("indicator").nunique(), list(out.columns)))

    # 1. layout
    assert list(out.index.names) == SUMMARY_GEO_COLS + ["indicator"], out.index.names
    assert list(out.columns) == list(years)

    # 2. 13 published sectors, no model sector above 13
    sec = sorted(i for i in out.index.get_level_values("indicator").unique()
                 if i.startswith("jobs_sec_"))
    assert sec == ["jobs_sec_%02d" % i for i in range(1, 14)], sec
    print("sectors: %s .. %s (%d)" % (sec[0], sec[-1], len(sec)))

    # 3. conservation: published sectors sum to the model's 18
    idx = out.reset_index()
    for tab, geo_id in [("cities", "city_id"), ("zones", "zone_id"),
                        ("schools", "school_id"), ("mi_house", "mi_house_id"),
                        ("mi_senate", "mi_senate_id"),
                        ("us_congress", "us_congress_id")]:
        if tab not in dict_ind:
            continue
        f = dict_ind[tab][-1]
        f = f[f.index != -1]
        model = f[["jobs_sec_%02d" % i for i in range(1, 19)]].sum().sum()
        blk = idx[idx[geo_id].notna() & idx.region_id.isna()]
        pub = blk[blk.indicator.isin(sec)][years[-1]].sum()
        flag = "" if abs(model - pub) < 1 else "  <-- MISMATCH"
        ok = ok and not flag
        print("%-12s 18-sector %12.0f | 13-sector %12.0f%s" % (tab, model, pub, flag))

    # 3b. the broad age bands equal their parts, in every block and the region row
    for band, parts in SUMMARY_AGE.items():
        a = idx[idx.indicator == band][years[-1]].sum()
        b = idx[idx.indicator.isin(parts)][years[-1]].sum()
        flag = "" if abs(a - b) < 1 else "  <-- MISMATCH"
        ok = ok and not flag
        print("%-15s %12.0f | %-29s %12.0f%s"
              % (band, a, " + ".join(parts), b, flag))
    pop = idx[idx.indicator == "pop"][years[-1]].sum()
    allband = idx[idx.indicator.isin(SUMMARY_AGE)][years[-1]].sum()
    flag = "" if abs(pop - allband) < 1 else "  <-- MISMATCH"
    ok = ok and not flag
    print("%-15s %12.0f | %-29s %12.0f%s" % ("pop", pop, "the three bands", allband, flag))

    # 4. one geography level per row, cities carrying large_area_id too
    n = idx[SUMMARY_GEO_COLS].notna().sum(axis=1)
    print("populated geo ids per row:", dict(n.value_counts().sort_index()))

    # 5. region row
    reg = idx[idx.region_id == REGION_ID]
    city = idx[idx.city_id.notna()]
    for ind in ["hh", "pop", "jobs_total"]:
        a = reg[reg.indicator == ind][years[-1]].iat[0]
        b = city[city.indicator == ind][years[-1]].sum()
        flag = "" if abs(a - b) < 1 else "  <-- MISMATCH"
        ok = ok and not flag
        print("region %-10s %12.0f | cities %12.0f%s" % (ind, a, b, flag))

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
