"""Split each MCD's household-population change into its causes.

hh_pop = households x type mix x persons per type, so a change in it is a
change in one of the three. The controls fix the region's (and each large
area's) mix and per-type sizes exactly, which leaves only two terms the model
decides: mcd_share, how an MCD's household count moves against its large
area's, and mix_loc, how its type mix moves against its large area's.

Terms, which sum to the change exactly:
    reg_growth  region household growth at the MCD's base mix and sizes
    la_growth   its large area's growth net of the region's
    mcd_share   its own household count net of its large area's   <- model
    mix_reg     the region's type-mix shift on its base mix       <- controls
    mix_la      its large area's mix shift net of the region's    <- controls
    mix_loc     its own mix net of its large area's               <- model
    size_reg    region per-type size change                       <- controls
    size_loc    its own per-type sizes net of the region's        <- model

Types are the HLCM segmentation: children x senior x income tercile.

Usage, from the repository root:
    PYTHONPATH=. python scripts/mcd_hh_pop_decomposition.py run1519 [run1518 ...]
    PYTHONPATH=. python scripts/mcd_hh_pop_decomposition.py --year 2040 run1519
Writes <run dir>/mcd_hh_pop_decomposition.csv per run.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import input_paths

MODEL = ['mcd_share', 'mix_loc', 'size_loc']
TERMS = ['reg_growth', 'la_growth', 'mcd_share', 'mix_reg', 'mix_la', 'mix_loc',
         'size_reg', 'size_loc']


def cells(store, year, par):
    k = '/base/' if year == 'base' else '/%s/' % year
    b = store[k + 'buildings'][['parcel_id']].join(par, on='parcel_id')
    h = store[k + 'households'][['building_id', 'income', 'children', 'persons']]
    p = store[k + 'persons'][['household_id', 'age']]
    sen = p[p.age >= 65].groupby('household_id').size().reindex(h.index).fillna(0)
    h['cell'] = ('k' + (h.children > 0).astype(int).astype(str)
                 + 's' + (sen > 0).astype(int).astype(str)
                 + 'i' + (pd.qcut(h.income, 3, labels=False) + 1).astype(str))
    h['mcd'] = h.building_id.map(b.semmcd)
    h['la'] = h.building_id.map(b.large_area_id)
    h = h[h.mcd.notna()]
    g = h.groupby(['mcd', 'cell']).agg(n=('persons', 'size'), pop=('persons', 'sum'))
    return g, h.groupby('mcd').la.first()


def decompose(run, year, names):
    with pd.HDFStore(os.path.join('runs', '%s.h5' % run), 'r') as s:
        par = s['/base/parcels'][['semmcd', 'large_area_id']]
        g0, la = cells(s, 'base', par)
        g1, _ = cells(s, year, par)
    cols = sorted(set(g0.index.get_level_values('cell'))
                  | set(g1.index.get_level_values('cell')))
    wide = lambda g, f: g[f].unstack(fill_value=0).reindex(columns=cols, fill_value=0)
    n0, p0, n1, p1 = wide(g0, 'n'), wide(g0, 'pop'), wide(g1, 'n'), wide(g1, 'pop')
    ix = n0.index.intersection(n1.index)
    n0, p0, n1, p1 = n0.loc[ix], p0.loc[ix], n1.loc[ix], p1.loc[ix]
    la = la.reindex(ix)
    N0, N1, P0, P1 = n0.sum(axis=1), n1.sum(axis=1), p0.sum(axis=1), p1.sum(axis=1)
    s0, s1 = n0.div(N0, axis=0), n1.div(N1, axis=0)
    # a cell empty in an MCD takes the regional size, so m is defined everywhere
    mr0 = (p0.sum() / n0.sum()).reindex(cols)
    mr1 = (p1.sum() / n1.sum()).reindex(cols)
    m0 = (p0 / n0.replace(0, np.nan)).fillna(pd.Series(mr0))
    m1 = (p1 / n1.replace(0, np.nan)).fillna(pd.Series(mr1))
    tilt = lambda a, b: (a / b.replace(0, np.nan)).fillna(1)
    nrm = lambda x: x.div(x.sum(axis=1), axis=0)
    # the MCD's base mix carried by the region's shift, then by its own LA's
    cf_r = nrm(s0.mul(tilt(n1.sum() / n1.sum().sum(), n0.sum() / n0.sum().sum()), axis=1))
    l0, l1 = n0.groupby(la).sum(), n1.groupby(la).sum()
    tl = tilt(nrm(l1), nrm(l0)).reindex(la.values)
    tl.index = ix
    cf_l = nrm(s0 * tl)
    gr = N1.sum() / N0.sum()
    gl = pd.Series(la.map(N1.groupby(la).sum() / N0.groupby(la).sum()).values, index=ix)
    sz_r = N1 * (s1.mul(pd.Series(mr1 - mr0), axis=1)).sum(axis=1)
    out = pd.DataFrame(dict(
        name=names.reindex(ix), large_area_id=la, hh_base=N0, hh_end=N1,
        pop_base=P0, pop_end=P1, d_pop=P1 - P0,
        reg_growth=P0 * (gr - 1), la_growth=P0 * (gl - gr),
        mcd_share=P0 * (N1 / N0 - gl),
        mix_reg=N1 * ((cf_r - s0) * m0).sum(axis=1),
        mix_la=N1 * ((cf_l - cf_r) * m0).sum(axis=1),
        mix_loc=N1 * ((s1 - cf_l) * m0).sum(axis=1),
        size_reg=sz_r, size_loc=N1 * (s1 * (m1 - m0)).sum(axis=1) - sz_r,
        size_base=P0 / N0, size_end=P1 / N1))
    out['residual'] = out.d_pop - out[TERMS].sum(axis=1)
    return out.sort_values('pop_base', ascending=False)


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('runs', nargs='+')
    a.add_argument('--year', default='2055')
    a.add_argument('--top', type=int, default=15)
    o = a.parse_args(argv)
    names = pd.read_hdf(input_paths.BASE_HDF, 'semmcds').city_name
    for run in o.runs:
        t = decompose(run, o.year, names)
        assert t.residual.abs().max() < 1.0, t.residual.abs().max()
        d = os.path.join('runs', run)
        if os.path.isdir(d):
            t.to_csv(os.path.join(d, 'mcd_hh_pop_decomposition.csv'))
        print('=== %s, base -> %s, %d MCDs ===' % (run, o.year, len(t)))
        print('hh %d -> %d (%+.2f%%) | hh_pop %d -> %d (%+.2f%%) | size %.3f -> %.3f'
              % (t.hh_base.sum(), t.hh_end.sum(), 100 * (t.hh_end.sum() / t.hh_base.sum() - 1),
                 t.pop_base.sum(), t.pop_end.sum(), 100 * (t.pop_end.sum() / t.pop_base.sum() - 1),
                 t.pop_base.sum() / t.hh_base.sum(), t.pop_end.sum() / t.hh_end.sum()))
        print('totals  ' + '  '.join('%s %+.0f' % (k, t[k].sum()) for k in TERMS))
        print('spread (sum |value|, the model terms first)  '
              + '  '.join('%s %.0f' % (k, t[k].abs().sum()) for k in MODEL + ['mix_reg']))
        print('\n%d largest |mcd_share|:' % o.top)
        c = ['name', 'large_area_id', 'hh_base', 'hh_end', 'd_pop'] + TERMS
        print(t.reindex(t.mcd_share.abs().sort_values(ascending=False).index)
              .head(o.top)[c].round(0).to_string())
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
