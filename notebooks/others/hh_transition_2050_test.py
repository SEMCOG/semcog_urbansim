# testing hh transition model
import os
import pandas as pd
from urbansim.utils import misc
import orca
import numpy as np, pandas as pd
from urbansim.models import transition, relocation
import operator
from multiprocessing import Pool
from functools import reduce
from memory_profiler import profile


orca.add_injectable("store", pd.HDFStore(
        "~/semcog_urbansim/data/all_semcog_data_02-02-18-final-forecast_newbid.h5",
        mode="r",
    ))


@orca.table('annual_household_control_totals')
def annual_household_control_totals(store):
    df = store['annual_household_control_totals']
    return df

@orca.table('remi_pop_total')
def remi_pop_total(store):
    df = store['remi_pop_total']
    return df

@orca.table('buildings')
def buildings(store):
    df = store['buildings']
    return df

@orca.table('parcels')
def parcels(store):
    df = store['parcels']
    return df

@orca.table(cache=True)
def households(store, buildings):
    df = store["households"]
    b = buildings.to_frame(["large_area_id"])
    b = b[b.large_area_id.isin({161.0, 3.0, 5.0, 125.0, 99.0, 115.0, 147.0, 93.0})]
    df.loc[df.building_id == -1, "building_id"] = np.random.choice(
        b.index.values, (df.building_id == -1).sum()
    )
    idx_invalid_building_id = np.in1d(df.building_id, b.index.values) == False
    df.loc[idx_invalid_building_id, "building_id"] = np.random.choice(
        b.index.values, idx_invalid_building_id.sum()
    )
    df["large_area_id"] = misc.reindex(b.large_area_id, df.building_id)
    return df.fillna(0)
    
@orca.table('persons')
def persons(store):
    df = store['persons']
    return df

@orca.column("buildings", cache=True, cache_scope="iteration")
def large_area_id(buildings, parcels):
    return misc.reindex(parcels.large_area_id, buildings.parcel_id)

@orca.column('persons', cache=True, cache_scope='iteration')
def large_area_id(households, persons):
    return misc.reindex(households.large_area_id, persons.household_id)

@profile
def presses_trans(xxx_todo_changeme1):
    (ct, hh, p, target, iter_var) = xxx_todo_changeme1
    ct_finite = ct[ct.persons_max <= 100]
    ct_inf = ct[ct.persons_max > 100]
    tran = transition.TabularTotalsTransition(ct_finite, "total_number_of_households")
    model = transition.TransitionModel(tran)
    new, added_hh_idx, new_linked = model.transition(
        hh, iter_var, linked_tables={"linked": (p, "household_id")}
    )
    new.loc[added_hh_idx, "building_id"] = -1
    pers = new_linked["linked"]
    pers = pers[pers.household_id.isin(new.index)]
    new.index.name = "household_id"
    pers.index.name = "person_id"
    out = [[new, pers]]
    target -= len(pers)
    best_qal = np.inf
    best = []
    for _ in range(3):
        tran = transition.TabularTotalsTransition(ct_inf, "total_number_of_households")
        model = transition.TransitionModel(tran)
        new, added_hh_idx, new_linked = model.transition(
            hh, iter_var, linked_tables={"linked": (p, "household_id")}
        )
        new.loc[added_hh_idx, "building_id"] = -1
        pers = new_linked["linked"]
        pers = pers[pers.household_id.isin(new.index)]
        qal = abs(target - len(pers))
        if qal < best_qal:
            new.index.name = "household_id"
            pers.index.name = "person_id"
            best = (new, pers)
            best_qal = qal
    out.append(best)
    return out

@profile
def hh_trans(households, persons, annual_household_control_totals, remi_pop_total, iter_var):
    region_ct = annual_household_control_totals.to_frame()
    max_cols = region_ct.columns[
        region_ct.columns.str.endswith("_max") & (region_ct == -1).any(axis=0)
    ]
    region_ct[max_cols] = region_ct[max_cols].replace(-1, np.inf)
    region_ct[max_cols] += 1
    region_hh = households.to_frame(households.local_columns + ["large_area_id"])
    region_hh.index = region_hh.index.astype(int)
    region_p = persons.to_frame(persons.local_columns)
    region_p.index = region_p.index.astype(int)
    region_target = remi_pop_total.to_frame()

    def cut_to_la(xxx_todo_changeme):
        (large_area_id, hh) = xxx_todo_changeme
        p = region_p[region_p.household_id.isin(hh.index)]
        target = int(region_target.loc[large_area_id, str(iter_var)])
        ct = region_ct[region_ct.large_area_id == large_area_id]
        del ct["large_area_id"]
        return ct, hh, p, target, iter_var

    arg_per_la = list(map(cut_to_la, region_hh.groupby("large_area_id")))
    # cunks_per_la = map(presses_trans, arg_per_la)
    pool = Pool(8)
    cunks_per_la = pool.map(presses_trans, arg_per_la)
    pool.close()
    pool.join()
    out = reduce(operator.concat, cunks_per_la)

    # fix indexes
    hhidmax = region_hh.index.values.max() + 1
    pidmax = region_p.index.values.max() + 1

    ## create {old_hh_id => new_hh_id} mapping
    hh_id_mapping = [x[0]['building_id'] for x in out]
    # list of number of hh added for each df in the hh_id_mapping
    hh_new_added = [(x == -1).sum() for x in hh_id_mapping]
    # cumulative sum for hh_new_added, add 0 at front
    hh_new_added_cumsum = [0] + list(np.cumsum(hh_new_added))
    for i, hhmap in enumerate(hh_id_mapping):
        ## create seperate mapping for each df in the list
        hhmap = hhmap.reset_index()
        hhmap["household_id_old"] = hhmap["household_id"]
        # assign hh_id to those newly added
        hhmap.loc[hhmap['building_id'] == -1, "household_id"] = list(
            range(hhidmax + hh_new_added_cumsum[i], hhidmax + hh_new_added_cumsum[i+1]))
        hh_id_mapping[i] = hhmap[["household_id_old", "household_id"]].set_index(
            "household_id_old"
        )
    ## hh df
    # concat all hh dfs and reset their index
    # out_hh = pd.concat([x[0].reset_index() for x in out], verify_integrity=True, ignore_index=True, copy=False)
    out_hh = pd.concat([
        pd.merge(x[0].reset_index(), hh_id_mapping[i], left_on='household_id', right_index=True) for i, x in enumerate(
            out)], verify_integrity=True, ignore_index=True, copy=False)
    # the number of new hh added
    # new_hh = (out_hh.building_id == -1).sum()
    # sort
    out_hh = out_hh.sort_values(by='household_id')
    # assign hh_id to those newly added
    # out_hh.loc[out_hh.building_id == -1, "household_id"] = list(
    #     range(hhidmax, hhidmax + new_hh)
    # )
    # set index to hh_id
    out_hh = out_hh.set_index("household_id_y")
    out_hh = out_hh[households.local_columns]
    out_hh.index.name = 'household_id'
    # persons df
    out_person = pd.concat([
        pd.merge(x[1].reset_index(), hh_id_mapping[i], left_on='household_id', right_index=True) for i, x in enumerate(
            out)], verify_integrity=True, ignore_index=True, copy=False)
    new_p = (out_person.household_id_x != out_person.household_id_y).sum()
    out_person.loc[out_person.household_id_x != out_person.household_id_y, "person_id"] = list(
        range(pidmax, pidmax + new_p)
    )
    out_person["household_id"] = out_person["household_id_y"]
    out_person = out_person.set_index('person_id')

    orca.add_table("households", out_hh[households.local_columns])
    orca.add_table("persons", out_person[persons.local_columns])

def hh_trans_old(
    households, persons, annual_household_control_totals, remi_pop_total, iter_var
):
    region_ct = annual_household_control_totals.to_frame()
    max_cols = region_ct.columns[
        region_ct.columns.str.endswith("_max") & (region_ct == -1).any(axis=0)
    ]
    region_ct[max_cols] = region_ct[max_cols].replace(-1, np.inf)
    region_ct[max_cols] += 1
    region_hh = households.to_frame(households.local_columns + ["large_area_id"])
    region_hh.index = region_hh.index.astype(int)
    region_p = persons.to_frame(persons.local_columns)
    region_p.index = region_p.index.astype(int)
    region_target = remi_pop_total.to_frame()

    def cut_to_la(xxx_todo_changeme):
        (large_area_id, hh) = xxx_todo_changeme
        p = region_p[region_p.household_id.isin(hh.index)]
        target = int(region_target.loc[large_area_id, str(iter_var)])
        ct = region_ct[region_ct.large_area_id == large_area_id]
        del ct["large_area_id"]
        return ct, hh, p, target, iter_var

    arg_per_la = list(map(cut_to_la, region_hh.groupby("large_area_id")))
    # cunks_per_la = map(presses_trans, arg_per_la)
    pool = Pool(8)
    cunks_per_la = pool.map(presses_trans, arg_per_la)
    pool.close()
    pool.join()
    out = reduce(operator.concat, cunks_per_la)

    # fix indexes
    out_hh_fixed = []
    out_p_fixed = []
    hhidmax = region_hh.index.values.max() + 1
    pidmax = region_p.index.values.max() + 1
    for hh, p in out:
        hh.index.name = "household_id"
        hh = hh.reset_index()
        hh["household_id_old"] = hh["household_id"]
        new_hh = (hh.building_id == -1).sum()
        hh.loc[hh.building_id == -1, "household_id"] = list(
            range(hhidmax, hhidmax + new_hh)
        )
        hhidmax += new_hh
        hhid_map = hh[["household_id_old", "household_id"]].set_index(
            "household_id_old"
        )
        p.index.name = "person_id"
        p = pd.merge(
            p.reset_index(), hhid_map, left_on="household_id", right_index=True
        )
        new_p = (p.household_id_x != p.household_id_y).sum()
        p.loc[p.household_id_x != p.household_id_y, "person_id"] = list(
            range(pidmax, pidmax + new_p)
        )
        pidmax += new_p
        p["household_id"] = p["household_id_y"]

        new_hh_df = hh.set_index("household_id")
        out_hh_fixed.append(new_hh_df[households.local_columns])
        out_p_fixed.append(p.set_index("person_id")[persons.local_columns])

    return pd.concat(out_hh_fixed, verify_integrity=True), pd.concat(out_p_fixed, verify_integrity=True) 
    # orca.add_table("households", pd.concat(out_hh_fixed, verify_integrity=True))
    # orca.add_table("persons", pd.concat(out_p_fixed, verify_integrity=True))
    

@orca.step()
def households_transition(
    households, persons, annual_household_control_totals, remi_pop_total, iter_var
):
    new_hh, new_p = hh_trans(households, persons, annual_household_control_totals, remi_pop_total, iter_var)
    # old_hh, old_p = hh_trans_old(households, persons, annual_household_control_totals, remi_pop_total, iter_var)
    print('break')

orca.run([
    "households_transition",  # households transition
], iter_vars=list(range(2020, 2021)))

print(orca.get_injectable('iter_var'))
hh_sim = orca.get_table('households').to_frame()

store = orca.get_injectable('store')
hh_base = store['households']
