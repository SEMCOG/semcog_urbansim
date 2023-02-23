import pandas as pd
import numpy as np


def make_hh_la(households, buildings):
    h = pd.merge(households, buildings[['city_id', 'zone_id']],
                 left_on='building_id', right_index=True, how='left')
    #h = h.loc[h.large_area_id == large_area_id]

    # if zone_id == -1, replace them with 0
    h['zone_id'] = h['zone_id'].clip(0)

    h['city_zone'] = h.city_id * 10000 + h.zone_id
    h['hhsize'] = h.persons
    h.loc[h.persons > 7, 'hhsize'] = 7
    h['inc_qt'] = pd.qcut(h.income, 4, labels=[1, 2, 3, 4])
    return h


def get_czone_weights(hbase_year, hyear, hyear_g):
    for i in range(10): # why this?
        # for each la, inc_qt, and hhsize
        # sample all current hh by baseyear distribution among city and city_zone(as weight)
        addses = []
        for ind, x in hyear_g.iteritems():
            slic = hbase_year.loc[[ind]]
            print(ind, x, slic.weights.sum(), (1.0 * x / slic.weights.sum()))
            new_city_zone = slic.sample(x, replace=True, weights=slic.weights)
            # update weight after sample
            hbase_year.loc[[ind], 'weights'] *= (1.0 * x / slic.weights.sum())
            addses.append(new_city_zone[['city_id', 'city_zone']])
        # merge all hh together
        czone = pd.concat(addses, axis=0)

        # city_zone distribution insider city
        czoneg = czone.groupby('city_id').count()
        czoneg['hh'] = hyear.groupby('city_id').size()
        czoneg['dif'] = czoneg['city_zone'] - czoneg['hh']
        czoneg['ratio'] = czoneg['hh'] / czoneg['city_zone']

        weightsg = hyear.groupby('city_id').size().to_frame('target')
        weightsg['new_weights_sum'] = hbase_year.groupby(
            'city_id').weights.sum()
        weightsg['dif'] = weightsg['new_weights_sum'] - weightsg['target']
        weightsg['ratio'] = weightsg['target'] / weightsg['new_weights_sum']

        hbase_year = pd.merge(hbase_year[['city_id', 'city_zone', 'weights']], weightsg[[
                              'ratio']], left_on='city_id', right_index=True, how='left')
        hbase_year['weights'] *= hbase_year.ratio

        return [czone, czoneg, weightsg, hbase_year]


def target_year_data(households, buildings, parcels, year):
    hyear = make_hh_la(households, buildings)
    hyear_g = hyear.groupby(['large_area_id', 'inc_qt', 'hhsize']).size()
    print(len(hyear))
    print(hyear_g.head())

    b = pd.merge(buildings, parcels[['large_area_id']], left_on='parcel_id', right_index=True, how='left')
    b = b.reset_index()

    # if zone_id == -1, replace them with 0
    b['zone_id'] = b['zone_id'].clip(0)
    b['city_zone'] = b.city_id * 10000 + b.zone_id

    b1 = b[['building_id', 'residential_units', 'city_zone',
            'city_id', 'large_area_id']].loc[b.residential_units > 0]
    b1['residential_units'] = b1['residential_units'].astype(int)

    b2 = pd.DataFrame(np.repeat(
        b1.values, b1['residential_units'].values, axis=0), columns=b1.columns)

    return [hyear, hyear_g, b2]


def assign_hh_to_hu(czone, b2):
    def place_hh_to_hu(c, b, v):
        print('! remain city_zone', len(c),  '! remain units', len(b))
        if v <= 0:
            return
        c_ind = c.sample(v).index
        b_ind = b.sample(v).index
        # assign b2 building_id and city_zone to czone
        czone.loc[c_ind, 'building_id'] = b2.loc[b_ind, 'building_id'].values
        czone.loc[c_ind, 'city_zone'] = b2.loc[b_ind, 'city_zone'].values
        # reset b2 building_id to 0
        b2.loc[b_ind, 'building_id'] = 0

    for la in czone.large_area_id.unique():
        cities = czone.loc[czone.large_area_id == la, 'city_id'].unique()
        for city in cities:
            print('city', city,)
            czone_city = czone.loc[czone.large_area_id == la].loc[czone.city_id == city]
            b2_city = b2.loc[b2.large_area_id == la].loc[b2.city_id == city]
            cz_unique = set(list(czone_city.city_zone) +
                            list(b2_city.city_zone))

            for cz in cz_unique:
                czone_city_zone = czone_city[czone_city.city_zone == cz][czone_city.building_id == -1]
                b2_city_zone = b2_city[b2_city.city_zone == cz][b2_city.building_id != 0]

                # sample 97%
                minv = int(min(len(czone_city_zone), len(b2_city_zone)) * 0.97)
                place_hh_to_hu(czone_city_zone, b2_city_zone, minv)

            # remaining hh with -1 building_id
            c_remain = czone.loc[(czone.city_zone // 1000 == city)
                                 & (czone.building_id == -1)]
            b_remain = b2.loc[(b2.city_zone // 1000 == city) & (b2.building_id != 0)]
            minv = min(len(c_remain), len(b_remain))
            place_hh_to_hu(c_remain, b_remain, minv)

        # LA hh remains
        la_remain = czone.loc[(czone.large_area_id == la)
                              & (czone.building_id == -1)]
        la_b_ind_sample = b2.loc[(b2.large_area_id == la) & (b2.building_id != 0)].sample(len(la_remain))
        minv = min(len(la_remain), len(la_b_ind_sample))
        place_hh_to_hu(la_remain, la_b_ind_sample, minv)

    return [czone, b2]


def match_hh_targets(hyear_new, hyear_newg, b2):
    # This function didn't work, mover and reserve didn't match
    # one way to fix it to only move min(mover, reserve) units
	#adjust households to match existing/target
    for la, df in hyear_newg.groupby('large_area_id'):
        cross_mcd = [2065, 2095]
        # *drop 2065 rows
        df = df[df.new_city_id.isin(cross_mcd)]
        if la == 125:
            # add them if in 125
            df = df.append(hyear_newg.query('new_city_id == 2065'))
            df = df.append(hyear_newg[hyear_new.new_city_id.isin(cross_mcd)])
        df_pos = df.loc[df.dif > 0].set_index('new_city_id')
        movers = []
        for city, row in df_pos.iterrows():
            idx = hyear_new[(hyear_new.new_city_id == city)].sample(int(row.dif)).index.values
            movers.append(idx)
        movers = np.concatenate(movers)

        # city with negative difference
        df_neg = df.loc[df.dif < 0].set_index('new_city_id')
        resevers = []
        for city, row in df_neg.iterrows():
            if int(-row.dif) > b2[(b2.city_id == city) & (b2.building_id > 0)].shape[0]:
                print("not enough hu to reserve for city ", city)
            try:
                idx = b2[(b2.city_id == city) & (b2.building_id > 0)].sample(int(-row.dif)).index.values
                resevers.append(idx)
            except:
                print("city has error", city)
        resevers = np.concatenate(resevers)

        print(la, movers.shape, resevers.shape, movers.shape == resevers.shape)

        if movers.shape[0] > resevers.shape[0]:
            # if not enough reserve because of 2065, trim it
            movers = movers[:resevers.shape[0]]
        hyear_new.loc[movers, 'building_id'] = b2.loc[resevers].building_id.values
        hyear_new.loc[movers, 'city_zone'] = b2.loc[resevers].city_zone.values
        hyear_new['new_city_id'] = (hyear_new.city_zone // 10000)
        hyear_new['zone_id'] = (hyear_new['city_zone'] % 10000).astype(int)
        b2.loc[resevers, 'building_id'] = 0

    return [hyear_new, b2]


def get_hh_refine_difference(hyear_new, hyear):
    ind1 = hyear_new.groupby(['large_area_id', 'new_city_id']).size().index
    ind2 = hyear.groupby(['large_area_id', 'city_id']).size().index
    hyear_newg = pd.DataFrame([], index=ind1.union(ind2))
    hyear_newg.index.names = ['large_area_id', 'new_city_id']

    hyear_newg['hh'] = hyear.groupby(['large_area_id', 'city_id']).size()
    hyear_newg['newhh'] = hyear_new.groupby(
        ['large_area_id', 'new_city_id']).size()
    hyear_newg.fillna(0, inplace=True)
    hyear_newg['dif'] = hyear_newg['newhh'] - hyear_newg['hh']

    print(hyear_newg.sum())
    return hyear_newg
