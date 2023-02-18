import pandas as pd
import numpy as np


def make_hh_la(households, buildings):
    h = pd.merge(households, buildings[['city_id', 'zone_id']],
                 left_on='building_id', right_index=True, how='left')
    #h = h.loc[h.large_area_id == large_area_id]
    h['city_zone'] = h.city_id * 10000 + h.zone_id
    h['hhsize'] = h.persons
    h.loc[h.persons > 7, 'hhsize'] = 7
    h['inc_qt'] = pd.qcut(h.income, 4, labels=[1, 2, 3, 4])
    return h


def get_czone_weights(hbase_year, hyear, hyear_g):
    for i in range(10):
        addses = []
        for ind, x in hyear_g.iteritems():
            slic = hbase_year.loc[[ind]]
            print(ind, x, slic.weights.sum(), (1.0 * x / slic.weights.sum()))
            new_city_zone = slic.sample(x, replace=True, weights=slic.weights)
            hbase_year.loc[[ind], 'weights'] *= (1.0 * x / slic.weights.sum())
            addses.append(new_city_zone[['city_id', 'city_zone']])

        czone = pd.concat(addses, axis=0)

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
    hyear = make_hh_la(households, buildings, year)
    hyear_g = hyear.groupby(['large_area_id', 'inc_qt', 'hhsize']).size()
    print(len(hyear))
    print(hyear_g.head())

    b = pd.merge(buildings, parcels[['large_area_id']], left_on='parcel_id', right_index=True, how='left')
    b = b.reset_index()
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
        czone.loc[c_ind, 'building_id'] = b.loc[b_ind,
                                                    'building_id'].values
        czone.loc[c_ind, 'city_zone'] = b.loc[b_ind,
                                                    'city_zone'].values
        # reset b2 building_id to 0
        b2.loc[b_ind, 'building_id'] = 0

    for la in czone.large_area_id.unique():
        cities = czone.loc[czone.large_area_id == la, 'city_id'].unique()
        for city in cities:
            print('city', city,)
            czone_city = czone.loc[czone.city_id == city]
            b2_city = b2.loc[b2.city_id == city]
            cz_unique = set(list(czone_city.city_zone) +
                            list(b2_city.city_zone))

            for cz in cz_unique:
                czone_city_zone = czone_city.loc[czone_city.city_zone == cz]
                b2_city_zone = b2_city.loc[b2_city.city_zone == cz]

                # sample 97%
                minv = int(min(len(czone_city_zone), len(b2_city_zone)) * 0.97)
                place_hh_to_hu(czone_city_zone, b2_city_zone, minv)

            # remaining hh with -1 building_id
            c_remain = czone.loc[(czone.city_id == city)
                                 & (czone.building_id == -1)]
            b_remain = b2.loc[(b2.city_id == city) & (b2.building_id != 0)]
            minv = min(len(c_remain), len(b_remain))
            place_hh_to_hu(c_remain, b_remain, minv)

        # LA hh remains
        la_remain = czone.loc[(czone.large_area_id == la)
                              & (czone.building_id == -1)]
        la_b_ind_sample = b2.loc[(b2.large_area_id == la) & (b2.building_id != 0)].sample(len(la_remain)).index
        minv = min(len(la_remain), len(la_b_ind_sample))
        place_hh_to_hu(la_remain, la_b_ind_sample, minv)

    return [czone, b2]


def match_hh_targets(hyear_new, hyear_newg, b2):
	#adjust households to match existing/target
    for la, df in hyear_newg.groupby('large_area_id'):
        df_pos = df.loc[df.dif > 0].set_index('new_city_id')
        movers = []
        for city, row in df_pos.iterrows():
            idx = hyear_new[(hyear_new.new_city_id == city)
                            ].sample(int(row.dif)).index.values
            movers.append(idx)
        movers = np.concatenate(movers)

        df_neg = df.loc[df.dif < 0].set_index('new_city_id')
        resevers = []
        for city, row in df_neg.iterrows():
            idx = b2[(b2.city_id == city) & (b2.building_id > 0)
                     ].sample(int(-row.dif)).index.values
            resevers.append(idx)
        resevers = np.concatenate(resevers)

        print(la, movers.shape, resevers.shape, movers.shape == resevers.shape)

        hyear_new.loc[movers,
                      'building_id'] = b2.loc[resevers].building_id.values
        hyear_new.loc[movers, 'city_zone'] = b2.loc[resevers].city_zone.values
        hyear_new['new_city_id'] = (hyear_new.city_zone/10000.0).astype(int)
        hyear_new['zone_id'] = (hyear_new['city_zone'] % 10000).astype(int)
        b2.loc[resevers, 'building_id'] = 0

    return [hyear_new, b2]


def get_hh_refine_difference(hyear_new, hyear, year):
    ind1 = hyear_new.groupby(['large_area_id', 'new_city_id']).size().index
    ind2 = hyear.groupby(['large_area_id', 'city_id']).size().index
    hyear_newg = pd.DataFrame([], index=ind1.union(ind2))
    hyear_newg.index.names = ['large_area_id', 'new_city_id']

    hyear_newg[year +
               '_hh'] = hyear.groupby(['large_area_id', 'city_id']).size()
    hyear_newg['newhh'] = hyear_new.groupby(
        ['large_area_id', 'new_city_id']).size()
    hyear_newg.fillna(0, inplace=True)
    hyear_newg['dif'] = hyear_newg['newhh'] - hyear_newg[year + '_hh']

    print(hyear_newg.sum())
    return hyear_newg
