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


def get_czone_weights(h2015_zone, hyear, hyear_g):
    for i in range(10):
        addses = []
        for ind, x in hyear_g.iteritems():
            slic = h2015_zone.loc[[ind]]
            print(ind, x, slic.weights.sum(), (1.0 * x / slic.weights.sum()))
            new_city_zone = slic.sample(x, replace=True, weights=slic.weights)
            h2015_zone.loc[[ind], 'weights'] *= (1.0 * x / slic.weights.sum())
            addses.append(new_city_zone[['b_city_id', 'city_zone']])

        czone = pd.concat(addses, axis=0)

        czoneg = czone.groupby('b_city_id').count()
        czoneg[year + '_hh'] = hyear.groupby('b_city_id').size()
        czoneg['dif'] = czoneg['city_zone'] - czoneg[year + '_hh']
        czoneg['ratio'] = czoneg[year + '_hh'] / czoneg['city_zone']

        weightsg = hyear.groupby('b_city_id').size().to_frame('target')
        weightsg['new_weights_sum'] = h2015_zone.groupby(
            'b_city_id').weights.sum()
        weightsg['dif'] = weightsg['new_weights_sum'] - weightsg['target']
        weightsg['ratio'] = weightsg['target'] / weightsg['new_weights_sum']

        h2015_zone = pd.merge(h2015_zone[['b_city_id', 'city_zone', 'weights']], weightsg[[
                              'ratio']], left_on='b_city_id', right_index=True, how='left')
        h2015_zone['weights'] *= h2015_zone.ratio

        return [czone, czoneg, weightsg, h2015_zone]


def target_year_data(households, buildings, parcels, year):
    hyear = make_hh_la(households, buildings, year)
    hyear_g = hyear.groupby(['large_area_id', 'inc_qt', 'hhsize']).size()
    print(len(hyear))
    print(hyear_g.head())

    b = pd.merge(buildings, parcels[['large_area_id']], left_on='parcel_id', right_index=True, how='left')
    b = b.reset_index()
    b['city_zone'] = b.b_city_id * 10000 + b.b_zone_id

    b1 = b[['building_id', 'residential_units', 'city_zone',
            'b_city_id', 'large_area_id']].loc[b.residential_units > 0]
    b1['residential_units'] = b1['residential_units'].astype(int)

    b2 = pd.DataFrame(np.repeat(
        b1.values, b1['residential_units'].values, axis=0), columns=b1.columns)

    return [hyear, hyear_g, b2]


def assign_hh_to_hu(czone, b2):
    for la in czone.large_area_id.unique():
        cities = czone.loc[czone.large_area_id == la, 'b_city_id'].unique()
        for city in cities:
            print('city', city,)
            czone_city = czone.loc[czone.b_city_id == city]
            b2_city = b2.loc[b2.b_city_id == city]
            cz_unique = set(list(czone_city.city_zone) +
                            list(b2_city.city_zone))

            for cz in cz_unique:
                czone_city_zone = czone_city.loc[czone_city.city_zone == cz]
                b2_city_zone = b2_city.loc[b2_city.city_zone == cz]

                minv = int(min(len(czone_city_zone), len(b2_city_zone)) * 0.97)
                if minv > 0:
                    cind = czone_city_zone.sample(minv).index
                    bind = b2_city_zone.sample(minv).index
                    czone.loc[cind, 'building_id'] = b2_city_zone.loc[bind,
                                                                      'building_id'].values
                    czone.loc[cind, 'city_zone'] = b2_city_zone.loc[bind,
                                                                    'city_zone'].values
                    b2.loc[bind, 'building_id'] = 0

            c_remain = czone.loc[(czone.b_city_id == city)
                                 & (czone.building_id == -1)]
            b_remain = b2.loc[(b2.b_city_id == city) & (b2.building_id != 0)]
            minv = min(len(c_remain), len(b_remain))
            print('! remain city_zone', len(c_remain),  '! remain units', len(b_remain))
            if minv > 0:
                crmn_ind = c_remain.sample(minv).index
                brmn_ind = b_remain.sample(minv).index
                czone.loc[crmn_ind, 'building_id'] = b_remain.loc[brmn_ind,
                                                                  'building_id'].values
                czone.loc[crmn_ind, 'city_zone'] = b_remain.loc[brmn_ind,
                                                                'city_zone'].values
                b2.loc[brmn_ind, 'building_id'] = 0

        la_remain = czone.loc[(czone.large_area_id == la)
                              & (czone.building_id == -1)]
        if len(la_remain) > 0:
            ind = b2.loc[(b2.large_area_id == la) & (b2.building_id != 0)].sample(len(la_remain)).index
            czone.loc[la_remain.index,
                      'building_id'] = b2.loc[ind].building_id.values
            czone.loc[la_remain.index,
                      'city_zone'] = b2.loc[ind].city_zone.values
            b2.loc[ind, 'building_id'] = 0

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
            idx = b2[(b2.b_city_id == city) & (b2.building_id > 0)
                     ].sample(int(-row.dif)).index.values
            resevers.append(idx)
        resevers = np.concatenate(resevers)

        print(la, movers.shape, resevers.shape, movers.shape == resevers.shape)

        hyear_new.loc[movers,
                      'building_id'] = b2.loc[resevers].building_id.values
        hyear_new.loc[movers, 'city_zone'] = b2.loc[resevers].city_zone.values
        hyear_new['new_city_id'] = (hyear_new.city_zone/10000.0).astype(int)
        hyear_new['b_zone_id'] = (hyear_new['city_zone'] % 10000).astype(int)
        b2.loc[resevers, 'building_id'] = 0

    return [hyear_new, b2]


def get_hh_refine_difference(hyear_new, hyear, year):
    ind1 = hyear_new.groupby(['large_area_id', 'new_city_id']).size().index
    ind2 = hyear.groupby(['large_area_id', 'b_city_id']).size().index
    hyear_newg = pd.DataFrame([], index=ind1.union(ind2))
    hyear_newg.index.names = ['large_area_id', 'new_city_id']

    hyear_newg[year +
               '_hh'] = hyear.groupby(['large_area_id', 'b_city_id']).size()
    hyear_newg['newhh'] = hyear_new.groupby(
        ['large_area_id', 'new_city_id']).size()
    hyear_newg.fillna(0, inplace=True)
    hyear_newg['dif'] = hyear_newg['newhh'] - hyear_newg[year + '_hh']

    print(hyear_newg.sum())
    return hyear_newg
