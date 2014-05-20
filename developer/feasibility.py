import pandas as pd
import numpy as np
import time
import os
from urbansim.utils import misc, spotproforma


def get_possible_rents_by_use(dset):
    parcels = dset.parcels
    buildings = dset.buildings
    average_residential_rent = buildings[buildings.residential_units>0].groupby('zone_id').unit_price_res.mean()
    average_non_residential_rent = buildings[buildings.non_residential_sqft>0].groupby('zone_id').unit_price_nonres.mean()
    zoneavgrents = pd.DataFrame({'ave_residential_rent':average_residential_rent,'ave_office_rent':average_non_residential_rent,
                                 'ave_retail_rent':average_non_residential_rent,'ave_industrial_rent':average_non_residential_rent})
    INTERESTRATE = .05
    PERIODS = 20
    for col in zoneavgrents.columns:
        price = zoneavgrents[col]*-1
        zoneavgrents[col] = np.pmt(INTERESTRATE,PERIODS,price)

    avgrents = pd.DataFrame(index=parcels.index)
    for btype in ['residential', 'office', 'retail', 'industrial']:
        avgrents[btype] = zoneavgrents['ave_%s_rent' %
                                       btype].ix[parcels.zone_id].values
        if btype != 'residential':
            avgrents[btype] *= 1.2
    print avgrents.describe()

    return avgrents


# BIG QUESTION - should the above "possible rents" be greater than the
# here computed actual rents?  probably, right?
def current_rent_per_parcel(far_predictions, avgrents):
    # this is bad - need the right rents for each type
    # my thinking here is that I don't want to go around tearing down
    # buildings to convert to other uses - have to think about
    # this more
    return far_predictions.total_sqft * avgrents.residential * .8


RENTMULTIPLIER = 1.0  # this is essentially a calibration constant
DEV = None


def feasibility_run(dset, year=2010):

    global DEV
    if DEV is None:
        print "Running pro forma"
        DEV = spotproforma.Developer()

    # parcels = dset.fetch('parcels').join(
        # dset.fetch('zoning_for_parcels'), how="left")
    parcels = dset.parcels
    avgrents = get_possible_rents_by_use(dset)
    # compute total_sqft on the parcel, total current rent, and current far
    far_predictions = pd.DataFrame(index=parcels.index)
    far_predictions['total_sqft'] = dset.buildings.groupby(
        'parcel_id').building_sqft.sum().fillna(0)
    far_predictions['total_units'] = dset.buildings.groupby(
        'parcel_id').residential_units.sum().fillna(0)
    far_predictions['year_built'] = dset.buildings.groupby(
        'parcel_id').year_built.min().fillna(1960)
    far_predictions['currentrent'] = current_rent_per_parcel(
        far_predictions, avgrents)
    far_predictions['parcelsize'] = parcels.parcel_sqft
    # some parcels have unrealisticly small sizes
    far_predictions.parcelsize[far_predictions.parcelsize < 300] = 300

    print "Get zoning:", time.ctime()
    zoning = pd.read_csv('.//data//zoning.csv').set_index('parcel_id')

    # only keeps those parcels with zoning
    parcels = pd.merge(parcels, zoning, left_index=True, right_index=True)

    # need to map building types in zoning to allowable forms in the developer
    # model
    type_d = {
        'residential': [16,17,18,19],
        'industrial': [32,33,39],
        'retail': [22,23,24,28],
        'office': [27],
        'mixedresidential': [21],
        #'mixedoffice': [14],
    }

    # we have zoning by like 16 building types and rents/far predictions by
    # 4 building types so we have to convert one into the other - would
    # probably be better to have rents segmented by the same 16 building
    # types if we had good observations for that
    for form, btypes in type_d.iteritems():

        btypes = type_d[form]
        for btype in btypes:

            print form, btype
            # is type allowed
            tmp = parcels[parcels['type%d' % btype] == 't'][
                ['max_far', 'max_height']]
            # at what far
            far_predictions['type%d_zonedfar' % btype] = tmp['max_far']
            # at what height
            far_predictions['type%d_zonedheight' % btype] = tmp['max_height']

            # need to use max_dua here!!
            if btype == 1:
                far_predictions['type%d_zonedfar' % btype] = .75
            elif btype == 2:
                far_predictions['type%d_zonedfar' % btype] = 1.2

            # do the lookup in the developer model - this is where the
            # profitability is computed
            far_predictions['type%d_feasiblefar' % btype], \
                far_predictions['type%d_profit' % btype] = \
                DEV.lookup(form, avgrents[spotproforma.uses].as_matrix(),
                           far_predictions.currentrent * RENTMULTIPLIER,
                           far_predictions.parcelsize,
                           far_predictions['type%d_zonedfar' % btype],
                           far_predictions['type%d_zonedheight' % btype])

            # don't redevelop historic buildings
            far_predictions['type%d_feasiblefar' % btype][
                far_predictions.year_built < 1945] = 0.0
            far_predictions['type%d_profit' % btype][
                far_predictions.year_built < 1945] = 0.0

    far_predictions = far_predictions.join(avgrents)
    print "Feasibility and zoning\n", far_predictions.describe()
    far_predictions['currentrent'] /= spotproforma.CAPRATE
    fname = './/data//far_predictions.csv'
    far_predictions.to_csv(fname, index_col='parcel_id', float_format="%.2f")
    dset.save_tmptbl("feasibility", far_predictions)

    print "Finished developer", time.ctime()
