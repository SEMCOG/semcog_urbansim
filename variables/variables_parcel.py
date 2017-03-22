import orca
import pandas as pd
from urbansim.utils import misc


#####################
# PARCELS VARIABLES
#####################

@orca.column('parcels', cache=True, cache_scope='iteration')
def acres(parcels):
    return parcels.parcel_sqft / 43560


@orca.column('parcels', cache=True, cache_scope='iteration')
def x(parcels):
    return parcels.centroid_x


@orca.column('parcels', cache=True, cache_scope='iteration')
def y(parcels):
    return parcels.centroid_y


@orca.column('parcels', cache=True, cache_scope='iteration')
def allowed(parcels):
    df = pd.DataFrame(index=parcels.index)
    df['allowed'] = True
    return df.allowed


def parcel_average_price(use, parcels):
    if len(orca.get_table('nodes_walk')) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_walk')[use],
                        orca.get_table('parcels').nodeid_walk)


def parcel_is_allowed(form):
    form_to_btype = orca.get_injectable("form_to_btype")
    buildings = orca.get_table("buildings").to_frame(["parcel_id", "building_type_id", "residential_units"])
    lone_house = buildings[
                     (buildings.building_type_id == 81) &
                     (buildings.residential_units == 1)].groupby(by="parcel_id").building_type_id.count() == 1
    orca.add_injectable("lone_house", lone_house)
    zoning = orca.get_table('zoning')
    allowed = [(zoning['type%d' % typ] > 0)
               for typ in form_to_btype[form]]

    s = pd.concat(allowed, axis=1).max(axis=1).reindex(orca.get_table('parcels').index).fillna(False)

    return s.astype("bool") & (~lone_house).reindex(zoning.index, fill_value=True)
    # return s.astype("bool")


@orca.column('parcels', cache=True, cache_scope='iteration')
def max_far(parcels):
    df = pd.DataFrame(index=parcels.index)
    df['max_far'] = orca.get_table('zoning').max_far
    return df.max_far


@orca.column('parcels', cache=True, cache_scope='iteration')
def max_dua(parcels):
    df = pd.DataFrame(index=parcels.index)
    df['max_dua'] = orca.get_table('zoning').max_dua
    return df.max_dua


@orca.column('parcels', cache=True, cache_scope='iteration')
def max_height(parcels):
    df = pd.DataFrame(index=parcels.index)
    df['max_height'] = orca.get_table('zoning').max_height
    return df.max_height


@orca.column('parcels', cache=True, cache_scope='iteration')
def pct_undev(parcels):
    df = pd.DataFrame(index=parcels.index)
    df['pct_undev'] = orca.get_table('zoning').pct_undev
    return df.pct_undev


@orca.column('parcels', cache=True, cache_scope='iteration')
def parcel_size(parcels):
    return parcels.parcel_sqft


@orca.column('parcels')
def ave_unit_size(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_walk.ave_unit_sqft, parcels.nodeid_walk)


@orca.column('parcels', cache=True, cache_scope='iteration')
def total_units(parcels, buildings):
    return buildings.residential_units.groupby(buildings.parcel_id).sum(). \
        reindex(parcels.index).fillna(0)


@orca.column('parcels', cache=True, cache_scope='iteration')
def total_job_spaces(parcels, buildings):
    return buildings.job_spaces.groupby(buildings.parcel_id).sum(). \
        reindex(parcels.index).fillna(0)


@orca.column('parcels', cache=True, cache_scope='iteration')
def total_sqft(parcels, buildings):
    return buildings.building_sqft.groupby(buildings.parcel_id).sum(). \
        reindex(parcels.index).fillna(0)


@orca.column('parcels')
def land_cost(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        return pd.Series(index=parcels.index)
    return (parcels.total_sqft * parcel_average_price("residential", parcels)). \
        reindex(parcels.index).fillna(0)


@orca.column('parcels', cache=True, cache_scope='iteration')
def parcel_far(parcels):
    return (parcels.total_sqft / parcels.parcel_sqft).fillna(0)


@orca.column('parcels', cache=True, cache_scope='iteration')
def school_district_achievement(parcels, schools):
    return misc.reindex(parcels['school_id'], schools.to_frame(['dcode', 'totalachievementindex']).
                        groupby('dcode').totalachievementindex.mean())


@orca.column('parcels', cache=True, cache_scope='iteration')
def drv_nearest_hospital(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_drv').drv_nearest_hospital,
                        parcels.nodeid_drv)


@orca.column('parcels', cache=True, cache_scope='iteration')
def drv_nearest_healthcenter(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_drv').drv_nearest_healthcenter,
                        parcels.nodeid_drv)


@orca.column('parcels', cache=True, cache_scope='iteration')
def drv_nearest_grocery(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_drv').drv_nearest_grocery,
                        parcels.nodeid_drv)


@orca.column('parcels', cache=True, cache_scope='iteration')
def drv_nearest_urgentcare(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_drv').drv_nearest_urgentcare,
                        parcels.nodeid_drv)


@orca.column('parcels', cache=True, cache_scope='iteration')
def drv_nearest_library(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_drv').drv_nearest_library,
                        parcels.nodeid_drv)


@orca.column('parcels', cache=True, cache_scope='iteration')
def drv_nearest_park(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_drv').drv_nearest_park,
                        parcels.nodeid_drv)


@orca.column('parcels', cache=True, cache_scope='iteration')
def walk_nearest_hospital(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_walk').walk_nearest_hospital,
                        parcels.nodeid_walk)


@orca.column('parcels', cache=True, cache_scope='iteration')
def walk_nearest_grocery(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_walk').walk_nearest_grocery,
                        parcels.nodeid_walk)


@orca.column('parcels', cache=True, cache_scope='iteration')
def walk_nearest_healthcenter(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_walk').walk_nearest_healthcenter,
                        parcels.nodeid_walk)


@orca.column('parcels', cache=True, cache_scope='iteration')
def walk_nearest_urgentcare(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_walk').walk_nearest_urgentcare,
                        parcels.nodeid_walk)


@orca.column('parcels', cache=True, cache_scope='iteration')
def walk_nearest_library(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_walk').walk_nearest_library,
                        parcels.nodeid_walk)


@orca.column('parcels', cache=True, cache_scope='iteration')
def walk_nearest_park(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(orca.get_table('nodes_walk').walk_nearest_park,
                        parcels.nodeid_walk)


@orca.column('parcels', cache=True, cache_scope='forever')
def crime_ucr_rate(crime_rates):
    return crime_rates['ucr_crime_rate']


@orca.column('parcels', cache=True, cache_scope='forever')
def crime_other_rate(crime_rates):
    return crime_rates['other_crime_rate']
