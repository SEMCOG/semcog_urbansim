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
    index = orca.get_table('parcels').index
    form_to_btype = orca.get_injectable("form_to_btype")
    buildings = orca.get_table("buildings").to_frame(
        ["parcel_id", "building_type_id", "residential_units", "building_age"])
    zoning = orca.get_table('zoning')

    lone_house = buildings[
                     (buildings.building_type_id == 81) &
                     (buildings.residential_units == 1)].groupby("parcel_id").building_type_id.count() == 1
    lone_house = lone_house.reindex(index, fill_value=False)

    new_building = buildings.groupby("parcel_id").building_age.min() <= 5
    new_building = new_building.reindex(index, fill_value=False)

    development = index.isin(orca.get_table('events_addition').parcel_id)

    demolition = index.isin(
        buildings[
            buildings.index.isin(orca.get_table('events_deletion').building_id)
        ].parcel_id
    )

    parcel_refin = set()
    refinements = orca.get_table('refiner_events').to_frame()

    for e in refinements.location_expression:
        parcel_refin |= set(buildings.query(e).parcel_id)

    refiner = index.isin(parcel_refin)

    protected = lone_house | new_building | development | demolition | refiner

    columns = ['type%d' % typ for typ in form_to_btype[form]]
    allowed = zoning.to_frame(columns).max(axis=1).reindex(index, fill_value=False)
    return (allowed > 0) & (~protected)


@orca.column('parcels', cache=True, cache_scope='iteration')
def max_far(parcels, zoning):
    return zoning.max_far.reindex(parcels.index)


@orca.column('parcels', cache=True, cache_scope='iteration')
def max_dua(parcels, zoning):
    return zoning.max_dua.reindex(parcels.index)


@orca.column('parcels', cache=True, cache_scope='iteration')
def max_height(parcels, zoning):
    return zoning.max_height.reindex(parcels.index)


@orca.column('parcels', cache=True, cache_scope='iteration')
def pct_undev(parcels, zoning):
    return zoning.pct_undev.reindex(parcels.index)


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
    schools = schools.to_frame(['dcode', 'totalachievementindex'])
    achievement = schools.groupby('dcode').totalachievementindex.mean()
    return misc.reindex(achievement, parcels.school_id).fillna(0)


@orca.column('parcels', cache=True, cache_scope='iteration')
def drv_nearest_hospital(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_drv.drv_nearest_hospital, parcels.nodeid_drv)


@orca.column('parcels', cache=True, cache_scope='iteration')
def drv_nearest_healthcenter(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_drv.drv_nearest_healthcenter, parcels.nodeid_drv)


@orca.column('parcels', cache=True, cache_scope='iteration')
def drv_nearest_grocery(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_drv.drv_nearest_grocery, parcels.nodeid_drv)


@orca.column('parcels', cache=True, cache_scope='iteration')
def drv_nearest_urgentcare(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_drv.drv_nearest_urgentcare, parcels.nodeid_drv)


@orca.column('parcels', cache=True, cache_scope='iteration')
def drv_nearest_library(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_drv.drv_nearest_library, parcels.nodeid_drv)


@orca.column('parcels', cache=True, cache_scope='iteration')
def drv_nearest_park(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_drv.drv_nearest_park, parcels.nodeid_drv)


@orca.column('parcels', cache=True, cache_scope='iteration')
def walk_nearest_hospital(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_walk.walk_nearest_hospital, parcels.nodeid_walk)


@orca.column('parcels', cache=True, cache_scope='iteration')
def walk_nearest_grocery(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_walk.walk_nearest_grocery, parcels.nodeid_walk)


@orca.column('parcels', cache=True, cache_scope='iteration')
def walk_nearest_healthcenter(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_walk.walk_nearest_healthcenter, parcels.nodeid_walk)


@orca.column('parcels', cache=True, cache_scope='iteration')
def walk_nearest_urgentcare(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_walk.walk_nearest_urgentcare, parcels.nodeid_walk)


@orca.column('parcels', cache=True, cache_scope='iteration')
def walk_nearest_library(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_walk.walk_nearest_library, parcels.nodeid_walk)


@orca.column('parcels', cache=True, cache_scope='iteration')
def walk_nearest_park(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_walk.walk_nearest_park, parcels.nodeid_walk)


@orca.column('parcels', cache=True, cache_scope='forever')
def crime_ucr_rate(crime_rates):
    return crime_rates['ucr_crime_rate']


@orca.column('parcels', cache=True, cache_scope='forever')
def crime_other_rate(crime_rates):
    return crime_rates['other_crime_rate']
