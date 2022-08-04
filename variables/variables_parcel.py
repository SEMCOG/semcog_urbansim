import orca
import pandas as pd
from urbansim.utils import misc


#####################
# PARCELS VARIABLES
#####################


@orca.column("parcels", cache=True, cache_scope="iteration")
def acres(parcels):
    return parcels.parcel_sqft / 43560


@orca.column("parcels", cache=True, cache_scope="iteration")
def x(parcels):
    return parcels.centroid_x


@orca.column("parcels", cache=True, cache_scope="iteration")
def y(parcels):
    return parcels.centroid_y


@orca.column("parcels", cache=True, cache_scope="iteration")
def allowed(parcels):
    df = pd.DataFrame(index=parcels.index)
    df["allowed"] = True
    return df.allowed


def parcel_average_price(use, parcels):
    if len(orca.get_table("nodes_walk")) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(
        orca.get_table("nodes_walk")[use], orca.get_table("parcels").nodeid_walk
    )


def parcel_is_allowed(form=None):
    index = orca.get_table("parcels").index
    form_to_btype = orca.get_injectable("form_to_btype")
    buildings = orca.get_table("buildings").to_frame(
        [
            # #35
            # "b_city_id",
            "city_id",
            "parcel_id",
            "building_type_id",
            "residential_units",
            "building_age",
            "event_id",
            "sp_filter",
        ]
    )
    zoning = orca.get_table("zoning")
    year = orca.get_injectable("year")

    # lone_house = buildings[
    #                  (buildings.building_type_id == 81) &
    #                  (buildings.residential_units == 1)].groupby("parcel_id").building_type_id.count() == 1
    # lone_house = lone_house.reindex(index, fill_value=False)

    new_building = buildings.groupby("parcel_id").building_age.min() <= 5
    new_building = new_building.reindex(index, fill_value=False)

    # #35
    # count_hu = buildings.groupby("b_city_id").residential_units.sum()
    count_hu = buildings.groupby("city_id").residential_units.sum()
    hu_target = orca.get_table("extreme_hu_controls").to_frame()
    hu_target["c0"] = (hu_target.end - hu_target.base) * (year - 2015) + hu_target.base
    hu_target["c5"] = (hu_target.end - hu_target.base) * (
        year - 2015 + 5
    ) + hu_target.base
    p_city = orca.get_table("parcels").city_id
    if form == "residential":
        city_is_full = count_hu > hu_target[["c0", "c5"]].max(1).reindex(
            count_hu.index, fill_value=0
        )
        city_is_full = city_is_full[city_is_full]
        city_is_full = p_city.isin(city_is_full.index).reindex(index, fill_value=False)
    elif form is None:
        city_is_full = count_hu < hu_target[["c0", "c5"]].min(1).reindex(
            count_hu.index, fill_value=0
        )
        city_is_full = city_is_full[city_is_full]
        city_is_full = p_city.isin(city_is_full.index).reindex(index, fill_value=False)
    else:
        city_is_full = False

    development = index.isin(orca.get_table("events_addition").parcel_id)

    demolition = index.isin(
        buildings[
            buildings.index.isin(orca.get_table("events_deletion").building_id)
        ].parcel_id
    )

    wold_have_bean_in_events = (
        orca.get_table("parcels").non_residential_sqft >= 50000
    ) & (year <= 2020)
    wold_have_bean_in_events = wold_have_bean_in_events.reindex(index, fill_value=False)

    gq = index.isin(
        buildings[
            buildings.index.isin(orca.get_table("group_quarters").building_id)
        ].parcel_id
    )

    parcel_refin = set()
    refinements = orca.get_table("refiner_events").to_frame()

    for e in refinements.location_expression:
        s = set(buildings.query(e).parcel_id)
        if len(s) < 20:
            parcel_refin |= s

    refiner = index.isin(parcel_refin)

    protected = (
        new_building
        | development
        | demolition
        | wold_have_bean_in_events
        | refiner
        | gq
        | city_is_full
    )

    if form:
        columns = ["type%d" % typ for typ in form_to_btype[form]]
    else:
        columns = [
            "type%d" % typ
            for typ in set(
                item for sublist in list(form_to_btype.values()) for item in sublist
            )
        ]

    allowed = zoning.to_frame(columns).max(axis=1).reindex(index, fill_value=0)

    # if form == 'residential':
    #     out = ((allowed > 0) & (~protected)).to_frame("out")
    #     out["allowed"] = allowed
    #     out["lone_house"] = lone_house
    #     out["new_building"] = new_building
    #     out["development"] = development
    #     out["demolition"] = demolition
    #     out["refiner"] = refiner
    #     out.to_csv("runs/parcel_is_allowed.csv")

    return (allowed > 0) & (~protected)


def parcel_is_allowed_2050(form=None):
    # indentify parcels allowed for construction
    # TODO, will replace parcel_is_allowed
    pcl_index = orca.get_table("parcels").index
    form_to_btype = orca.get_injectable("form_to_btype")
    buildings = orca.get_table("buildings").to_frame(
        [
            "city_id",
            "city_id",
            "parcel_id",
            "building_type_id",
            "residential_units",
            "building_age",
            "event_id",
            "sp_filter",
        ]
    )
    zoning = orca.get_table("zoning")
    year = orca.get_injectable("year")

    pcl_new_building = buildings.groupby("parcel_id").building_age.min() <= 5
    pcl_new_building = pcl_new_building.reindex(pcl_index, fill_value=False)

    pcl_addition = pcl_index.isin(orca.get_table("events_addition").parcel_id)

    pcl_demolition = pcl_index.isin(
        buildings[
            buildings.index.isin(orca.get_table("events_deletion").building_id)
        ].parcel_id
    )

    pcl_big_nonres = (orca.get_table("parcels").non_residential_sqft >= 50000) & (
        year <= 2020
    )
    pcl_big_nonres = pcl_big_nonres.reindex(pcl_index, fill_value=False)

    pcl_gq = pcl_index.isin(
        buildings[
            buildings.index.isin(orca.get_table("group_quarters").building_id)
        ].parcel_id
    )

    parcel_refin = set()
    refinements = orca.get_table("refiner_events").to_frame()
    for e in refinements.location_expression:
        s = set(buildings.query(e).parcel_id)
        if len(s) < 20:
            parcel_refin |= s
    pcl_refiner = pcl_index.isin(parcel_refin)

    protected = (
        pcl_new_building
        | pcl_addition
        | pcl_demolition
        | pcl_big_nonres
        | pcl_refiner
        | pcl_gq
    )

    if form:
        columns = ["type%d" % typ for typ in form_to_btype[form]]
    else:
        columns = [
            "type%d" % typ
            for typ in set(
                item for sublist in list(form_to_btype.values()) for item in sublist
            )
        ]

    allowed = zoning.to_frame(columns).max(axis=1).reindex(pcl_index, fill_value=0)

    return (allowed > 0) & (~protected)


@orca.column("parcels", cache=True, cache_scope="iteration")
def max_far(parcels, zoning):
    return zoning.max_far.reindex(parcels.index)


@orca.column("parcels", cache=True, cache_scope="iteration")
def max_dua(parcels, zoning):
    return zoning.max_dua.reindex(parcels.index)


@orca.column("parcels", cache=True, cache_scope="iteration")
def max_height(parcels, zoning):
    return zoning.max_height.reindex(parcels.index)


@orca.column("parcels", cache=True, cache_scope="iteration")
def parcel_size(parcels):
    # apply pct_undev to parcel_size, which will be used in feasibility step
    return parcels.parcel_sqft - ((parcels.pct_undev.clip(0, 100) / 100) * parcels.parcel_sqft)


@orca.column("parcels")
def ave_unit_size(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_walk.ave_unit_sqft, parcels.nodeid_walk)


@orca.column("parcels", cache=True, cache_scope="iteration")
def total_units(parcels, buildings):
    return (
        buildings.residential_units.groupby(buildings.parcel_id)
        .sum()
        .reindex(parcels.index)
        .fillna(0)
    )


@orca.column("parcels", cache=True, cache_scope="iteration")
def total_job_spaces(parcels, buildings):
    return (
        buildings.job_spaces.groupby(buildings.parcel_id)
        .sum()
        .reindex(parcels.index)
        .fillna(0)
    )


@orca.column("parcels", cache=True, cache_scope="iteration")
def residential_sqft(parcels, buildings):
    return (
        buildings.residential_sqft.groupby(buildings.parcel_id)
        .sum()
        .reindex(parcels.index)
        .fillna(0)
    )


@orca.column("parcels", cache=True, cache_scope="iteration")
def non_residential_sqft(parcels, buildings):
    return (
        buildings.non_residential_sqft.groupby(buildings.parcel_id)
        .sum()
        .reindex(parcels.index)
        .fillna(0)
    )


@orca.column("parcels", cache=True, cache_scope="iteration")
def total_sqft(parcels, buildings):
    return (
        buildings.building_sqft.groupby(buildings.parcel_id)
        .sum()
        .reindex(parcels.index)
        .fillna(0)
    )


@orca.column("parcels")
def land_cost(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        return pd.Series(index=parcels.index)
    res = parcels.residential_sqft * parcel_average_price("residential", parcels)
    non_res = parcels.non_residential_sqft * parcel_average_price(
        "ave_nonres_sqft_price", parcels
    )
    return (res + non_res).reindex(parcels.index).fillna(0)


@orca.column("parcels", cache=True, cache_scope="iteration")
def parcel_far(parcels):
    return (parcels.total_sqft / parcels.parcel_sqft).fillna(0)


@orca.column("parcels", cache=True, cache_scope="iteration")
def school_district_achievement(parcels, schools):
    schools = schools.to_frame(["dcode", "totalachievementindex"])
    achievement = schools.groupby("dcode").totalachievementindex.mean()
    return misc.reindex(achievement, parcels.school_id).fillna(0)


@orca.column("parcels", cache=True, cache_scope="iteration")
def drv_nearest_hospital(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_drv.drv_nearest_hospital, parcels.nodeid_drv)


@orca.column("parcels", cache=True, cache_scope="iteration")
def drv_nearest_healthcenter(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_drv.drv_nearest_healthcenter, parcels.nodeid_drv)


@orca.column("parcels", cache=True, cache_scope="iteration")
def drv_nearest_grocery(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_drv.drv_nearest_grocery, parcels.nodeid_drv)


@orca.column("parcels", cache=True, cache_scope="iteration")
def drv_nearest_urgentcare(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_drv.drv_nearest_urgentcare, parcels.nodeid_drv)


@orca.column("parcels", cache=True, cache_scope="iteration")
def drv_nearest_library(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_drv.drv_nearest_library, parcels.nodeid_drv)


@orca.column("parcels", cache=True, cache_scope="iteration")
def drv_nearest_park(parcels, nodes_drv):
    if len(nodes_drv) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_drv.drv_nearest_park, parcels.nodeid_drv)


@orca.column("parcels", cache=True, cache_scope="iteration")
def walk_nearest_hospital(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_walk.walk_nearest_hospital, parcels.nodeid_walk)


@orca.column("parcels", cache=True, cache_scope="iteration")
def walk_nearest_grocery(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_walk.walk_nearest_grocery, parcels.nodeid_walk)


@orca.column("parcels", cache=True, cache_scope="iteration")
def walk_nearest_healthcenter(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_walk.walk_nearest_healthcenter, parcels.nodeid_walk)


@orca.column("parcels", cache=True, cache_scope="iteration")
def walk_nearest_urgentcare(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_walk.walk_nearest_urgentcare, parcels.nodeid_walk)


@orca.column("parcels", cache=True, cache_scope="iteration")
def walk_nearest_library(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_walk.walk_nearest_library, parcels.nodeid_walk)


@orca.column("parcels", cache=True, cache_scope="iteration")
def walk_nearest_park(parcels, nodes_walk):
    if len(nodes_walk) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    return misc.reindex(nodes_walk.walk_nearest_park, parcels.nodeid_walk)


@orca.column("parcels", cache=True, cache_scope="forever")
def crime_ucr_rate(crime_rates):
    return crime_rates["ucr_crime_rate"]


@orca.column("parcels", cache=True, cache_scope="forever")
def crime_other_rate(crime_rates):
    return crime_rates["other_crime_rate"]
