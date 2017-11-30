import numpy as np
import orca


#####################
# TRANSIT VARIABLES
#####################

@orca.column('transit_stops', cache=True, cache_scope='iteration')
def nodeid_walk(transit_stops):
    return orca.get_injectable('net_walk').get_node_ids(transit_stops['point_x'], transit_stops['point_y'])


#####################
# SCHOOL VARIABLES
#####################

@orca.column('schools', cache=True, cache_scope='iteration')
def nodeid_drv(schools):
    return orca.get_injectable('net_drv').get_node_ids(schools['point_x'], schools['point_y'])


@orca.column('schools', cache=True, cache_scope='iteration')
def nodeid_walk(schools):
    return orca.get_injectable('net_walk').get_node_ids(schools['point_x'], schools['point_y'])


#####################
# NODES_DRV VARIABLES
#####################

# net_drv = orca.get_injectable('net_drv')

# 'GroceryStores', 'HealthCenters', 'Hospitals', 'Libraries',       'Park_Entrance_points', 'UrgentCare'

def get_nearest(net, dfpoi, cats, searchdis, numpoi, maxdis):
    net.set_pois(''.join(cats), dfpoi['point_x'], dfpoi['point_y'])
    return net.nearest_pois(searchdis, ''.join(cats), num_pois=numpoi, max_distance=maxdis)[1]


@orca.column('nodes_drv', cache=True, cache_scope='iteration')
def drv_nearest_hospital(poi):
    cats = ['Hospitals']
    t = poi.to_frame()[poi.category.isin(cats)]
    return get_nearest(orca.get_injectable('net_drv'), t, cats, 15, 1, 16)


@orca.column('nodes_drv', cache=True, cache_scope='iteration')
def drv_nearest_grocery(poi):
    cats = ['GroceryStores']
    t = poi.to_frame()[poi.category.isin(cats)]
    return get_nearest(orca.get_injectable('net_drv'), t, cats, 15, 1, 16)


@orca.column('nodes_drv', cache=True, cache_scope='iteration')
def drv_nearest_healthcenter(poi):
    cats = ['HealthCenters']
    t = poi.to_frame()[poi.category.isin(cats)]
    return get_nearest(orca.get_injectable('net_drv'), t, cats, 15, 1, 16)


@orca.column('nodes_drv', cache=True, cache_scope='iteration')
def drv_nearest_library(poi):
    cats = ['Libraries']
    t = poi.to_frame()[poi.category.isin(cats)]
    return get_nearest(orca.get_injectable('net_drv'), t, cats, 15, 1, 16)


@orca.column('nodes_drv', cache=True, cache_scope='iteration')
def drv_nearest_park(poi):
    cats = ['Park_Entrance_points']
    t = poi.to_frame()[poi.category.isin(cats)]
    return get_nearest(orca.get_injectable('net_drv'), t, cats, 15, 1, 16)


@orca.column('nodes_drv', cache=True, cache_scope='iteration')
def drv_nearest_urgentcare(poi):
    cats = ['UrgentCare']
    t = poi.to_frame()[poi.category.isin(cats)]
    return get_nearest(orca.get_injectable('net_drv'), t, cats, 15, 1, 16)


#####################
# NODES_WALK VARIABLES
#####################


@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def walk_nearest_hospital(poi):
    cats = ['Hospitals']
    t = poi.to_frame()[poi.category.isin(cats)]
    return get_nearest(orca.get_injectable('net_walk'), t, cats, 7920, 1, 7921)


@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def walk_nearest_grocery(poi):
    cats = ['GroceryStores']
    t = poi.to_frame()[poi.category.isin(cats)]
    return get_nearest(orca.get_injectable('net_walk'), t, cats, 7920, 1, 7921)


@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def walk_nearest_healthcenter(poi):
    cats = ['HealthCenters']
    t = poi.to_frame()[poi.category.isin(cats)]
    return get_nearest(orca.get_injectable('net_walk'), t, cats, 7920, 1, 7921)


@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def walk_nearest_library(poi):
    cats = ['Libraries']
    t = poi.to_frame()[poi.category.isin(cats)]
    return get_nearest(orca.get_injectable('net_walk'), t, cats, 7920, 1, 7921)


@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def walk_nearest_park(poi):
    cats = ['Park_Entrance_points']
    t = poi.to_frame()[poi.category.isin(cats)]
    return get_nearest(orca.get_injectable('net_walk'), t, cats, 7920, 1, 7921)


@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def walk_nearest_urgentcare(poi):
    cats = ['UrgentCare']
    t = poi.to_frame()[poi.category.isin(cats)]
    return get_nearest(orca.get_injectable('net_walk'), t, cats, 7920, 1, 7921)


@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def node_r1500_acre(nodes_walk):
    return nodes_walk['node_r1500_sqft'] / 43560


@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def ln_empden(nodes_walk):
    return np.log1p(nodes_walk.jobs / nodes_walk.node_r1500_acre).fillna(0)


@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def ln_popden(nodes_walk):
    return np.log1p(nodes_walk.population / nodes_walk.node_r1500_acre).fillna(0)


@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def percent_high_income(nodes_walk):
    return np.log1p(nodes_walk.highinc_hhs / nodes_walk.households).fillna(0)


@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def percent_mid_income(nodes_walk):
    return np.log1p(nodes_walk.midinc_hhs / nodes_walk.households).fillna(0)


@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def percent_low_income(nodes_walk):
    return np.log1p(nodes_walk.lowinc_hhs / nodes_walk.households).fillna(0)


@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def percent_race1(nodes_walk):
    return np.log1p(nodes_walk.race_1_hhs / nodes_walk.households).fillna(0)


@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def percent_race2(nodes_walk):
    return np.log1p(nodes_walk.race_2_hhs / nodes_walk.households).fillna(0)


@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def percent_race3(nodes_walk):
    return np.log1p(nodes_walk.race_3_hhs / nodes_walk.households).fillna(0)


@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def percent_race4(nodes_walk):
    return np.log1p(nodes_walk.race_4_hhs / nodes_walk.households).fillna(0)


@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def percent_hh_with_children(nodes_walk):
    return np.log1p(nodes_walk.hhs_with_children / nodes_walk.households).fillna(0)


@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def housing_cost(nodes_walk):
    return (nodes_walk.residential * nodes_walk.ave_unit_sqft) / 5.0

