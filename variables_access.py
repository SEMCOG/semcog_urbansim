import pandas as pd, numpy as np
from urbansim.utils import misc
import dataset

###
import orca
import utils
import pandana as pdna


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
def nodeid_walk(schools, parcels):
    return orca.get_injectable('net_walk').get_node_ids(schools['point_x'], schools['point_y'])



#####################
# NODES_DRV VARIABLES
#####################

#net_drv = orca.get_injectable('net_drv')

#'GroceryStores', 'HealthCenters', 'Hospitals', 'Libraries',       'Park_Entrance_points', 'UrgentCare'

def get_nearest(net, dfpoi, cats, searchdis, numpoi, maxdis):
    net.set_pois(''.join(cats), dfpoi['point_x'], dfpoi['point_y'])
    return net.nearest_pois(searchdis, ''.join(cats), num_pois=numpoi, max_distance=maxdis)[1]

    
@orca.column('nodes_drv', cache=True, cache_scope='iteration')
def drv_nearest_hospital(nodes_drv, poi):
    cats = ['Hospitals']
    t = poi.to_frame()[poi.category.isin(cats)]
    return get_nearest(orca.get_injectable('net_drv'), t, cats, 15, 1, 16 )


@orca.column('nodes_drv', cache=True, cache_scope='iteration')
def drv_nearest_grocery(nodes_drv, poi):
    cats = ['GroceryStores']
    t = poi.to_frame()[poi.category.isin(cats)]    
    return get_nearest(orca.get_injectable('net_drv'), t, cats, 15, 1, 16 )

@orca.column('nodes_drv', cache=True, cache_scope='iteration')
def drv_nearest_healthcenter(nodes_drv, poi):
    cats = ['HealthCenters']
    t = poi.to_frame()[poi.category.isin(cats)]    
    return get_nearest(orca.get_injectable('net_drv'), t, cats, 15, 1, 16 )


@orca.column('nodes_drv', cache=True, cache_scope='iteration')
def drv_nearest_library(nodes_drv, poi):
    cats = ['Libraries']
    t = poi.to_frame()[poi.category.isin(cats)]    
    return get_nearest(orca.get_injectable('net_drv'), t, cats, 15, 1, 16 )


@orca.column('nodes_drv', cache=True, cache_scope='iteration')
def drv_nearest_park(nodes_drv, poi):
    cats = ['Park_Entrance_points']
    t = poi.to_frame()[poi.category.isin(cats)]    
    return get_nearest(orca.get_injectable('net_drv'), t, cats, 15, 1, 16 )


@orca.column('nodes_drv', cache=True, cache_scope='iteration')
def drv_nearest_urgentcare(nodes_drv, poi):
    cats = ['UrgentCare']
    t = poi.to_frame()[poi.category.isin(cats)] 
    return get_nearest(orca.get_injectable('net_drv'), t, cats, 15, 1, 16 )




       
#####################
# NODES_WALK VARIABLES
#####################


@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def walk_nearest_hospital(nodes_walk, poi):
    cats = ['Hospitals']
    t = poi.to_frame()[poi.category.isin(cats)]
    print 't',t
    return get_nearest(orca.get_injectable('net_walk'), t, cats, 7920 , 1, 7921 )

@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def walk_nearest_grocery(nodes_walk, poi):
    cats = ['GroceryStores']
    t = poi.to_frame()[poi.category.isin(cats)]    
    return get_nearest(orca.get_injectable('net_walk'), t, cats, 7920 , 1, 7921)

@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def walk_nearest_healthcenter(nodes_walk, poi):
    cats = ['HealthCenters']
    t = poi.to_frame()[poi.category.isin(cats)]     
    return get_nearest(orca.get_injectable('net_walk'), t, cats, 7920 , 1, 7921 )

@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def walk_nearest_library(nodes_walk, poi):
    cats = ['Libraries']
    t = poi.to_frame()[poi.category.isin(cats)]     
    return get_nearest(orca.get_injectable('net_walk'), t, cats, 7920 , 1, 7921 )

@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def walk_nearest_park(nodes_walk, poi):
    cats = ['Park_Entrance_points']
    t = poi.to_frame()[poi.category.isin(cats)]    
    return get_nearest(orca.get_injectable('net_walk'), t, cats, 7920 , 1, 7921 )

@orca.column('nodes_walk', cache=True, cache_scope='iteration')
def walk_nearest_urgentcare(nodes_walk, poi):
    cats = ['UrgentCare']
    t = poi.to_frame()[poi.category.isin(cats)]     
    return get_nearest(orca.get_injectable('net_walk'), t, cats, 7920 , 1, 7921 )











