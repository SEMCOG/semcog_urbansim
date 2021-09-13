#!/usr/bin/env python
# coding: utf-8

# In[13]:


from urbansim.utils import misc, networks
import os, dataset

networks.NETWORKS = networks.Networks(
    [os.path.join(misc.data_dir(), x) for x in ['osm_semcog.pkl']],
    factors=[1.0],
    maxdistances=[2000],
    twoway=[1],
    impedances=None)

print(networks.NETWORKS.external_nodeids)

dset = dataset.SemcogDataset("data/semcog_data.h5")

parcels = dset.parcels
parcels['x'] = parcels.centroid_x
parcels['y'] = parcels.centroid_y
parcels = networks.NETWORKS.addnodeid(parcels)
dset.save_tmptbl("parcels", parcels)
nodes = networks.from_yaml(dset, "networks.yaml")

print(networks.NETWORKS.external_nodeids)


# In[1]:


from urbansim.utils import misc, networks
import os
import urbansim.sim.simulation as sim
import dataset

@sim.model('build_networks')
def build_networks(parcels):
    if networks.NETWORKS is None:
        networks.NETWORKS = networks.Networks(
            [os.path.join(misc.data_dir(), x) for x in ['osm_semcog.pkl']],
            factors=[1.0],
            maxdistances=[2000],
            twoway=[1],
            impedances=None)
    p = parcels.to_frame()
    p['x'] = p.centroid_x
    p['y'] = p.centroid_y
    parcels = networks.NETWORKS.addnodeid(p)
    sim.add_table("parcels", parcels)
    
sim.run(['build_networks'])

