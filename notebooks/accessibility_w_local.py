#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pandana as pdna
import time


# In[2]:


h5store=pd.HDFStore('./semcog_modelnetwork.h5', 'r')

nodes=h5store['nodes']
edges=h5store['edges']
local_nodes=h5store['local_nodes']
local_edges=h5store['local_edges']
local_nodes.index.name = None


# In[3]:


local_nodes.head(2)


# In[4]:


pdna.network.reserve_num_graphs(2)


# In[5]:


#build full network
net_full=pdna.Network(nodes["x"], nodes["y"], edges["from"], edges["to"],edges[["weight"]])
net_full.precompute(30)


# In[6]:


#build local network
net_local=pdna.Network(local_nodes["x"], local_nodes["y"], local_edges["from"], local_edges["to"],local_edges[["weight"]])
net_local.precompute(30)


# In[8]:


#get parcels from data
pstore=pd.HDFStore('../data/semcog_data.h5','r')
parcels=pstore.parcels


# In[11]:


t1=time.time()
x, y = parcels.centroid_x, parcels.centroid_y
parcels["node_ids"] = net_local.get_node_ids(x, y)
print time.time()-t1


# In[14]:


parcels[parcels["node_ids"] ==40]


# In[16]:


net_full.set(parcels['node_ids'], variable=parcels.land_value, name="landv")
ave_landv = net_full.aggregate(10, type="ave", decay="flat",name="landv")


# In[17]:


df_out=pd.DataFrame(ave_landv,columns=["landv"])
df_out.head()


# In[19]:


pd.merge(parcels, df_out, left_on="node_ids", right_index=True, how='left').to_csv('result.csv')


# In[4]:


##spatial join using rtree

# from rtree import index
# indx = index.Index()

# #get local nodes and insert into spatial index
# df_local=h5store.local_nodes
# for nid, x,y in zip(df_local.index.values,df_local['x'],df_local['y']):
#     indx.insert(nid, (x, y))

# # nearest neighbor search for all parcels
# nodelist=[]
# for x,y in zip(parcels.centroid_x,parcels.centroid_y):
#     nodelist.append(list(indx.nearest((x,y), 1))[0])      

# parcels['node_ids']=pd.Series(nodelist,index=parcels.index)


# In[ ]:




