#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import pandas as pd
import pickle


# In[2]:


#first input shape file has all features, 2nd has limited features such as local roads 
fn_shape='./travel_model_roads_newtime.shp'
fn_local='./travel_model_local.shp'
#will create two pickle files corresponding to above inputs
h5store=pd.HDFStore('./semcog_modelnetwork.h5')
#for later to select the maximum between ab and ba time
weight='AB_DRVTIME' 
weight2='BA_DRVTIME'


# In[3]:


net=nx.read_shp(fn_shape)
df_nodes=pd.DataFrame(net.nodes(),columns=['x','y'])
h5store['nodes']=df_nodes


# In[4]:


net_local=nx.read_shp(fn_local)
df_local=pd.DataFrame(net_local.nodes(),columns=['x','y'])

df_nodes['nodeid']=df_nodes.index.values
#join nodeid from full node list to limited node list
df_local=pd.merge(df_local,df_nodes, how='left', left_on=['x','y'], right_on=['x','y'])
#assign full list node ids to limited nodes
df_local=df_local.set_index('nodeid')
df_local.index.name = None
h5store['local_nodes']=df_local
    


# In[5]:


#create edge data frame with from and to nodes and edge weights
edgelist=[]
for from_, to_, data in net.edges_iter(data=True):
    edgelist.append([from_[0],from_[1],to_[0],to_[1],max(data[weight],data[weight2])])
df_edges=pd.DataFrame(edgelist,columns=['from_x','from_y','to_x','to_y','weight'])


# In[6]:


df_edges.head(2)


# In[7]:


#join nodeid to starting nodes
df_edges=pd.merge(df_edges, df_nodes, how='left', left_on=['from_x','from_y'], right_on=['x','y'])
#join nodeid to ending nodes
df_edges=pd.merge(df_edges, df_nodes, how='left', left_on=['to_x','to_y'], right_on=['x','y'], suffixes=('_from', '_to'))

df_edges.rename(columns={'nodeid_from': 'from', 'nodeid_to': 'to'}, inplace=True)
df_edges=df_edges[['from','to','weight']]


# In[8]:


h5store['edges']=df_edges


# In[9]:


#create edge data frame with from and to nodes and edge weights
edgelist=[]
for from_, to_, data in net_local.edges_iter(data=True):
    edgelist.append([from_[0],from_[1],to_[0],to_[1],max(data[weight],data[weight2])])
df_edges2=pd.DataFrame(edgelist,columns=['from_x','from_y','to_x','to_y','weight'])


# In[10]:


df_local['nodeid']=df_local.index.values
#join nodeid to starting nodes
df_edges2=pd.merge(df_edges2, df_local, how='left', left_on=['from_x','from_y'], right_on=['x','y'])
#join nodeid to ending nodes
df_edges2=pd.merge(df_edges2, df_local, how='left', left_on=['to_x','to_y'], right_on=['x','y'], suffixes=('_from', '_to'))

df_edges2.rename(columns={'nodeid_from': 'from', 'nodeid_to': 'to'}, inplace=True)
df_edges2=df_edges2[['from','to','weight']]


# In[11]:


h5store['local_edges']=df_edges2


# In[12]:


h5store.close()


# In[12]:




