#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import pandas as pd
import pickle
nx.__version__


# In[2]:


#first input shape file has all features, 2nd has limited features such as local roads 
fn_shapes=['./travel_model_roads_newtime.shp','./travel_model_local.shp'] 
#will create two pickle files corresponding to above inputs
fn_pkls=['modelroads_twoway.pkl','local_nodes.pkl' ] 
#for later to select the maximum between ab and ba time
weight='AB_DRVTIME' 
weight2='BA_DRVTIME'


# In[3]:


#read two shape files
nets=[nx.read_shp(shape) for shape in fn_shapes]


# In[12]:


#create node dataframe and assign index value as nodeid
df_nodes=[]
for net in nets:
    df=pd.DataFrame(net.nodes(),columns=['x','y'])
    df['nodeid']=df.index.values
    df_nodes.append(df)


# In[13]:


if len(fn_shapes)>1:
    #join nodeid from full node list to limited node list
    df_nodes[1]=pd.merge(df_nodes[1], df_nodes[0], how='left', left_on=['x','y'], right_on=['x','y'],suffixes=('', '_all'))
    #assign full list node ids to limited nodes
    df_nodes[1]['nodeid']=df_nodes[1]['nodeid_all']
    df_nodes[1]=df_nodes[1].drop('nodeid_all', 1)
    
    with open(fn_pkls[1], 'wb') as handle:
        pickle.dump(df_nodes[1], handle)
    


# In[15]:


i=0 #process first network only 
#create edge data frame with from and to nodes and edge weights
edgelist=[]
for from_, to_, data in nets[i].edges_iter(data=True):
    edgelist.append([from_[0],from_[1],to_[0],to_[1],max(data[weight],data[weight2])])
df_edges=pd.DataFrame(edgelist,columns=['from_x','from_y','to_x','to_y','edgeweights'])

#join nodeid to starting nodes
df_edges=pd.merge(df_edges, df_nodes[i], how='left', left_on=['from_x','from_y'], right_on=['x','y'])
#join nodeid to ending nodes
df_edges=pd.merge(df_edges, df_nodes[i], how='left', left_on=['to_x','to_y'], right_on=['x','y'], suffixes=('_from', '_to'))

#create dictionary to store edges and nodes
dicpkl={}
dicpkl['edgeids']=df_edges.index.values.astype('int32')
dicpkl['edges']=df_edges[['nodeid_from','nodeid_to']].values.astype('int32')
dicpkl['edgeweights']=df_edges['edgeweights'].values.astype('float32')
dicpkl['nodeids']=df_nodes[i].index.values.astype('int32')
dicpkl['nodes']=df_nodes[i][['x','y']].values.astype('float32')

print('pickle'+str(i), [(key, dicpkl[key].size) for key in list(dicpkl.keys())])

#save to pickle 
with open(fn_pkls[i], 'wb') as handle:
    pickle.dump(dicpkl, handle)

pd.read_pickle(fn_pkls[i]) #read to verify


# In[16]:




