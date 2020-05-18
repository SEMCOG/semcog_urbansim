#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import pandas as pd
import pickle
nx.__version__


# In[2]:


fn_shape='./roads3.shp'
fn_pickle='modelroads_twoway.pkl'
#for one-way network
weight='AB_DRVTIME'


# In[3]:


net=nx.read_shp(fn_shape)


# In[4]:


dfnodes=pd.DataFrame(net.nodes(),columns=['x','y'])
dfnodes['nodeid']=dfnodes.index.values


# In[5]:


dfnodes.head(5)


# In[6]:


edgelist=[]
for from_, to_, data in net.edges_iter(data=True):
    edgelist.append([from_[0],from_[1],to_[0],to_[1],data[weight]])
#edgelist


# In[8]:


dfedges=pd.DataFrame(edgelist,columns=['from_x','from_y','to_x','to_y','edgeweights'])
dfedges[:1]


# In[9]:


dfedges=pd.merge(dfedges, dfnodes, how='left', left_on=['from_x','from_y'], right_on=['x','y'])


# In[10]:


dfedges=pd.merge(dfedges, dfnodes, how='left', left_on=['to_x','to_y'], right_on=['x','y'], suffixes=('_from', '_to'))


# In[11]:


dfedges.head(2)


# In[12]:


dicpkl={}
dicpkl['edgeids']=dfedges.index.values.astype('int32')
dicpkl['edges']=dfedges[['nodeid_from','nodeid_to']].values.astype('int32')
dicpkl['edgeweights']=dfedges['edgeweights'].values.astype('float32')
dicpkl['nodeids']=dfnodes.index.values.astype('int32')
dicpkl['nodes']=dfnodes[['x','y']].values.astype('float32')


# In[13]:


print dicpkl['edgeids'].size, dicpkl['edges'].size, dicpkl['edgeweights'].size, dicpkl['nodeids'].size, dicpkl['nodes'].size


# In[14]:


with open(fn_pickle, 'wb') as handle:
    pickle.dump(dicpkl, handle)


# In[15]:


pd.read_pickle(fn_pickle)


# In[ ]:




