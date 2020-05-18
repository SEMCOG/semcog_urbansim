#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import models
import urbansim.sim.simulation as sim
from urbansim.maps import dframe_explorer


# In[2]:


d = {tbl: sim.get_table(tbl).to_frame() for tbl in ['buildings', 'jobs', 'households']}


# In[3]:


dframe_explorer.start(d, 
        center=[37.7792, -122.2191],
        zoom=11,
        shape_json='data/zones.json',
        geom_name='ZONE_ID', # from JSON file
        join_name='zone_id', # from data frames
        precision=2)


# [Click here to explore dataset](http://localhost:8765)

# In[ ]:




