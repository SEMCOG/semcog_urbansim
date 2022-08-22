#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic("matplotlib", "inline")
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

import models, utils
import orca
import pandas as pd
from urbansim.utils import misc, networks


# In[ ]:


orca.run(["build_networks"])


# In[ ]:


# orca.run(["neighborhood_vars",
#         ], iter_vars=[2011], data_out=utils.get_run_filename(), out_interval=1)


# In[ ]:


# nodes=orca.get_table('nodes')
# nodes=nodes.to_frame()
# parcels=orca.get_table('parcels')
# parcels=parcels.to_frame().set_index('_node_id')


# In[ ]:


# parcels=orca.get_table('parcels')
# parcels=parcels.to_frame().set_index('_node_id')
# parcels=pd.merge(parcels, nodes, left_index=True, right_index=True, how='left')
# parcels.columns


# In[ ]:


# for gentype in ['residential1','residential2','retail', 'office', 'industrial','medical']:
#    pp=parcels[parcels[gentype]>0]
#    print gentype
#    #pp[gentype].plot(kind='hist',bins=20)
#    print pp[gentype].value_counts(sort=False,ascending=True, bins=20)


# In[ ]:


# nn=nodes[nodes.residential2>0]
# nn.residential2.plot(kind='hist',bins=20)
# nn.residential2.value_counts(sort=False,ascending=True, bins=20)


# In[ ]:


# pp=parcels[parcels.residential1>0]
# pp.residential1.plot(kind='hist',bins=20)
# pp.residential1.value_counts(sort=False,ascending=True, bins=20)


# In[ ]:


# nodes.columns


# In[ ]:


# parcels=orca.get_table('parcels')
# parcels=parcels.to_frame().set_index('_node_id')
# parcels=pd.merge(parcels, nodes, left_index=True, right_index=True, how='left')
# parcels.columns


# In[ ]:


orca.run(
    [
        #     "clear_cache",            # recompute variables every year
        "neighborhood_vars",  # neighborhood variables
        #     "households_transition",  # households transition
        #     "households_relocation",  # households relocation model
        #     "jobs_transition",        # jobs transition
        #     "jobs_relocation",        # jobs relocation model
        #     "scheduled_development_events",
        #     "nrh_simulate",           # non-residential rent hedonic
        #     "rsh_simulate",           # residential sales hedonic
        #     "hlcm_simulate",          # households location choice
        #     "elcm_simulate",          # employment location choice
        "price_vars",  # compute average price measures
        "feasibility",  # compute development feasibility
        "residential_developer",  # build actual buildings,
        #     "non_residential_developer",   # build actual buildings
        #     "government_jobs_scaling_model",
        #     "gq_pop_scaling_model",
        #     "refiner",
        #     "travel_model",
    ],
    iter_vars=list(range(2016, 2017)),
    data_out=utils.get_run_filename(),
    out_interval=1,
)


# In[ ]:


bb = orca.get_table("buildings").to_frame()


# In[ ]:


bb.columns


# In[ ]:


nb = bb[bb.year_built > 2015]


# In[ ]:


parcels = orca.get_table("parcels").to_frame()


# In[ ]:


newblg = pd.merge(nb, parcels, left_on="parcel_id", right_index=True, how="left")


# In[ ]:


newblg.columns.values


# In[ ]:


newblg.groupby(["year_built", "city_id_y"]).residential_units.sum()


# In[ ]:


newblg[newblg.sev_value != 0].groupby(["year_built", "city_id_y"]).sev_value.mean()


# In[ ]:


newblg.sev_value.plot(kind="hist", bins=100, range=[0, 5e5])


# In[ ]:


newblg.to_csv("newbuildings.csv")


# In[ ]:


# In[ ]:


bb.columns


# In[ ]:


bb[bb.building_type_id == 81].market_value.plot(kind="hist", bins=100, range=[0, 1e6])


# In[ ]:


lmhouse = bb[
    (bb.building_type_id == 81) & (bb.market_value < 1e5) & (bb.market_value > 100)
]


# In[ ]:


(lmhouse.market_value / lmhouse.building_sqft).plot(kind="hist", bins=100)


# In[ ]:


lmhouse.year_built.plot(kind="hist", bins=100, range=[1920, 2010])


# In[ ]:


bb[
    (bb.building_type_id == 81) & (bb.year_built > 1850) & (bb.market_value < 1.5e6)
].plot("year_built", "market_value", "scatter")


# In[ ]:


bb[
    (bb.building_type_id == 81)
    & (bb.year_built > 1930)
    & (bb.year_built < 1980)
    & (bb.market_value < 0.15e6)
].plot("year_built", "market_value", "hexbin")


# In[ ]:


bb[(bb.building_type_id.isin([80, 81, 82, 83, 84])) & (bb.market_value > 0)].groupby(
    "city_id"
).market_value.mean()


# In[ ]:


bb[(bb.building_type_id.isin([80, 81, 82, 83, 84])) & (bb.market_value > 0)].groupby(
    "city_id"
).market_value.median()


# In[ ]:


cities = orca.get_table("cities").to_frame()


# In[ ]:


cities.columns


# In[ ]:


cities.loc[bb.groupby("city_id").market_value.mean().index]


# In[ ]:


# In[ ]:


# # basic output tables

# In[ ]:


# nodes
nodes = orca.get_table("nodes").to_frame()
nodes.columns


# In[ ]:


nodes.columns


# In[ ]:


# nodes[['new_far','sum_total_sqft','sum_parcel_size']]


# In[ ]:


# feasibility
feasi = orca.get_table("feasibility")
# feasi.columns


# In[ ]:


# parcels
parcels = orca.get_table("parcels").to_frame()
parcels.columns


# In[ ]:


parcels.land_use_type_id.unique()


# In[ ]:


orca.list_tables()


# In[ ]:


# buildings
buildings = orca.get_table("buildings").to_frame()
buildings.columns


# In[ ]:


# new buildings
newbldgs = pd.read_csv("new_buildings.csv")
# newbldgs=buildings[buildings.year_built==2016]
# newbldgs


# In[ ]:


# zoning
zoning = orca.get_table("zoning").to_frame()
zoning.columns


# In[ ]:


lone_house = orca.get_injectable("lone_house").to_frame()
# lone_house


# In[ ]:


pcllone = pd.merge(parcels, lone_house, left_index=True, right_index=True, how="left")


# In[ ]:


pcllone.to_csv("parcel_lone_house.csv")


# # Join network based indicators to ...

# In[ ]:


# to parcels
pclnode = pd.merge(parcels, nodes, left_on="_node_id", right_index=True, how="left")
# pclnode.to_csv('parcel_nodes.csv')


# In[ ]:


pclnode.to_csv("parcel_nodes_lu_far.csv")


# In[ ]:


# to buildings
bldnode = pd.merge(
    buildings, pclnode, left_on="parcel_id", right_index=True, how="left"
)


# In[ ]:


# to new buildings
newbnode = pd.merge(
    newbldgs, pclnode, left_on="parcel_id", right_index=True, how="left"
)


# In[ ]:


pclnode["far1"] = pclnode.total_sqft / pclnode.parcel_size


# In[ ]:


pclnode.columns


# In[ ]:


pclnode[["new_far", "max_far", "parcel_size", "total_sqft"]].head()


# In[ ]:


pclnode[pclnode["new_far"] > pclnode["max_far"]]


# In[ ]:


pclnode[pclnode["new_far"] < (pclnode["far1"] - 0.01)][["new_far", "far1", "_node_id"]]


# In[ ]:


pclnode[pclnode["new_parcel_size"] < pclnode["parcel_size"]][
    ["new_parcel_size", "parcel_size"]
]


# In[ ]:


pclnode[(pclnode["new_pct_undev"]) < pclnode["pct_undev"]][
    ["new_pct_undev", "pct_undev"]
]


# # analysis

# In[ ]:


# recent buildings correlates to
bldrecent = bldnode[(bldnode.year_built > 2011) & (bldnode.sqft_per_unit > 0)]
bldrecent.shape
cort = bldrecent.corr()


# In[ ]:


# sqft_per_unit
# cort['sqft_per_unit']
# bldrecent.plot(x='sqft_per_unit', y='ave_unit_sqft')


# In[ ]:


# select parcels
parcels.loc[1012389]


# In[ ]:


# office use values counts
npp.office.value_counts(sort=True, bins=50)


# In[ ]:


# histogram for office prices based on network module
npp = nodes_prices[nodes_prices.medical > 0]
npp.medical1k.plot(kind="hist", bins=200)


# In[ ]:


# histogram for office prices based on network module
npp = nodes_prices[nodes_prices.office1k > 0]
npp.office1k.plot(kind="hist", bins=200)


# In[ ]:


pp = parcels[parcels.land_use_type_id == 11]
pp.parcel_sqft.plot(kind="hist", bins=40, range=[0, 100000])
pp.parcel_sqft.value_counts(sort=True, bins=50)


# In[ ]:


pd.unique(parcels.land_use_type_id)


# In[ ]:


ps = (
    buildings[(buildings.building_type_id == 81) & (buildings.residential_units == 1)]
    .groupby(by="parcel_id")
    .building_type_id.count()
    == 1
)


# In[ ]:


parcels.head()


# In[ ]:


# # fix hdf5 data

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


hf = pd.HDFStore("./data/alderaan_semcog_data_fix.h5")


# In[ ]:


hf


# In[ ]:


hf["buildings"] = hf["buildings"][hf["buildings"]["building_type_id"].notnull()]
hf["buildings"] = hf["buildings"][hf["buildings"]["market_value"].notnull()]


# In[ ]:


lacal = hf["buildings"]
for c in lacal.columns:
    print(c, lacal[c].dtype)
    if lacal[c].dtype == np.float64:
        lacal[c] = lacal[c].astype(np.int32)
del hf["buildings"]
hf["buildings"] = lacal


# In[ ]:


hf["/zoning"]["future_use"].unique()


# In[ ]:


for item in list(hf.keys()):
    print(item, hf[item].shape)


# In[ ]:


b = hf["/buildings"]


# In[ ]:


del b["owner_units"]


# In[ ]:


hf["/buildings"] = b


# In[ ]:


af = hf["/zoning"]


# In[ ]:


af["max_far"] = af["max_far"] / 100.0


# In[ ]:


af.head()


# In[ ]:


hf["/buildings"] = af


# In[ ]:


af.head()


# In[ ]:


hf["/zoning"] = af


# In[ ]:


hf.close()


# In[ ]:


# for t in hf.keys():
#     df = hf[t]
#     print
#     print t
#     print df.describe()

# building_types
# land_use_types

hf["building_types"]


# In[ ]:

