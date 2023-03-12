#%%
import orca
import shutil

import os
import models, utils
from urbansim.utils import misc, networks
import time
import output_indicators

data_out = utils.get_run_filename()
print(data_out)

import sys

#%%
f = os.statvfs("/home")
freespace = f.f_bavail * f.f_bsize / (1048576 * 1024.0)
print("freespace:", freespace)
if freespace < 10:
    print(freespace, "GB available. Disk space is too small, stop running")
    sys.exit()

start_time = time.ctime()

#%%
## ================review input tables  =======================================

#%%
orca.list_tables()

# orca.run(["households_transition"], iter_vars=list(range(2020, 2025)))
orca.run(
    ["build_networks_2050", "neighborhood_vars"]  # "refiner",
    #    + orca.get_injectable("repm_step_names")
    #    + ["increase_property_values"]  # In place of ['nrh_simulate', 'rsh_simulate']
)  # increase feasibility based on projected income

#%%
tabl = "events_addition"
df = orca.get_table(tabl).to_frame(orca.get_table(tabl).local_columns)

#%%
tabl = "parcels"
dfp = orca.get_table(tabl).to_frame(orca.get_table(tabl).local_columns)


#%%
#
# #%%
dfa = dfp.reindex(df.parcel_id)[["city_id", "large_area_id"]]


#%%
import pandas as pd
import os

data_path = r"/home/da/share/urbansim/RDF2050/model_inputs/base_hdf"
if os.path.exists(data_path) == False:
    data_path = "/home/da/share/U_RDF2050/model_inputs/base_hdf"
hdf_list = [
    (data_path + "/" + f)
    for f in os.listdir(data_path)
    if ("forecast_data_input" in f) & (f[-3:] == ".h5")
]
hdf_last = max(hdf_list, key=os.path.getctime)
print(hdf_last)
#%%
# st=pd.HDFStore("data/all_semcog_data_02-02-18-final-forecast_newbid.h5", "r")

#%%
import pandas as pd
import os


st0 = pd.HDFStore(hdf_last, "r")
#%%
st0["households"].children.sum()
#%%

#%%


#%%

#%%

st0["mcd_total"]
#%%
st0["jobs_2019"].loc[st0["jobs_2019"].sector_id == 16]
#%%
st["jobs"].loc[st["jobs"].sector_id == 16]

#%%
## ============================================================

import pandas as pd
import os

st = pd.HDFStore("runs/run2095.h5", "r")

#%%
year = "2021"
b = st[f"{year}/buildings"].merge(
    st[f"{year}/parcels"][["city_id", "semmcd"]],
    left_on="parcel_id",
    right_index=True,
    how="left",
)
#%%
h = st[f"{year}/households"].merge(
    b[["semmcd"]], left_on="building_id", right_index=True, how="left"
)
#%%
h_base = st["base/households"].loc[st["base/households"].building_id >= 90000000]
#%%
h = pd.concat([h, h_base], axis=0)
#%%


#%%
hmcd = h.groupby("semmcd").size()
#%%
hmcd.loc[1025]
#%%
st["base/annual_household_control_totals"].loc[2025].total_number_of_households.sum()

#%%
import numpy as np

hctrl_0 = st["base/annual_household_control_totals"]
hctrl = hctrl_0.loc[2022]
hctrl = hctrl.replace(-1, np.inf)
#%%

count = 0
qlist = []
hind = []
for ind, r in hctrl.iterrows():

    qstr = "".join(
        [
            f"(large_area_id=={r.large_area_id} )&(race_id=={r.race_id}) &",
            f"(age_of_head>={r.age_of_head_min}) & (age_of_head<={r.age_of_head_max}) &",
            f"(persons>={r.persons_min}) & (persons<={r.persons_max}) &",
            f"(children>={r.children_min}) & (children<={r.children_max}) &",
            f"(cars>={r.cars_min}) & (cars<={r.cars_max}) &",
            f"(workers>={r.workers_min}) & (workers<={r.workers_max}) &",
            f"(income>={r.income_min}) & (income<={r.income_max})",
        ]
    )
    hqq = h.query(qstr)

    hqq = h.query(qstr)
    hind.append(hqq.index)
    lenq = len(hqq)
    if lenq == 0:
        print(qstr)
        print(r.total_number_of_households)
        qlist.append(qstr)
    count += 1
    if count % 1000 == 0:
        print("\n")
        print(count, lenq, end=" |")
#%%
st["base/households"].query(
    "(large_area_id==3.0 )&(race_id==1.0) &(age_of_head>=18.0) & (age_of_head<=24.0) &(persons>=3.0) & (persons<=3.0) &(children>=1.0) & (children<=1.0) &(cars>=1.0) & (cars<=1.0) &(workers>=1.0) & (workers<=1.0) &(income>=64018.0)"
)
#%%

#%%

#%%

#%%

#%%


#%%
qg = pd.merge(
    st0["2025/group_quarters"],
    st0["2025/buildings"][["city_id"]],
    left_on="building_id",
    right_index=True,
    how="left",
)
gqc = qg.groupby("city_id").size()

#%%
qctrl = st0["base/group_quarters_control_totals"]

#%%
qctrl = qctrl[qctrl.year == 2025]["count"]

#%%

(gqc - qctrl).sum()


#%%
gqc.loc[551]
#%%
qctrl.loc[551]
#%%


city_large = (
    st0["base/parcels"][["city_id", "large_area_id"]]
    .drop_duplicates()
    .set_index("city_id")
)
#%%
for ind, row in city_large.iterrows():
    print(ind, row)
#%%
qctrl


#%%

#%%

#%%
regiont = (
    st0["/annual_household_control_totals"]
    .groupby(["year", "large_area_id"])
    .total_number_of_households.sum()
    .unstack()
    .T
)
#%%
mcdt = st0["/mcd_total"]
la_map = {0: 5, 1: 3, 2: 125, 3: 99, 4: 161, 5: 115, 6: 147, 7: 93}
mcdt["large_area_id"] = mcdt.index // 1000
mcdt["large_area_id"] = mcdt["large_area_id"].map(la_map)

#%%
mcdt = mcdt.groupby("large_area_id").sum()
#%%
mcdt.groupby("large_area_id").sum() - regiont
#%%
regiont

#%%
mcdt.columns = [int(c) for c in mcdt.columns]


#%%
regiont - mcdt

#%%

#%%


#%%
for k in st0.keys():
    print(k)
    st[k] = st0[k]
#%%
st["households"].loc[
    st["households"].building_id.isin(st["pseudo_building_2020"].index)
]


#%%
st0.close()
st.close()

#%%
st0 = pd.HDFStore(hdf_last, "r")
# st = pd.HDFStore('runs/run2008.h5')

#%%
"/jobs" in st0.keys()
#%%
[n for n in st0.keys() if "/jobs" in n]

#%%
jj = pd.read_csv(
    "/home/da/share/urbansim/RDF2050/model_inputs/base_tables/jobs_12162022.csv"
)
#%%


#%%
jo = pd.read_csv(
    "/home/da/share/urbansim/RDF2050/model_inputs/base_tables/tests/final_jobs_table_2019_rounded_20230125_new.csv"
)

#%%
st0["buildings"]
#%%
jo
#%%
jo.merge(st0["buildings"], left_on="building_id", right_index=True, how="inner")
# .to_csv("/home/da/share/urbansim/RDF2050/model_inputs/base_tables/tests/employment_2019_nosqftjob_in_2020buildings.csv")
#%%
jo.loc[~jo.building_id.isin(st0["buildings"].index)]

#%%
jo.loc[jo.building_id.isin(st0["buildings"].index)].to_csv(
    "/home/da/share/urbansim/RDF2050/model_inputs/base_tables/tests/employment_2019_nosqftjob_in_2020buildings.csv"
)

#%%
jo.merge(
    st0["buildings"], left_on="building_id", right_index=True, how="inner"
).building_type_id.value_counts()

#%%
jj = pd.merge(
    st0["jobs_2019"],
    st0["buildings"][["parcel_id"]],
    left_on="building_id",
    right_index=True,
    how="left",
)
# df = orca.get_table(tabl).to_frame()

#%%

jj.loc[jj.parcel_id.isnull()].to_csv(
    "/home/da/share/urbansim/RDF2050/model_review/employment/jobs_2019_invalid_bid.csv"
)

#%% ====================================
#%%
import pandas as pd
import os

st1 = pd.HDFStore("runs/run2098_zonal.h5")

#%%
pd.read

#%%
st1["base/households"]
#%%
st1["base/households"]
#%%

#%%

#%%

#%%

#%%


#%%
hh25 = st1["2025/households"].merge(
    st1["2025/buildings"][["city_id", "residential_units"]],
    left_on="building_id",
    right_index=True,
    how="left",
)

buildings.residential_units.sub(households.building_id.value_counts(), fill_value=0)

#%%
vu = st1["2025/buildings"].residential_units.sub(
    st1["2025/households"].building_id.value_counts(), fill_value=0
)


#%%
bb = st1["2025/buildings"]
bb["vu"] = vu

#%%
bb.groupby("city_id").vu.sum().loc[1130]

#%%
bb.loc[(bb.city_id == 1130) & (bb.vu > 0)]
#%%

#%%
df.loc[df.year_built == 2020]

#%%
dfr = orca.get_table("refiner_events").to_frame(
    orca.get_table("refiner_events").local_columns
)
dfr.head()

#%%
import time

start_time = time.time()
data_out = "/runs/ss.h5"

run_info = f"""data_out: {data_out} \
            \nRun number: {os.path.basename(data_out.replace('.h5', ''))} \
            \nStart time: {time.ctime(start_time)}"""
print(run_info)


#%%

#%%
##======= update run number =======
import os

f = open(os.path.join(os.getenv("DATA_HOME", "."), "RUNNUM"), "r")
num = int(f.read())
print(f)
print(num)

#%%
f = open(os.path.join(os.getenv("DATA_HOME", "."), "RUNNUM"), "w")
f.write("2000")
f.close()


#%%
# =====================make resume HDF=============================================
# make resume HDF
#%%
import pandas as pd

# stf = pd.HDFStore("/home/da/share/U_RDF2050/model_inputs/base_hdf/forecast_data_input_121422.h5", "r")
# stf = pd.HDFStore("/home/da/share/U_RDF2050/model_runs/run336.h5")
stf = pd.HDFStore("runs/run2032.h5", "r")

#%%
j9 = stf["base/jobs_2019"].merge(
    stf["base/buildings"][["parcel_id"]],
    left_on="building_id",
    right_index=True,
    how="left",
)
#%%
j9 = j9.merge(stf["base/parcels"], left_on="parcel_id", right_index=True, how="left")
#%%
stf["base/buildings"]
#%%

#%%

#%%

st33 = pd.HDFStore("data/forecast_data_input_121822_2033.h5")
# for f in stf.keys():
#     if "/base/" in f:
#         print(f)
#         st33[f.replace("/base/", "")] = stf[f]
for f in stf.keys():
    if "/2033/" in f:
        print(f)
        st33[f.replace("/2033/", "")] = stf[f]
st33.close()


#%%
st33 = pd.HDFStore("data/forecast_data_input_121822_2033.h5", "r")
st33.keys()

#%%
st33.close()

#%%
import numpy as np

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
print(c)
#%%
df["a"] = np.nan
#%%
df.to_csv("nan.csv")

#%%


#%%

#%%


#%%
# ==================================================================
# check model run results
import pandas as pd
import orca
import numpy as np

#%%
dfp = pd.read_csv(
    "/home/da/share/U_RDF2050/population_synthesis/2020/run4_11302022/output/synthetic_households_backup.csv"
)
# import variables

#%%
dfp.columns

#%%
st_run = pd.HDFStore("./runs/run2021.h5", "r")

#%%
b = st_run["base/buildings"]
# b = st_run["2050/buildings"].merge(st_run["2050/parcels"][["city_id"]], left_on="parcel_id", right_index=True, how='left')

#%%
b.loc[(b.building_type_id == 81) & (b.non_residential_sqft > 0)]

#%%
b.head()

#%%

#%%

#%%
#%%

#%%


st_in = pd.HDFStore(
    "/home/da/share/U_RDF2050/model_inputs/base_hdf/forecast_data_input_011023.h5", "r"
)
st_run = pd.HDFStore("./runs/run2006_indicators.h5", "r")

#%%
hh = st_in["households"]
pp = st_in["persons"]
hh_cols = hh.columns
pp_cols = pp.columns
gq = st_in["/group_quarters"]
gq_hh = st_in["/group_quarters_households"]
gq_join = gq.join(gq_hh[set(gq_hh.columns) - set(gq.columns)])
gq_join["blkgrp"] = -1

#%%
print(hh_cols)
print(pp_cols)
print(gq.columns)
print(gq_hh.columns)
print(gq_join.columns)

#%%
gq_join[pp_cols]


#%%
gq_join = gq.join(gq_hh[set(gq_hh.columns) - set(gq.columns)])
gq_join["type"] = 3  # noninstitutional GQ
gq_join.loc[gq_join.gq_code < 500, "type"] = 2  # institutional GQ

gq_join["household_id"] = list(range(8_000_000, 8_000_000 + len(gq_join)))
gq_join["person_id"] = gq_join["household_id"]
gq_join["member_id"] = 1
gq_join["relate"] = 0

gq_join["worker"] = gq_join["workers"]
for at in ["blkgrp", "sex", "school_grade", "naicsp", "industry"]:
    gq_join[at] = np.nan

#%%

gq_all = gq_join.copy()
gq_all = gq_all.set_index("person_id")
gq_person = gq_all[pp_cols].fillna(-9)

gq_join = gq_join.set_index("household_id")
gq_hh = gq_join[hh_cols].fillna(-9)

#%%
gq_person


#%%
gq_join[pp_cols]

#%%
st_in.keys()

#%%
gq

#%%

#%%

#%%


#%%
range(8_000_000, len(gq_join))

#%%
st_run.keys()

#%%
st_run["base/jobs_2019"] = st_in["jobs_2019"]
print(df.columns.str.startswith("sec"))
df.columns[~df.columns.str.startswith("sec")]

#%%
list(df.columns[df.columns.str.startswith("sec")]) + ["a", "b"]
#%%
# st_run["base/jobs_2019"] = st_in["jobs_2019"]
st_run.close()
st_in.close()

#%%
orca.add_table("households", st_in["households"])
orca.get_table("households").to_frame(
    orca.get_table("households").local_columns + ["large_area_id"]
)
#%%

#%%
st_run.keys()

#%%
st_run["2040/jobs"].loc[st_run["2040/jobs"].building_id.isnull()]
#%%
st_in["events_deletion"].loc[st_in["events_deletion"].year_built == 2020]

#%%
for x in st_in["building_types"].index:
    print(x)

#%%
st_in["events_deletion"].loc[st_in["events_deletion"].building_id == 1170068]

#%%
st_run["base/jobs"].loc[st_run["base/jobs"].building_id == -1].sector_id

#%%
orca.add_table("jobs", st_run["base/jobs"])

#%%
st_in["annual_employment_control_totals"].groupby("year").total_number_of_jobs.sum()
#%%


#%%
import orca

#%%
orca.add_table("buildings", st_run["base/buildings"])
orca.add_table("jobs", st_run["base/jobs"])

#%%
[f"jobs_sec_{str(i).zfill(2)}" for i in [1, 2, 3]]
#%%
orca.get_table("buildings").building_type_id
#%%


#%%
#%%
st_run["base/events_addition"]
#%%
df = st_in["events_addition"]
#%%
df.loc[df.parcel_id == 4130003]
#%%
df.loc[df.parcel_id == 4130003, "city_id"] = 4010
df.loc[df.parcel_id == 4130003, "zone_id"] = 2339


#%%
st_in["events_addition"] = df
#%%
st_run["base/events_addition"] = df
#%%


#%%
st_run["base/events_addition"] = df
#%%
st_run.close()
st_in.close()
#%%


#%%
for runn in ["runs/run187.h5", "runs/run188.h5"]:
    print(runn)
    st_add = pd.HDFStore(runn, "r")

    for k in st_add.keys():
        if "/base/" not in k:
            print(k)
            st_run[k] = st_add[k]
    st_add.close()


#%%
st_run.close()

#%%

import os

os.getcwd()
f"{2000}/aa"

#%%
import pandas as pd
import numpy as np

#%%

st = pd.HDFStore("runs/run198.h5", "r")
ctrl_bins = {
    "age_of_head": [0, 5, 18, 25, 35, 65],
    "persons": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "children": [0, 1, 2, 3],
    "cars": [0, 1, 2, 3],
    "workers": [0, 1, 2, 3],
}
#%%

yrs = [str(x) for x in range(2025, 2051, 5)]
hzs = []
for yr in yrs:
    run_hhs = st[yr + "/" + "households"]
    run_hhs["age_of_head"] = np.digitize(
        run_hhs["age_of_head"], ctrl_bins["age_of_head"]
    )
    hzs.append(
        run_hhs.groupby(["large_area_id", "race_id", "age_of_head"])
        .persons.mean()
        .to_frame(f"hh_size_{yr}")
    )


pd.concat(hzs, axis=1).to_csv(
    "../forecast_data_input/forecast_controls/run_la_race_age_hhsize.csv"
)

# bb= pd.merge(st[yr +"/"+ 'buildings'][['parcel_id']],
#              st[yr +"/"+ 'parcels'][['large_area_id']], left_on='parcel_id', right_index=True, how='left')
# syn_hhs= pd.merge(st[yr +"/"+ 'households'],
#              bb[['large_area_id']], left_on='building_id', right_index=True, how='left')
# syn_hhpop = pd.merge(st[yr +"/"+ 'persons'],
#              syn_hhs[['large_area_id']], left_on='household_id', right_index=True, how='left')


#%%

pd.concat(hzs, axis=1)

# .to_csv("y2050_la_race_age_hhsize.csv")

#%%

#%%
dfm = pd.read_csv("./data/mcd_totals_test_1222_la.csv", index_col="mcd")
#%%
st2["/mcd_total"] = dfm
#%%
st2.close()
#%%
from outputs.indicators2 import *


#%%
import os

os.getcwd()

#%%

#%%


#%%
df = st["/annual_relocation_rates_for_households"]
df["probability_of_relocating"] = 0
st["/annual_relocation_rates_for_households"] = df
#%%
st["/annual_relocation_rates_for_households"]

#%%
st2.close()


#%%
st = pd.HDFStore("./runs/run193.h5", "r")
#%%
st["/base/mcd_control_total"]

#%%
year_names = ["yr" + str(i) for i in list(range(2020, 2050 + 1, 1))]
for i, y in list(enumerate(year_names))[::5]:
    print(i, y)


#%%
st.keys()

#%%
mcd_ctrl = st2["mcd_total"]

#%%
la_map = {0: 5, 1: 3, 2: 125, 3: 99, 4: 161, 5: 115, 6: 147, 7: 93}
mcd_ctrl["large_area_id"] = mcd_ctrl.index // 1000
mcd_ctrl["large_area_id"] = mcd_ctrl["large_area_id"].map(la_map)
#%%
mcd_ctrl.groupby("large_area_id").sum()[
    [str(x) for x in [2020, 2025, 2030, 2035, 2040, 2045, 2050]]
]

#%%
import numpy

numpy.show_config()


#%%


#%%
#%%
stf["events_addition"].loc[stf["events_addition"].zone_id > 0]

#%%
stf["demolition_rates"].to_csv(
    "/home/da/share/U_RDF2050/model_inputs/base_tables/demolition_rates.csv"
)

#%%
os.path.join(os.getenv("DATA_HOME", "."), "RUNNUM")
#%%
import orca

#%%
orca.add_injectable("t", "bbadsf")
#%%
orca.get_injectable("t")
#%%
evn = stf["events_addition"]

pcl = stf["parcels"]

#%%
evn.drop("zone_id", axis=1, inplace=True)
evn2 = pd.merge(
    evn, pcl[["zone_id"]], left_on="parcel_id", right_index=True, how="left"
)


#%%
evn2
#%%
stf["events_addition"] = evn2
#%%
stf.close()

#%%
hhs = []
emps = []
for yr in "base" + [str(y) for y in range(2021, 2051)]:
    print(yr)
    bi = stf["/" + str(yr) + "/buildings"].query("sp_filter < 0").index
    hh = stf["/" + str(yr) + "/households"]
    ee = stf["/" + str(yr) + "/jobs"]
    hhs.append(
        hh.loc[hh.building_id.isin(bi)].groupby("building_id").size().to_frame(yr)
    )
    emps.append(
        ee.loc[ee.building_id.isin(bi)].groupby("building_id").size().to_frame(yr)
    )


#%%
#%%

ff = []
for (a, b) in ff:
    print(a, b)


#%%
dfe = dfe.loc[dfe.building_id.isin(blist)].set_index("building_id")


#%%
len(dfe.loc[dfe.building_id.isin(blist)])
#%%
dfe


#%%
import pandas as pd

dfj = pd.merge(dfr, dfe, left_on="event_id", right_index=True, how="left")


#%%
nlist = dfj[dfj.stories.isnull()].event_id_x

#%%
dfb.loc[nlist]
#%%
dfb
#%%
dfj.to_csv("gq_events_with_buildings.csv")

#%%
import pandas as pd

hdf_store = pd.HDFStore("./data/forecast_data_input_120722.h5")


#%%
dfevn = hdf_store["events_deletion"]
#%%
dfevn.loc[dfevn.building_id == 9012231]
#%%
dfb = hdf_store["buildings"]


#%%
dfb.loc[9012231]


#%%
len(hdf_store["events_addition"]) + len(hdf_store["events_deletion"])

#%%
len(hdf_store["events_deletion"])


#%%
df = hdf_store["annual_relocation_rates_for_jobs"]
df.to_csv("annual_relocation_rates_for_jobs.csv")


#%%
df = hdf_store["annual_relocation_rates_for_jobs"]
df.loc[[1, 7, 12, 13, 15, 18], "job_relocation_probability"] = 0

#%%
df["job_relocation_probability"] = df["job_relocation_probability"] / 5
#%%
hdf_store["annual_relocation_rates_for_jobs"] = df
#%%

#%%
hdf_store["annual_relocation_rates_for_jobs"]
#%%

hdf_store.close()
#%%
import time

#%%
a = time.time()
#%%

time.ctime(a)
#%%
time.ctime()
#%%

#%%

#%%

#%%

#%%

# orca.run(
#     [
#         "build_networks_2050",
#         "neighborhood_vars",
#         # "scheduled_demolition_events",
#         # "random_demolition_events",
#         # "scheduled_development_events",
#         # "refiner",
#         # "households_transition",
#         # "fix_lpr",  # await data
#         # "households_relocation_2050",
#         # "jobs_transition",
#         # "jobs_relocation_2050",
#         # "feasibility",
#         # "residential_developer",
#         # "non_residential_developer",
#     ]
#     + orca.get_injectable("repm_step_names")
#     + ["increase_property_values"]  # In place of ['nrh_simulate', 'rsh_simulate']
#     + ["mcd_hu_sampling"]  # Hack to make more feasibility
#     + orca.get_injectable(
#         "hlcm_step_names"
#     )  # disable for now, wait until new estimation
#     + orca.get_injectable("elcm_step_names")
#     + [
#         "elcm_home_based",
#         "jobs_scaling_model",
#         "gq_pop_scaling_model",
#         # "travel_model", #Fixme: on hold
#         "update_bg_hh_increase",
#     ],
#     iter_vars=list(range(2020, 2021)),
#     data_out=data_out,
#     out_base_tables=[
#         "jobs",
#         "base_job_space",
#         "employment_sectors",
#         "annual_relocation_rates_for_jobs",
#         "households",
#         "persons",
#         "annual_relocation_rates_for_households",
#         "buildings",
#         "pseudo_building_2020",
#         "parcels",
#         "zones",
#         "semmcds",
#         "counties",
#         "target_vacancies_mcd",
#         "target_vacancies",
#         "building_sqft_per_job",
#         "annual_employment_control_totals",
#         "travel_data",
#         "travel_data_2030",
#         "zoning",
#         "large_areas",
#         "building_types",
#         "land_use_types",
#         # "workers_labor_participation_rates",
#         "employed_workers_rate",
#         "transit_stops",
#         "crime_rates",
#         "schools",
#         "poi",
#         "group_quarters",
#         "group_quarters_control_totals",
#         "annual_household_control_totals",
#         "remi_pop_total",
#         "events_addition",
#         "events_deletion",
#         "refiner_events",
#     ],
#     out_run_tables=[
#         "buildings",
#         "jobs",
#         "base_job_space",
#         "parcels",
#         "households",
#         "persons",
#         "group_quarters",
#         "dropped_buildings",
#     ],
#     out_interval=1,
#     compress=True,
# )

# output_indicators.main(data_out)
# print("Simulation started at %s, finished at %s. " % (start_time, time.ctime()))

# dir_out = data_out.replace('.h5', '')
# shutil.copytree(dir_out, '/mnt/hgfs/U/RDF2045/model_runs/' + os.path.basename(os.path.normpath(dir_out)))
# shutil.copy(data_out, '/mnt/hgfs/J')
