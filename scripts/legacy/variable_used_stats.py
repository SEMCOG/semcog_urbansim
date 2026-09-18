# run this script in "semcog_urbansim" folder
# extracts all variables used in yaml files

# In[ ]:

import yaml


###
# extract all variables used and save the counts
# %%
from os import listdir
from os.path import isfile, join
from collections import Counter

fpath = "./configs/repm"  # repm
fpath = "./configs/hlcm_calib"  # hlcm calib
fpath = "./configs/hlcm/large_area_income_quartile"  # income quartile
fpath = "./configs/elcm/large_area_sector"  # income quartile
onlyfiles = [f for f in listdir(fpath) if isfile(join(fpath, f))]

# %%
lstv = []
for f in onlyfiles:
    with open(join(fpath, f), "r") as file:
        ym = yaml.safe_load(file)
        if isinstance(ym["model_expression"], dict):
            expss = ym["model_expression"]["right_side"]
        else:
            expss = ym["model_expression"]

        if isinstance(expss, list):
            lstv += expss
        else:
            lstv += expss.replace(" ", "").split("+")

dfc = pd.DataFrame(Counter(lstv).items(), columns=["variables", "counts"])
dfc.sort_values(by="counts", ascending=False).to_csv(
    fpath.replace("./", "").replace("/", "_") + ".csv"
)


#%%
"right_side" in ym["model_expression"].keys()


# %%
ym
# %%
