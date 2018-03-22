import os
import random
import pandas as pd
from urbansim.utils import misc
import orca
import verify_data_structure


@orca.injectable('year')
def year():
    default_year = 2015
    try:
        iter_var = orca.get_injectable('iter_var')
        if iter_var is not None:
            return iter_var
        else:
            return default_year
    except:
        return default_year


orca.add_injectable("transcad_available", False)

orca.add_injectable("emp_btypes", [1, 3, 4, 5, 6, 7, 8, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 35, 38, 39])

# maps building type ids to general building types; reduces dimensionality

# keys: binging type
# vause: network filter landues general_type
orca.add_injectable("building_type_map", {
    11: "educational",
    12: "cultural",
    13: "religious",
    14: "governmental",
    21: "retail",
    22: "automotive",
    23: "office",
    24: "financial",
    25: "accommodations",
    26: "eating",
    31: "manufacturing",
    32: "wholesale",
    33: "storage",
    41: "transportation",
    42: "delivery",
    43: "parking",
    51: "healthcare",
    52: "hospital",
    53: "nursing",
    61: "theater",
    62: "recreation",
    71: "agricultural",
    81: "residential",
    82: "residential",
    83: "residential",
    84: "residential",
    99: "accessory",
})

###

##current building types
##

###





# this maps building "forms" from the developer model
# to building types so that when the developer builds a
# "form" this can be converted for storing as a type
# in the building table - in the long run, the developer
# forms and the building types should be the same and the
# developer model should account for the differences

# keys: from proforma forms
# valus: biling typs aplyed to parcelses
orca.add_injectable("form_to_btype", {
    "retail": [21],
    "automotive": [22],
    "office": [23],
    "financial": [24],
    "accommodations": [25],
    "eating": [26],
    "manufacturing": [31],
    "wholesale": [32],
    "healthcare": [51],
    "hospital": [52],
    "nursing": [53],
    "recreation": [62],
    "residential": [81, 82, 83],
})

# orca.add_injectable("form_to_btype", {
#     'residential': [81, 82, 83],
#     'industrial': [31, 32],
#     'retail': [21, 22, 25, 26, 62],
#     'office': [23, 24],
#     'medical': [51, 52, 53],
#     'mixedresidential': [21, 22, 81, 83],
#     'mixedoffice': [21, 22, 81, 83],
# })



seed = 271828
print "using seed", seed
random.seed(seed)
pd.np.random.seed(seed)


def verify():
    hdf_store = pd.HDFStore(os.path.join(misc.data_dir(), "all_semcog_data_02-02-18.h5"), mode="r")

    new = verify_data_structure.yaml_from_store(hdf_store)
    with open(r"configs/data_structure.yaml", "w") as out:
        out.write(new)

    return hdf_store


orca.add_injectable("store", verify())
