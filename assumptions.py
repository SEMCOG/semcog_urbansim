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
    11: "Institutional",
    12: "Institutional",
    13: "Institutional",
    14: "Institutional",
    21: "Retail",
    22: "Retail",
    23: "Office",
    24: "Office",
    25: "Retail",
    26: "Retail",
    31: "Industrial",
    32: "Industrial",
    33: "TCU",
    41: "TCU",
    42: "TCU",
    43: "TCU",
    51: "Medical",
    52: "Medical",
    53: "Medical",
    61: "Institutional",
    62: "Retail",
    71: "Others",
    81: "Residential",
    82: "Residential",
    83: "Residential",
    84: "Residential",
    99: "Others",
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
    'residential': [81, 82, 83],
    'industrial': [31, 32],
    'retail': [21, 22, 25, 26, 62],
    'office': [23, 24],
    'medical': [51, 52, 53],
    'mixedresidential': [21, 22, 81, 83],
    'mixedoffice': [21, 22, 81, 83],
})

seed = 271828
print("using seed", seed)
random.seed(seed)
pd.np.random.seed(seed)


def verify():
    hdf_store = pd.HDFStore(os.path.join(misc.data_dir(), "run4032_school_v2_baseyear_py2.h5"), mode="r")

    new = verify_data_structure.yaml_from_store(hdf_store)
    with open(r"configs/data_structure.yaml", "w") as out:
        out.write(new)

    return hdf_store


orca.add_injectable("store", verify())
