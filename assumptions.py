import os
import pandas as pd
from urbansim.utils import misc
import orca
import verify_data_structure

orca.add_injectable("transcad_available", False)

orca.add_injectable("emp_btypes", [1,3,4,5,6,7,8,10,21,22,23,24,25,26,27,28,29,31,32,33,35,38,39])

# maps building type ids to general building types; reduces dimensionality

# keys: binging type
# vause: network filter landues general_type
orca.add_injectable("building_type_map", {
    11: "institutional",
    12: "institutional",
    13: "institutional",
    14: "institutional",
    21: "commercial",
    22: "commercial",
    23: "office",
    24: "office",
    25: "commercial",
    26: "commercial",
    31: "industrial",
    32: "industrial",
    33: "TCU",
    41: "TCU",
    42: "TCU",
    43: "TCU",
    51: "medical",
    52: "medical",
    53: "medical",
    61: "institutional",
    62: "commercial",
    71: "others",
    81: "residential",
    82: "residential",
    83: "residential",
    84: "residential",
    99: "others",
})

# this maps building "forms" from the developer model
# to building types so that when the developer builds a
# "form" this can be converted for storing as a type
# in the building table - in the long run, the developer
# forms and the building types should be the same and the
# developer model should account for the differences

# keys: from proforma forms
# valus: biling typs aplyed to parcelses
orca.add_injectable("form_to_btype", {
    'residential': [81, 82, 83, 84],
    'industrial': [31, 32, 33],
    'retail': [21, 22],
    'office': [23, 24],
    'mixedresidential': [21, 22, 81, 83, 84],
    'mixedoffice': [21, 22, 81, 83, 84],
})


def verify():
    with open(r"configs/data_structure.yaml", "r") as out:
        structure = out.read()

    hdf_store = pd.HDFStore(os.path.join(misc.data_dir(), "semcog_data_fix.h5"), mode="r")

    new = verify_data_structure.yaml_from_store(hdf_store)

    if structure != new:
        with open("mismatched_data_description.yaml", "w") as out:
            out.write(new)
        assert structure == new

    return hdf_store

orca.add_injectable("store", verify())
