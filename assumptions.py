import os
import random
import pandas as pd
import numpy as np
from urbansim.utils import misc
import orca
import verify_data_structure


@orca.injectable("year")
def year():
    default_year = 2020
    try:
        iter_var = orca.get_injectable("iter_var")
        if iter_var is not None:
            return iter_var
        else:
            return default_year
    except:
        return default_year


orca.add_injectable("transcad_available", False)

orca.add_injectable(
    "emp_btypes",
    [
        1,
        3,
        4,
        5,
        6,
        7,
        8,
        10,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        31,
        32,
        33,
        35,
        38,
        39,
    ],
)

# maps building type ids to general building types; reduces dimensionality

# keys: binging type
# vause: network filter landues general_type
orca.add_injectable(
    "building_type_map",
    {
        11: "Institutional",
        13: "Institutional",
        14: "Institutional",
        21: "Retail",
        23: "Office",
        31: "Industrial",
        32: "Industrial",
        33: "TCU",
        41: "TCU",
        42: "TCU",
        51: "Medical",
        52: "Medical",
        53: "Medical",
        61: "Entertainment",
        63: "Hospitality",
        65: "Hospitality",
        71: "Others",
        81: "Residential",
        82: "Residential",
        83: "Residential",
        84: "Residential",
        91: "Entertainment",
        92: "Institutional",
        93: "Institutional",
        94: "Retail",
        95: "TCU",
    },
)

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
orca.add_injectable(
    "form_to_btype",
    {
        "residential": [81, 82, 83],
        "industrial": [31, 32],
        "retail": [21, 22, 25, 26, 62],
        "office": [23, 24],
        "medical": [51, 52, 53],
        "mixedresidential": [21, 22, 81, 83],
        "mixedoffice": [21, 22, 81, 83],
    },
)

seed = 271828
print("using seed", seed)
random.seed(seed)
np.random.seed(seed)


def verify():
    # hdf_store = pd.HDFStore(os.path.join(misc.data_dir(), "run4032_school_v2_baseyear_py2.h5"), mode="r")
    # hdf_store = pd.HDFStore(
    #     "~/semcog_urbansim/data/all_semcog_data_02-02-18-final-forecast_newbid.h5",
    #     mode="r",
    # )
    data_path = r'/home/da/share/urbansim/RDF2050/model_inputs/base_hdf'
    hdf_list = [
        (data_path + "/" + f)
        for f in os.listdir(data_path)
        if ("forecast_data_input" in f) & (f[-3:] == ".h5")
    ]
    hdf_last = max(hdf_list, key=os.path.getctime)
    hdf_store = pd.HDFStore(hdf_last, "r")
    # hdf = pd.HDFStore(data_path + "/" +"forecast_data_input_091422.h5", "r")
    print("HDF data: ", hdf_last)

    new = verify_data_structure.yaml_from_store(hdf_store)
    with open("/home/da/semcog_urbansim/configs/data_structure.yaml", "w") as out:
        out.write(new)

    return hdf_store


orca.add_injectable("store", verify())
