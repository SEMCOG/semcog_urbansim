#import urbansim.sim.simulation as sim
import os
import pandas as pd
from urbansim.utils import misc
import orca

orca.add_injectable("transcad_available", False)

orca.add_injectable("emp_btypes", [1,3,4,5,6,7,8,10,21,22,23,24,25,26,27,28,29,31,32,33,35,38,39])

# maps building type ids to general building types; reduces dimensionality
orca.add_injectable("building_type_map", {
    17: "Residential",
    18: "Residential",
    19: "Residential",
    27: "Office",
    25: "Hotel",
    3: "School",
    32: "Industrial",
    33: "Industrial",
    39: "Industrial",
    28: "Retail",
    23: "Retail",
    24: "Retail",
    22: "Office"
})

# this maps building "forms" from the developer model
# to building types so that when the developer builds a
# "form" this can be converted for storing as a type
# in the building table - in the long run, the developer
# forms and the building types should be the same and the
# developer model should account for the differences
orca.add_injectable("form_to_btype", {
    'residential': [17, 18, 19],
    'industrial': [32,33,39],
    'retail': [28,23,24],
    'office': [22],
    'mixedresidential': [21],
    'mixedoffice': [21],
})

orca.add_injectable("store", pd.HDFStore(os.path.join(misc.data_dir(),
                                                     "semcog_data_fix.h5"), mode="r"))
