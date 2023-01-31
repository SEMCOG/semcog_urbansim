import numpy as np
from developer.develop import Developer
from developer.sqftproforma import SqFtProForma

# Get default dictionary
pf_dict = SqFtProForma.from_defaults().to_dict

# Modify dictionary

pf_dict["profit_factor"] = 1.0
# Adjust cost downwards based on RS Means test factor

# This is overwritten below
# pf_dict['costs'] = {btype: list(np.array(pf_dict['costs'][btype]) * 0.9) for btype in
#                pf_dict['costs']}

# Adjust price downwards based on RS Means test factor
pf_dict["parking_cost_d"] = {
    ptype: pf_dict["parking_cost_d"][ptype] * 0.9 for ptype in pf_dict["parking_cost_d"]
}

pf_dict["uses"] = [
    "retail",
    "industrial",
    "office",
    "medical",
    "entertainment",
    "residential",
]
pf_dict["residential_uses"] = [False, False, False, False, False, True]
pf_dict["forms"] = {
    "retail": {"retail": 1.0},
    "industrial": {"industrial": 1.0},
    "office": {"office": 1.0},
    "residential": {"residential": 1.0},
    "medical": {"medical": 1.0},
    "entertainment": {"entertainment": 1.0},
    # "mixedresidential": {"retail": 0.1, "residential": 0.9},
    # "mixedoffice": {"office": 0.7, "residential": 0.3},
}
pf_dict["parking_rates"] = {
    "retail": 2.0,
    "industrial": 0.6,
    "office": 1.0,
    "medical": 1.0,
    "entertainment": 2.0,
    "residential": 1.0,
}
pf_dict["building_efficiency"] = 0.85
pf_dict["parcel_coverage"] = 0.85

pf_dict["costs"] = {
    "residential": [106.0, 96.0, 160.0, 180.0, 180.0, 205.0, 205.0, 205.0, 999.0],
    "industrial": [125.0, 125.0, 130.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0],
    "office": [165.0, 180.0, 180.0, 180.0, 175.0, 175.0, 175.0, 999.0, 999.0],
    "medical": [210.0, 240.0, 300.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0],
    "retail": [120.0, 145.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0],
    "entertainment": [120.0, 145.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0],
}

pf_dict["heights_for_costs"] = [12, 24, 36, 48, 72, 108, 216, 216, np.inf]

# Add construction time for new use

pf_dict["construction_months"]["medical"] = pf_dict["construction_months"]["office"]


# Instantiate new pro forma object
pf = SqFtProForma(**pf_dict)
pf.to_yaml("configs/proforma.yaml")

# Test that pro forma can be loaded
npf = SqFtProForma.from_yaml(str_or_buffer="configs/proforma.yaml")
# print((npf.get_debug_info("mixedoffice", "deck")))
