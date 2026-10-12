"""Write configs/households_by_city_2020.csv: households by city_id in 2020.

residential_developer starts each city's household trend from the change
between 2020 and the base year. The 2020 counts come from the RDF2050 base
HDF (base year 2020), whose households are all placed in buildings.

Run from the repo root:
    python scripts/make_households_by_city_2020.py [path/to/forecast_data_input_031523.h5]
"""
import os
import sys

import pandas as pd

DEFAULT_HDF = os.path.join(
    os.environ.get("SEMCOG_INPUT_DIR", "/mnt/D/RDF2055/forecast_inputs/base_year"),
    "forecast_data_input_031523.h5",
)
OUT = os.path.join("configs", "households_by_city_2020.csv")


def main(hdf_path):
    with pd.HDFStore(hdf_path, mode="r") as store:
        households = store["households"][["building_id"]]
        buildings = store["buildings"][["parcel_id"]]
        parcels = store["parcels"][["city_id"]]
    city = households.join(buildings, on="building_id").join(parcels, on="parcel_id")["city_id"]
    counts = city.dropna().astype(int).value_counts().sort_index()
    counts.rename_axis("city_id").rename("households").to_csv(OUT)
    print(f"{OUT}: {len(counts)} cities, {int(counts.sum()):,} households "
          f"({len(households) - int(counts.sum()):,} without a city)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HDF)
