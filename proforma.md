# Pro Forma update

Documenting update steps here.

#### Run baseline (before changes)
Needed to make a couple of updates to `proforma_test.ipynb` to get things
running:
* Create script for easier debugging
* Point `assumptions.py` to the new data file (all_semcog_data_05-22-17.h5)
* Add parcels DataFrame as an argument for `parcel_price_callback`
in `run_feasibility` in utils.py
* The proforma.py file now runs through cleanly


