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


#### Generate new SqFtProForma config YAML file
* Create `proforma_settings.py`
* Get default pro forma dictionary: `pf = SqFtProForma.from_defaults().to_dict`
* Modify dictionary with custom settings (reusing your settings)
  * Commented out first assignment of "costs" variable
  * Added entry in "construction_months" for medical use
* Instantiate new pro forma object from dictionary (using `**kwargs`) and
run `to_yaml()` to easily save as a YAML file.
* Test that pro forma can be loaded from this

