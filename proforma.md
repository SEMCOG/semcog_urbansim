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


#### Create and run new_feasibility step
* Copy "feasibility" model from urbansim_parcels
* Point to helper function `urbansim_parcels.utils`
* Copy `parcel_average_price` callback function from variables module and
  modify slightly (remove the df parameter)
* New model uses `proforma.yaml` created above


#### Create and run new_res_developer step
* Copy "res_developer.yaml" from urbansim_parcels repo and manually edit
  the values
* copy "res_developer" model from urbansim_parcels, update parameters
* Add new random_type function to fit the new format
* Did not hardcode target_units number like in utils.py; can set using
num_units_to_build parameter
