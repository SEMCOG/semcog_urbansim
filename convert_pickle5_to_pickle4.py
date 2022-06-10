# need to switch to python 3.8 before running the script
import pandas as pd
import pickle

new_hdf = pd.HDFStore("data/all_semcog_data_02-02-18-final-forecast_newbid_pickle4.h5")
old_hdf = pd.HDFStore( "data/all_semcog_data_02-02-18-final-forecast_newbid.h5", mode="r")

for k in old_hdf.keys():
	df = old_hdf[k]
	pickle.HIGHEST_PROTOCOL = 4
	df.to_hdf(new_hdf, k, mode='a')
	pickle.HIGHEST_PROTOCOL = 5
	print(k)

new_hdf.close()
old_hdf.close()