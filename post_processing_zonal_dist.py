import pandas as pd
from zonal_redist import get_baseyear_dist, target_year_zonal_distribution, target_year_data, make_hh_la, get_czone_weights, assign_hh_to_hu, get_hh_refine_difference, match_hh_targets

def main():
	hdf= pd.HDFStore('/mnt/hgfs/RDF2050/run2087_zonal.h5', 'r')
	new_hdf = pd.HDFStore('~/semcog_urbansim/data/run2087_with_age.h5', 'w')

	# get base year distribution
	hh = hdf['/base/households']
	b = hdf['/base/buildings']
	p = hdf['/base/parcels']
	b = b.join(p[['zone_id']], on='parcel_id')
	hbase_zone = get_baseyear_dist(hh, b)

	for year in range(2025, 2051, 5):
		# Run year zonal distribution
		households = hdf[ '/%s/households'%year]
		buildings = hdf[ '/%s/buildings'%year]
		parcels = hdf[ '/%s/parcels'%year]
		buildings = buildings.join(parcels[['zone_id']], on='parcel_id')
		hyear_new = target_year_zonal_distribution(households, buildings, parcels, hbase_zone)
		# update households table
		new_hdf["/%s/households" % year] = hyear_new
		print("Finished zonal_distribution and ", "/%s/households" % year, " was added")
	# clean up
	hdf.close()
	new_hdf.close()

if __name__ == "__main__":
	main()
