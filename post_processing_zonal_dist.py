import pandas as pd
from zonal_redist import target_year_data, make_hh_la, get_czone_weights, assign_hh_to_hu, get_hh_refine_difference, match_hh_targets

def main():
	hdf = pd.HDFStore('', 'r')

	# get base year distribution
	hh = hdf['/base/households']
	b = hdf['/base/buildings']
	p = hdf['/base/parcels']
	b = b.join(p[['zone_id']], on='parcel_id')

	hbase = make_hh_la(hh, b)
	# concat all groupby columsn as string to reduce runtime
	hbase['concat'] = hbase.inc_qt.astype(int).map(str) + '-' + hbase.hhsize.map(str)
	hbase['concat'] = hbase['concat'] + '-' + hbase.large_area_id.map(str)
	hbase['concat'] = hbase['concat'] + '-' + hbase.city_id.map(str)
	hbase['concat'] = hbase['concat'] + '-' + hbase.city_zone.map(str)

	hbase_zone = hbase.groupby('concat').size()
	hbase_zone.name = 'weights'
	hbase_zone = hbase_zone * 1.00
	hbase_zone += 0.001

	# restore all columns from concat string
	hbase_zone = hbase_zone.reset_index()
	hbase_zone['inc_qt'] = hbase_zone['concat'].str.split('-').str[0].astype(int)
	hbase_zone['hhsize'] = hbase_zone['concat'].str.split('-').str[1].astype(int)
	hbase_zone['large_area_id'] = hbase_zone['concat'].str.split('-').str[2].astype(int)
	hbase_zone['city_id'] = hbase_zone['concat'].str.split('-').str[3].astype(int)
	hbase_zone['city_zone'] = hbase_zone['concat'].str.split('-').str[4].astype(int)
	hbase_zone = hbase_zone.drop(columns='concat')

	hbase_zone = hbase_zone.set_index(['large_area_id', 'inc_qt', 'hhsize'])
	# orca.add_table('baseyear_households_by_zone', hbase_zone)

	year = 2050
	# Run year 2050 zonal distribution
	households = hdf['/%s/households'%year]
	buildings = hdf['/%s/buildings'%year]
	parcels = hdf['/%s/parcels'%year]
	buildings = b.join(p[['zone_id']], on='parcel_id')

	#prepare target year hhs, hhs grpby inc and size, b2(building id repeat by res units)
	hyear, hyear_g, b2 = target_year_data(households, buildings, parcels, 2050)

	#adjust weights and get city-zone sample distribution
	czone, czoneg, weightsg, hbase_zone = get_czone_weights(
		hbase_zone, hyear, hyear_g)

	czone = czone.reset_index()
	czone['building_id'] = -1
	#iter city-zone, assign HH to HUs by min(HH, HU), keep remaining HH and HU, do full random assginment at the end
	czone, b2 = assign_hh_to_hu(czone, b2)

	czone = czone.set_index(['large_area_id', 'inc_qt', 'hhsize'])
	hyear_new = hyear.copy()
	# give new bid and city_zone to households
	for ind, v in hyear_new.groupby(['large_area_id', 'inc_qt', 'hhsize']):
		hyear_new.loc[v.index, 'building_id'] = czone.loc[ind].building_id.values
		hyear_new.loc[v.index, 'city_zone'] = czone.loc[ind].city_zone.values

	hyear_new['new_city_id'] = hyear_new.city_zone // 10000

	#get difference table between new and target
	hyear_newg = get_hh_refine_difference(hyear_new, hyear)
	print('dif table summary', hyear_newg.sum())
	hyear_newg = hyear_newg.reset_index()

	hyear_new, hyear_newg = match_hh_targets(hyear_new, hyear_newg, b2)

	hyear_new['city_id'] = hyear_new['new_city_id']
	hyear_new['zone_id'] = hyear_new['city_zone'] % 10000
	hyear_new.drop(['city_zone', 'hhsize', 'inc_qt', 'new_city_id',
					'city_id', 'zone_id'], axis=1, inplace=True)
	# update households table
	hdf['/2050/households_after_zd'] = hyear_new
	print("Finished zonal_distribution and /2050/households_after_zd was added")
	# clean up
	hdf.close()

if __name__ == "__main__":
	main()