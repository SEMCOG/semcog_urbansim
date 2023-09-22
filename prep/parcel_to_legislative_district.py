import os
import pandas as pd
import geopandas as gpd
import tobler
from sqlalchemy import create_engine, text

# set connection string, in terminal
# export POSTGIS_CONNECT_STR="postgresql+psycopg2://username:password@servername:5432/dbname"

# Load environment variables
DB_CONNECT_STR = os.environ.get('POSTGIS_CONNECT_STR')

if __name__ == "__main__":
    # load parcel
    sql = """SELECT the_geom geom, parcel_id FROM terra_parcels_2019;"""
    conn = create_engine(DB_CONNECT_STR).connect()
    parcels = gpd.read_postgis(text(sql), conn)

    # load state house
    mi_house = gpd.read_file('~/share/da/Staff/Xie/share/GIS/mi_house.shp')
    mi_house['mi_house_id'] = mi_house['district']
    mi_house = mi_house[['mi_house_id', 'geometry']]

    # load state senate
    mi_senate = gpd.read_file('~/share/da/Staff/Xie/share/GIS/mi_senate.shp')
    mi_senate['mi_senate_id'] = mi_senate['district']
    mi_senate = mi_senate[['mi_senate_id', 'geometry']]

    # spatial join
    print('joining michigan house')
    parcels = parcels.sjoin(mi_house, how='left', predicate='within')[['geom', 'parcel_id', 'mi_house_id']]
    # fix parcel intersect 2+ districts
    failed_parcels = parcels.loc[parcels['mi_house_id'].isna(), ['geom', 'parcel_id']]
    parcels.loc[failed_parcels.index, 'mi_house_id'] = tobler.area_weighted.area_join(mi_house, failed_parcels, ['mi_house_id'])['mi_house_id']

    # fix parcel not intersecting any district
    failed_parcels = parcels.loc[parcels['mi_house_id'].isna(), ['geom', 'parcel_id']]
    parcels.loc[failed_parcels.index, 'mi_house_id'] = failed_parcels.sjoin_nearest(mi_house, how='left')['mi_house_id']

    print('joining michigan senate')
    parcels = parcels.sjoin(mi_senate, how='left', predicate='within')[['geom', 'parcel_id', 'mi_house_id', 'mi_senate_id']]

    # fix parcel intersect 2+ districts
    failed_parcels = parcels.loc[parcels['mi_senate_id'].isna(), ['geom', 'parcel_id']]
    parcels.loc[failed_parcels.index, 'mi_senate_id'] = tobler.area_weighted.area_join(mi_senate, failed_parcels, ['mi_senate_id'])['mi_senate_id']

    # fix parcel not intersecting any district
    failed_parcels = parcels.loc[parcels['mi_senate_id'].isna(), ['geom', 'parcel_id']]
    parcels.loc[failed_parcels.index, 'mi_senate_id'] = failed_parcels.sjoin_nearest(mi_senate, how='left')['mi_senate_id']

    # quality controls
    assert parcels.isna().sum().sum() == 0
    assert parcels.parcel_id.duplicated().sum() == 0

    # save parcel_id to ids
    parcels[['parcel_id', 'mi_house_id', 'mi_senate_id']].to_csv('data/parcel_to_legis.csv', index=False)
    