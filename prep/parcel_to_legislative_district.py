import os
import pandas as pd
import geopandas as gpd
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
    mi_house = gpd.read_file('~/share/da/Staff/Xie/share/GIS/MIHouseDistricts.shp')
    mi_house['mi_house_id'] = mi_house['DISTRICT']
    mi_house = mi_house[['mi_house_id', 'geometry']]

    # load state senate
    mi_senate = gpd.read_file('~/share/da/Staff/Xie/share/GIS/MISenateDistricts.shp')
    mi_senate['mi_senate_id'] = mi_senate['DISTRICT']
    mi_senate = mi_senate[['mi_senate_id', 'geometry']]

    # spatial join
    print('joining')
    parcels = parcels.sjoin(mi_house, how='left')[['geom', 'parcel_id', 'mi_house_id']]
    parcels = parcels.sjoin(mi_senate, how='left')[['geom', 'parcel_id', 'mi_house_id', 'mi_senate_id']]

    # save parcel_id to ids
    parcels[['parcel_id', 'mi_house_id', 'mi_senate_id']].to_csv('data/parcel_to_legis.csv', index=False)
    