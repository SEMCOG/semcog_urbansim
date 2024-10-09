import pandas as pd
import numpy as np
import os
import time
import warnings
import requests
import json
import orca
from collections import defaultdict
from urbansim.utils import misc
from time import sleep
from cartoframes import to_carto
from cartoframes import update_privacy_table
from cartoframes.auth import set_default_credentials
from indicators.model_output_shorten import *

warnings.filterwarnings("ignore")

def main():
    out_dir = "/home/da/semcog_urbansim/runs/whatnots_2020"

    store_la = pd.HDFStore("/home/da/semcog_urbansim/runs/run1288.h5", mode="r")
    orca.add_injectable('store', pd.HDFStore('/home/da/share/urbansim/RDF2050/model_inputs/base_hdf/forecast_data_input_031523.h5', 'r'))
    print(store_la)

    all_years_dir = out_dir
    if not (os.path.exists(all_years_dir)):
        os.makedirs(all_years_dir)

    # load essential tables for orca
    for tbl in [
        "semmcds",
        "zones",
        "large_areas",
        "building_sqft_per_job",
        "zoning",
        "events_addition",
        "events_deletion",
        "refiner_events",
        "employment_sectors",
        "building_types",
    ]:
        orca.add_table(tbl, store_la["base/" + tbl])
    store_la.close()
    
    ### 2020
    # add target year inputs
    # for tbl in ["parcels", "buildings"]:
    #     orca.add_table(tbl, store_la[f"{target_year}/" + tbl])
    from indicators.model_output_shorten import parcels, buildings, households, persons
    hdf20 = pd.HDFStore("/home/da/share/urbansim/RDF2050/model_inputs/base_hdf/forecast_data_input_031523.h5", 'r')
    orca.add_table('buildings', buildings(hdf20))
    orca.add_table('parcels', parcels(hdf20, orca.get_table('zoning')))
    hh = pd.read_csv("/mnt/hgfs/urbansim/RDF2050/population_synthesis/historical/2020(2022)/100124_run_2020/households_after_refinement.csv", index_col=0)
    p = pd.read_csv("/mnt/hgfs/urbansim/RDF2050/population_synthesis/historical/2020(2022)/100124_run_2020/persons_after_refinement.csv", index_col=0)
    import variables
    orca.add_table('households', households(hh, orca.get_table('buildings')))
    orca.add_table('persons', persons(p))
    hdf20.close()

    # make cities with large area table
    p = orca.get_table("parcels")

    p = p.to_frame(["large_area_id", "city_id", "zone_id"])

    cities = (
        p[["city_id", "large_area_id"]].drop_duplicates("city_id").set_index("city_id")
    )
    orca.add_table("cities", cities)

    # initialize whatnot
    whatnot = p.reset_index().drop_duplicates(
        ["large_area_id", "city_id", "zone_id"]
    )

    # add missing zones from buildings
    orca_b = orca.get_table('buildings')
    b_whatnot = orca_b.to_frame(['large_area_id', 'city_id', 'zone_id'])
    b_whatnot = b_whatnot.drop_duplicates().reset_index(drop=True)
    whatnot = pd.concat([whatnot, b_whatnot], axis=0, ignore_index=True)

    # clean up whatnot index,
    whatnot = whatnot.drop_duplicates(
        ["large_area_id", "city_id", "zone_id"]
    ).reset_index(drop=True)

    whatnot.index.name = "whatnot_id"
    orca.add_table("whatnots", whatnot)

    # load indicator to orca
    for tab, geo_id in [
        ("cities", "city_id"),
        ("semmcds", "semmcd"),
        ("zones", "zone_id"),
        ("large_areas", "large_area_id"),
        ("whatnots", "whatnot_id"),
    ]:
        make_indicators(tab, geo_id)

    geom = ["cities", "semmcds", "zones", "large_areas", "whatnots"]
    tbls_to_load = [
        "jobs",
        "households",
        "persons",
        "group_quarters",
        "base_job_space",
        "dropped_buildings",
    ]

    start = time.time()
    # produce indicators by year
    print("producing all indicators by year ...")
    dict_ind = defaultdict(list)

    year = 2020
    print("processing ", year)
    for tab in geom:
        dict_ind[tab].append(orca.get_table(tab).to_frame(list_indicators()))

    end = time.time()
    print("runtime:", end - start)

    year_names = ['yr2020']
    ## region should have same value no matter how you slice it.
    df = pd.DataFrame()
    for tab in list(dict_ind):
        for year, year_data in zip(year_names, dict_ind[tab]):
            df[(year, tab)] = year_data.sum()
    df = df.T
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.sort_index().sort_index(axis=1)

    whatnots_exclude_vars = list(df.columns[df.columns.str.startswith("pct_")]) + [
        "res_vacancy_rate",
        "nonres_vacancy_rate",
        "household_size",
        "hh_pop_age_median",
    ]
    df.drop(whatnots_exclude_vars, axis=1, inplace=True)

    sumstd = df.groupby(level=0).std().sum().sort_values()
    print(sumstd[sumstd > 0])
    print(df[sumstd[sumstd > 0].index])
    # Todo: add part of fenton to semmcd table
    print(
        set(orca.get_table("semmcds").to_frame(list_indicators()).hh_pop.index)
        ^ set(orca.get_table("semmcds").hh_pop.index)
    )

    # process and save whatnots_output table by year interval
    start = time.time()
    whatnots_local = orca.get_table("whatnots").local.fillna(0)
    whatnots_output = []
    for i, y in enumerate(year_names):
        df = dict_ind["whatnots"][i].copy()
        df.index.name = "whatnot_id"
        del df["res_vacancy_rate"]
        del df["nonres_vacancy_rate"]
        del df["household_size"]
        del df["hh_pop_age_median"]

        df[whatnots_local.columns] = whatnots_local
        df.set_index(["large_area_id", "city_id", "zone_id"], inplace=True)
        df = df.fillna(0)
        df = df.sort_index().sort_index(axis=1)

        df.columns.name = "indicator"
        df = df.stack().to_frame()
        df["year"] = y
        df.set_index("year", append=True, inplace=True)
        whatnots_output.append(df)

    whatnots_output = pd.concat(whatnots_output).unstack(fill_value=0)
    whatnots_output.index.rename("city_id", level=1, inplace=True)
    whatnots_output.index.rename("zone_id", level=2, inplace=True)
    whatnots_output.columns = year_names

    whatnots_output.to_csv(os.path.join(all_years_dir, "whatnots_output.csv"))

    # upload_whatnots_to_postgres(os.path.basename(out_dir), whatnots_output)
    if False:
        upload_whatnots_to_carto(os.path.basename(out_dir), whatnots_output)
    end = time.time()
    print("runtime whatnots:", end - start)

 
if __name__ == "__main__":
    main()