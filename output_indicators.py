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
from indicators.model_outputs import *

warnings.filterwarnings("ignore")


def orca_year_dataset(hdf, tbls_to_load, year, is_base):
    """
    load orca tables with necessary geographies by specific year. Base year read from HDF base folder.
    hdf: hdf storage object
    tbls_to_load: list of tables
    year: int
    base: boolean, True means this year is base year
    """

    orca.clear_cache()
    orca.add_injectable("year", year)
    hdf_year = "base" if is_base or (year == 2019) else year

    for tbl in tbls_to_load:
        if (year == 2019) and (tbl == "jobs"):
            name = f"{hdf_year}/{tbl}_2019"
        else:
            name = f"{hdf_year}/{tbl}"
        if name in hdf:
            df = hdf[name]
        else:
            sub_name = [n for n in hdf.keys() if tbl in n][0]
            print(f"No table named {name}. Using the structure from {sub_name}.")
            df = hdf[sub_name].iloc[0:0]

        if tbl == 'parcels':
            df = add_legislative_ids(df)

        if tbl in {"households", "jobs"} and "large_area_id" not in df.columns:
            print("impute large_area_id")
            df["large_area_id"] = misc.reindex(
                orca.get_table("buildings").large_area_id, df.building_id
            )

        orca.add_table(tbl, df.fillna(0))

def add_legislative_ids(parcels):
    # add state house and senate ids
    leg_path = "data/parcel_to_legis.csv"
    if not os.path.exists(leg_path):
        print("Error: Missing parcel to legislative id mapping table")
        exit(1)
    p2l = pd.read_csv(leg_path, index_col='parcel_id')

    # fill pseudo parcels with -1
    parcels['mi_house_id'] = p2l['mi_house_id']
    parcels['mi_house_id'] = parcels['mi_house_id'].fillna(-1).astype(int)

    parcels['mi_senate_id'] = p2l['mi_senate_id']
    parcels['mi_senate_id'] = parcels['mi_senate_id'].fillna(-1).astype(int)

    parcels['us_congress_id'] = p2l['us_congress_id']
    parcels['us_congress_id'] = parcels['us_congress_id'].fillna(-1).astype(int)
    return parcels
    

def upload_whatnots_to_postgres(run_name, whatnots):
    table_name = "whatnots_" + run_name
    conn_str = "postgresql://gisad:forecast20@plannerprojection:5432/land"
    whatnots = whatnots.reset_index()
    whatnots["large_area_id"] = whatnots["large_area_id"].astype(int)
    whatnots["city_id"] = whatnots["city_id"].astype(int)
    whatnots["zone_id"] = whatnots["zone_id"].astype(int)
    whatnots["school_id"] = whatnots["school_id"].astype(int)
    whatnots["mi_house_id"] = whatnots["mi_house_id"].astype(int)
    whatnots["mi_senate_id"] = whatnots["mi_senate_id"].astype(int)
    whatnots["us_congress_id"] = whatnots["us_congress_id"].astype(int)
    whatnots["parcel_id"] = whatnots["parcel_id"].astype(int)
    print("Uploading whatnots table %s to postgres..." % table_name)
    whatnots.to_sql(table_name, conn_str, index=False, if_exists="replace")
    return


def upload_whatnots_to_carto(run_name, whatnots):
    ### code snippet below is from /home/da/share/da/Staff/Finkleman/carto_whatnot_import_snippet.py
    cred_path = "carto_cred.json"
    with open(cred_path, "r") as f:
        cred = json.load(f)
    api_key = cred["api_key"]
    # https://github.com/CartoDB/carto-python
    # pip install cartoframes
    run_number = run_name[3:]  # grab this from the the log file or somewhere else
    tablename = "whatnots_" + run_number

    # upload to carto
    set_default_credentials(cred_path)
    to_carto(whatnots.reset_index(), tablename, if_exists="replace")

    # make get requests to the datasets page so that the privacy can update correctly
    datasets_page = "https://semcogmaps.carto.com/u/semcog/api/v1/viz//?exclude_shared=false&per_page=12&shared=no&locked=false&only_liked=false&deepInsights=false&types=table&page=1&order=favorited,updated_at&order_direction=desc,desc&with_dependent_visualizations=10&load_do_totals=true&api_key={0}".format(
        api_key
    )
    get1 = requests.get(datasets_page)
    get2 = requests.get(datasets_page)
    get3 = requests.get(datasets_page)
    sleep(10)
    print(get3.text)
    update_privacy_table(tablename, "link")  # so that it can be view publicly

    # add three indexes using the sql api, will need to add more to this when we add in the school district data
    index_query = """CREATE INDEX IF NOT EXISTS w{0}_indicator_zone_id_idx ON semcog.{0} USING btree (indicator, zone_id); 
    CREATE INDEX IF NOT EXISTS w{0}_indicator_city_id_idx ON semcog.{0} USING btree (indicator, city_id); 
    CREATE INDEX IF NOT EXISTS w{0}_indicator_school_id_idx ON semcog.{0} USING btree (indicator, school_id);
    CREATE INDEX IF NOT EXISTS w{0}_indicator_mi_house_idx ON semcog.{0} USING btree (indicator, mi_house_id);
    CREATE INDEX IF NOT EXISTS w{0}_indicator_mi_senate_idx ON semcog.{0} USING btree (indicator, mi_senate_id);
    CREATE INDEX IF NOT EXISTS w{0}_indicator_us_congress_idx ON semcog.{0} USING btree (indicator, us_congress_id);
    CREATE INDEX IF NOT EXISTS w{0}_indicator_large_area_id_idx ON semcog.{0} USING btree (indicator, large_area_id);""".format(
        tablename
    )
    url = """http://semcog.cartodb.com/api/v2/sql?q={0}&api_key={1}""".format(
        index_query, api_key
    )
    create_index = requests.post(url)

    # added tablename to tables_of_whatnot
    add_tablename_sql = (
        """INSERT INTO tables_of_whatnot (table_name) SELECT '%s' WHERE NOT EXISTS (SELECT table_name FROM tables_of_whatnot WHERE table_name = '%s');"""
        % (tablename, tablename)
    )
    url = """http://semcog.cartodb.com/api/v2/sql?q={0}&api_key={1}""".format(
        add_tablename_sql, api_key
    )
    add_tablename = requests.post(url)
    return


def main(
    run_name, baseyear, finalyear, spacing=5, upload_to_carto=True, add_2019=False
):

    out_dir = run_name.replace(".h5", "")
    store_la = pd.HDFStore(run_name, mode="r")
    print(store_la)

    base_year = baseyear
    target_year = finalyear
    # spacing = 30 // (len(set(j[1: 5] for j in list(store_la.keys()) if j[1:5].isnumeric() and int(j[1:5]) > base_year)))
    save_detailed_tables = False  # save hh, person, building records or not

    if spacing == 1:
        all_years_dir = os.path.join(out_dir, "annual")
    else:
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
    # add target year inputs
    for tbl in ["parcels", "buildings"]:
        orca.add_table(tbl, store_la[f"{target_year}/" + tbl])

    import variables

    # add legis ids to parcels
    orca.add_table("parcels", add_legislative_ids(orca.get_table("parcels").local))

    # make cities with large area table
    p = orca.get_table("parcels")

    p = p.to_frame(["large_area_id", "us_congress_id", "mi_senate_id", "mi_house_id", "city_id", "school_id", "zone_id"])

    cities = (
        p[["city_id", "large_area_id"]].drop_duplicates("city_id").set_index("city_id")
    )
    schools = (
        p[["school_id", "large_area_id"]].drop_duplicates("school_id").set_index("school_id")
    )
    mi_house = (
        p[["mi_house_id"]].drop_duplicates("mi_house_id").set_index("mi_house_id")
    )
    mi_senate = (
        p[["mi_senate_id"]].drop_duplicates("mi_senate_id").set_index("mi_senate_id")
    )
    us_congress = (
        p[["us_congress_id"]].drop_duplicates("us_congress_id").set_index("us_congress_id")
    )
    orca.add_table("cities", cities)
    orca.add_table("schools", schools)
    orca.add_table("mi_house", mi_house)
    orca.add_table("mi_senate", mi_senate)
    orca.add_table("us_congress", us_congress)

    # initialize whatnot
    whatnot = p.reset_index().drop_duplicates(
        ["large_area_id", "us_congress_id", "mi_senate_id", "mi_house_id", "city_id", "school_id", "zone_id", "parcel_id"]
    )

    # target year building
    b = store_la[f"/{target_year}/buildings"]

    # for 1-year analysis purpose, interesting_parcel_ids: parcels with new built or demolition and size > 2 acres
    if spacing == 1:
        interesting_parcel_ids = set(b[b.year_built > base_year].parcel_id) | set(
            store_la[f"/{target_year}/dropped_buildings"].parcel_id
        )
        acres = orca.get_table("parcels").acres
        interesting_parcel_ids = interesting_parcel_ids & set(acres[acres > 2].index)
    else:
        interesting_parcel_ids = set()
    orca.add_injectable("interesting_parcel_ids", interesting_parcel_ids)

    # add missing zones from buildings
    orca_b = orca.get_table('buildings')
    b_whatnot = orca_b.to_frame(['large_area_id', "us_congress_id", "mi_senate_id", "mi_house_id", 'city_id', 'school_id', 'zone_id', 'parcel_id'])
    b_whatnot = b_whatnot.drop_duplicates().reset_index(drop=True)
    whatnot = pd.concat([whatnot, b_whatnot], axis=0, ignore_index=True)

    whatnot.loc[~whatnot.parcel_id.isin(interesting_parcel_ids), "parcel_id"] = 0

    # clean up whatnot index,
    whatnot = whatnot.drop_duplicates(
        ["large_area_id", "us_congress_id", "mi_senate_id", "mi_house_id", "city_id", "school_id", "zone_id", "parcel_id"]
    ).reset_index(drop=True)

    whatnot.index.name = "whatnot_id"
    orca.add_table("whatnots", whatnot)

    # load indicator to orca
    for tab, geo_id in [
        ("cities", "city_id"),
        ("semmcds", "semmcd"),
        ("schools", "school_id"),
        ("mi_house", "mi_house_id"),
        ("mi_senate", "mi_senate_id"),
        ("us_congress", "us_congress_id"),
        ("zones", "zone_id"),
        ("large_areas", "large_area_id"),
        ("whatnots", "whatnot_id"),
    ]:
        make_indicators(tab, geo_id)

        # geo level: school district
    years = list(range(base_year, target_year + 1, spacing))
    if add_2019:
        years = [2019] + years
    year_names = ["yr" + str(i) for i in years]
    geom = ["cities", "us_congress", "mi_senate", "mi_house", "schools", "semmcds", "zones", "large_areas", "whatnots"]
    tbls_to_load = [
        "parcels",
        "buildings",
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
    for year in years:
        print("processing ", year)
        orca_year_dataset(store_la, tbls_to_load, year, year == base_year)
        for tab in geom:
            dict_ind[tab].append(orca.get_table(tab).to_frame(list_indicators()))
    end = time.time()
    print("runtime:", end - start)

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
        df.set_index(["large_area_id", "us_congress_id", "mi_senate_id", "mi_house_id", "city_id", "school_id", "zone_id", "parcel_id"], inplace=True)
        df = df.fillna(0)
        df = df.sort_index().sort_index(axis=1)

        df.columns.name = "indicator"
        df = df.stack().to_frame()
        df["year"] = y
        df.set_index("year", append=True, inplace=True)
        whatnots_output.append(df)

    whatnots_output = pd.concat(whatnots_output).unstack(fill_value=0)
    whatnots_output.index.rename("us_congress_id", level=1, inplace=True)
    whatnots_output.index.rename("mi_senate_id", level=2, inplace=True)
    whatnots_output.index.rename("mi_house_id", level=3, inplace=True)
    whatnots_output.index.rename("city_id", level=4, inplace=True)
    whatnots_output.index.rename("school_id", level=5, inplace=True)
    whatnots_output.index.rename("zone_id", level=6, inplace=True)
    whatnots_output.columns = year_names

    if add_2019:
        # whatnots for internal
        whatnots_output_internal = whatnots_output.drop("yr2020", axis=1)
        whatnots_output_internal.rename(columns={"yr2019": "yr2020"}, inplace=True)
        if spacing == 1:
            whatnots_output_internal[year_names[1::5]].to_csv(
                os.path.join(out_dir, "whatnots_output_internal.csv")
            )
        whatnots_output_internal.to_csv(
            os.path.join(all_years_dir, "whatnots_output_internal.csv")
        )
        # whatnots for external
        not_jobs = [x for x in whatnots_output.index if "jobs" not in x[-1]]
        whatnots_output.loc[not_jobs, "yr2019"] = np.nan
        if spacing == 1:
            whatnots_output[[year_names[0]] + year_names[1::5]].to_csv(
                os.path.join(out_dir, "whatnots_output_external.csv")
            )
        whatnots_output.to_csv(
            os.path.join(all_years_dir, "whatnots_output_external.csv")
        )
        whatnots_output = whatnots_output_internal
    else:
        if spacing == 1:
            whatnots_output[year_names[::5]].to_csv(
                os.path.join(out_dir, "whatnots_output.csv")
            )
        whatnots_output.to_csv(os.path.join(all_years_dir, "whatnots_output.csv"))

    # upload_whatnots_to_postgres(os.path.basename(out_dir), whatnots_output)
    if upload_to_carto is True:
        upload_whatnots_to_carto(os.path.basename(out_dir), whatnots_output)
    end = time.time()
    print("runtime whatnots:", end - start)

    ### save indicators to excel files
    print("\n* Making indicators by year")
    start = time.time()
    geom = ["cities", "large_areas", "us_congress", "mi_senate", "mi_house", "semmcds", "schools", "zones"]
    not_jobs = [x for x in list_indicators() if "jobs" not in x]
    if add_2019:
        y5 = year_names[1::5]
        for tab in dict_ind:
            dict_ind[tab][0][not_jobs] = np.nan
    else:
        y5 = year_names[0::5]

    for tab in geom:
        print(tab)
        # indicator for year, also save 5-year indicator files if spacing ==1
        xls_name = tab + "_by_indicator_for_year.xlsx"

        if spacing == 1:
            writer5 = pd.ExcelWriter(os.path.join(out_dir, xls_name))

        writer = pd.ExcelWriter(os.path.join(all_years_dir, xls_name))
        for i, y in enumerate(year_names):
            df = dict_ind[tab][i]
            df = df.dropna(axis=1, how="all")
            df = df.fillna(0)
            df = df.sort_index().sort_index(axis=1)

            df.to_excel(writer, y)
            if (spacing == 1) & (y in y5):  # 5-year indicator files
                df.to_excel(writer5, y)
        writer.save()
        if spacing == 1:
            writer5.save()

        # year for indicator
        print("\n* Making years by indicator")
        xls_name = tab + "_by_year_for_indicator.xlsx"
        if spacing == 1:
            writer5 = pd.ExcelWriter(os.path.join(out_dir, xls_name))

        writer = pd.ExcelWriter(os.path.join(all_years_dir, xls_name))
        if tab == "cities" or tab == "semmcds":
            la_id = orca.get_table(tab).large_area_id
            # name = orca.get_table(tab).city_name
        if tab == "large_areas":
            name = orca.get_table(tab).large_area_name
        for ind in list_indicators():
            df = pd.concat([df[ind] for df in dict_ind[tab]], axis=1)
            df.columns = year_names
            if tab == "cities" or tab == "semmcds":
                df["large_area_id"] = la_id
                df.set_index("large_area_id", append=True, inplace=True)
            if tab == "large_areas":
                df["large_area_name"] = name
                df.set_index("large_area_name", append=True, inplace=True)
            if len(df.columns) > 0:
                print("saving:", ind)
                if add_2019:
                    df = df.dropna(axis=1, how="all")
                #     df = df.drop('yr2019', axis=1)
                df = df.fillna(0)
                df = df.sort_index().sort_index(axis=1)
                df.to_excel(writer, ind)
                if spacing == 1:
                    df[y5].to_excel(writer5, ind)
            else:
                print("something is wrong with:", ind)
        writer.save()
        if spacing == 1:
            writer5.save()

    end = time.time()
    print("runtime geom:", end - start)

    # save disaggregated tables: buildings, households, persons
    if save_detailed_tables == True:
        print("\nSaving detailed tables.....")
        start = time.time()
        for year in range(base_year, target_year + 1, 5):
            print("buildings for", year)
            orca_year_dataset(store_la, tbls_to_load, year, year == base_year)
            buildings = orca.get_table("buildings")
            df = buildings.to_frame(
                buildings.local_columns + ["city_id", "large_area_id", "x", "y"]
            )
            # df = df[df.building_type_id != 99]
            df = df.fillna(0)
            df = df.sort_index().sort_index(axis=1)
            df.to_csv(os.path.join(out_dir, "buildings_yr" + str(year) + ".csv"))

            persons = orca.get_table("persons")
            df = persons.to_frame(persons.local_columns + ["city_id", "large_area_id"])
            df = df.fillna(0)
            df = df.sort_index().sort_index(axis=1)
            df.to_csv(os.path.join(out_dir, "hh_persons_yr" + str(year) + ".csv"))

            households = orca.get_table("households")
            df = households.to_frame(
                households.local_columns + ["city_id", "large_area_id"]
            )
            df = df.fillna(0)
            df = df.sort_index().sort_index(axis=1)
            df.to_csv(os.path.join(out_dir, "households_yr" + str(year) + ".csv"))
        end = time.time()
        print("runtime:", end - start)

    # construction and demolition
    print("\nSaving building differences (construction and demolition).....")
    start = time.time()
    if add_2019:
        years = years[1:]
    year_names = ["yr" + str(i) for i in years]
    writer = pd.ExcelWriter(os.path.join(out_dir, "buildings_dif_by_year.xlsx"))
    for year, year_name in zip(years, year_names):
        print("buildings for", year)
        orca_year_dataset(store_la, tbls_to_load, year, year == base_year)
        buildings = orca.get_table("buildings")
        df = buildings.to_frame(buildings.local_columns + ["city_id", "large_area_id"])
        df = df[df.year_built == year]
        df = df.fillna(0)
        df = df.sort_index().sort_index(axis=1)
        df.to_excel(writer, "const_" + year_name)

        demos = orca.get_table("dropped_buildings")
        df = demos.to_frame(demos.local_columns + ["city_id", "large_area_id"])
        df = df[df.year_demo == year]
        df = df.fillna(0)
        df = df.sort_index().sort_index(axis=1)
        df.to_excel(writer, "demo_" + year_name)

    writer.save()
    end = time.time()
    print("runtime:", end - start)

    store_la.close()


if __name__ == "__main__":
    ## test script
    main(
        "./runs/run2008.h5",
        2020,
        2021,
        add_2019=True,
        spacing=5,
        upload_to_carto=False,
    )

