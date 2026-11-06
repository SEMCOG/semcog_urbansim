"""parcel_id -> state house / state senate / US congress district.

Rebuilt for RDF2055 against gisad.terra_parcels_2023 and the district layers
that now live in the same database, replacing the earlier run over
terra_parcels_2019 and shapefiles on the share. The 2019 parcel set is what
left 25,674 of the 2023 parcels unassigned, and with them 85,114 jobs and
45,557 households, in the -1 district of every legislative indicator.

A parcel takes the district it has the most area in. For a parcel inside one
district that is that district, and for one straddling a boundary it is the
area-weighted winner, so the two passes of the earlier version collapse into
one. Districts are subdivided first: point-in-polygon against whole
multipolygons is what made the join slow.

Usage, from the repository root:
    POSTGIS_CONNECT_STR=postgresql://user:pwd@host:5432/land \
        python scripts/prep/parcel_to_legislative_district.py
Without the variable the connection string is read from SQL_CONFIG below.
"""
import os
import sys

import pandas as pd
import psycopg2

SQL_CONFIG = "/mnt/D/projects/forecast_data_input/config/sql.yaml"
# where output_indicators reads it from (input_paths.PARCEL_TO_LEGIS_CSV)
OUT = ("/mnt/hgfs/urbansim/RDF2055/model_inputs/base_tables/"
       "parcel_to_legis_rdf2055.csv")
PARCELS = "gisad.terra_parcels_2023"
LAYERS = {"mi_house_id": "gisad.terminus_mi_house",
          "mi_senate_id": "gisad.terminus_mi_senate",
          "us_congress_id": "gisad.terminus_us_congress"}
# vertices per subdivided piece; smaller is a faster index probe, more rows
SUBDIVIDE = 128


def connect_str():
    s = os.environ.get("POSTGIS_CONNECT_STR")
    if s:
        return s.replace("postgresql+psycopg2://", "postgresql://")
    import yaml
    with open(SQL_CONFIG) as f:
        return yaml.safe_load(f)["db_land_connection_str"]


def build(cur):
    keys = list(LAYERS)
    cur.execute("CREATE TEMP TABLE d AS " + " UNION ALL ".join(
        "SELECT %d lay, district::int did, ST_Subdivide(the_geom, %d) g FROM %s"
        % (i, SUBDIVIDE, tbl) for i, tbl in enumerate(LAYERS.values())))
    cur.execute("CREATE INDEX ON d USING gist(g)")
    cur.execute("ANALYZE d")
    cur.execute("SELECT lay, count(*), count(DISTINCT did) FROM d GROUP BY lay ORDER BY lay")
    for lay, pieces, n in cur.fetchall():
        print("  %-15s %3d districts in %5d pieces" % (keys[lay], n, pieces))

    # largest overlap per parcel and layer; pieces of one district are summed
    # first so a district split across several of them is not undercounted
    cur.execute("""
        CREATE TEMP TABLE hit AS
        SELECT DISTINCT ON (parcel_id, lay) parcel_id, lay, did
        FROM (SELECT p.parcel_id, d.lay, d.did,
                     sum(ST_Area(ST_Intersection(d.g, p.the_geom))) a
              FROM %s p JOIN d ON ST_Intersects(d.g, p.the_geom)
              GROUP BY 1, 2, 3) t
        ORDER BY parcel_id, lay, a DESC, did
    """ % PARCELS)
    cur.execute("SELECT lay, count(*) FROM hit GROUP BY lay ORDER BY lay")
    print("  assigned by overlap:", {keys[l]: n for l, n in cur.fetchall()})

    # a parcel whose geometry misses every district takes the nearest one
    cur.execute("""
        INSERT INTO hit
        SELECT p.parcel_id, l.lay, n.did
        FROM %s p
        CROSS JOIN (SELECT DISTINCT lay FROM d) l
        LEFT JOIN hit h ON h.parcel_id = p.parcel_id AND h.lay = l.lay
        CROSS JOIN LATERAL (SELECT did FROM d
                            WHERE d.lay = l.lay ORDER BY d.g <-> p.the_geom LIMIT 1) n
        WHERE h.parcel_id IS NULL
    """ % PARCELS)
    if cur.rowcount:
        print("  assigned by nearest district: %d" % cur.rowcount)

    cur.execute("SELECT parcel_id, lay, did FROM hit")
    df = pd.DataFrame(cur.fetchall(), columns=["parcel_id", "lay", "did"])
    df["lay"] = df.lay.map(dict(enumerate(keys)))
    return df.pivot(index="parcel_id", columns="lay", values="did")[keys]


def main():
    conn = psycopg2.connect(connect_str(), connect_timeout=30)
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM " + PARCELS)
    n_parcels = cur.fetchone()[0]
    print("%s: %d parcels" % (PARCELS, n_parcels))

    out = build(cur)
    conn.close()

    assert out.isna().sum().sum() == 0, out.isna().sum()
    assert not out.index.duplicated().any()
    assert len(out) == n_parcels, "%d of %d parcels assigned" % (len(out), n_parcels)

    out = out.astype(int).sort_index()
    out.to_csv(OUT)
    print("wrote %s: %d rows" % (OUT, len(out)))
    for c in out.columns:
        print("  %-15s %3d districts, %s..%s" % (c, out[c].nunique(), out[c].min(), out[c].max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
