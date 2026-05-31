# Street Networks

**Primary source:** OpenStreetMap (walk network), SEMCOG highway/TDM network (drive network)  
**File:** `semcog_2050_networks.h5` — a separate HDF5 from the main model input file (filename may need to be updated for the RDF2055 forecast)

The street network file provides the road and path geometry used by [Pandana](https://udst.github.io/pandana/) to compute dynamic neighborhood accessibility variables during the simulation. Each year, the `build_networks_2050` step loads these networks, snaps parcels and buildings to the nearest node, and aggregates nearby jobs, households, and amenities into building-level variables used by the location choice models.

This file is distinct from the [pre-computed accessibility indicators](accessibility.md) — those are static parcel-level outputs. The network file enables the simulation to recompute dynamic, year-varying accessibility each year as the distribution of jobs and households shifts.

---

## File Structure

The HDF5 contains one set of tables per named network. Each network has:

| Table type | Purpose |
|---|---|
| `nodes_*` | Network node locations (x, y coordinates + node ID) |
| `edges_*` | Connections between nodes with travel cost attributes |
| `local_nodes_*` | Subset of nodes for neighborhood-level queries (drive networks only) |
| `local_edges_*` | Corresponding subset of edges (drive networks only) |

---

## Networks in Production Use

The production simulation uses two networks:

### Walk Network — `osm_roads_walk_2020`

Based on **OpenStreetMap** road and path data for the 2020 base year.

| Table | Columns | Notes |
|---|---|---|
| `nodes_osm_roads_walk_2020` | `x`, `y`, `nodeid` | Node coordinates (state plane) |
| `edges_osm_roads_walk_2020` | `from`, `to`, `feet`, `minutes`, `meters` | Edge cost in feet, travel minutes, and meters |

**Cost used for queries:** `feet` (distance-based) or `minutes` (time-based)  
**Typical query radius:** 500 m (neighborhood density), 1,500 m (local population/jobs)

### Drive Networks — `highway_ext_2020` and `highway_ext_2030`

Based on the **SEMCOG highway / TDM road network**. Two vintages are provided — base year (2020) and a future year (2030) reflecting planned road improvements.

| Table | Columns | Notes |
|---|---|---|
| `nodes_highway_ext_2020` | `x`, `y`, `nodeid` | |
| `edges_highway_ext_2020` | `from`, `to`, `peak_mins`, `nonpk_mins`, `miles` | AM peak and off-peak travel times |
| `nodes_highway_ext_2020_local` | `x`, `y`, `nodeid` | Local subset for neighborhood queries |
| `edges_highway_ext_2020_local` | `from`, `to`, `peak_mins`, `nonpk_mins`, `miles` | |
| `nodes_highway_ext_2030` | same schema | Future network (planned improvements) |
| `edges_highway_ext_2030` | same schema | |
| `nodes_highway_ext_2030_local` | same schema | |
| `edges_highway_ext_2030_local` | same schema | |

**Cost used for queries:** `peak_mins` (commute accessibility) or `nonpk_mins` (off-peak)  
**Typical query radius:** 15–60 minutes (job access, retail proximity)

---

## What the Networks Compute

Each simulation year, Pandana uses these networks to produce **node-level** aggregations. These are then broadcast to buildings and parcels. Key variables computed:

**Walk-based (neighborhood density):**
- Residential and non-residential density within 500 m
- Population and households within 1,500 m
- Income mix, household demographics within 1,500 m
- Retail and service job counts within 500–2,000 m
- Average residential and non-residential prices nearby
- Transit stops within quarter mile

**Drive-based (regional access):**
- Jobs within 30, 45, and 60 minutes
- Population within 10 and 20 minutes
- Retail and shopping jobs within 15 and 30 minutes
- Nearby school achievement scores

---

## How Parcels Connect to the Network

Each parcel has a `nodeid_walk` and `nodeid_drv` column — the ID of the nearest walk or drive network node. These are pre-computed and stored in the parcels table. Buildings inherit them via the parcel. These columns are generated during model setup and do not need to be manually prepared.

If a parcel's node ID is missing or invalid, that building will have null network-based variables, which can affect location choice model predictions.

---

## Update Procedure

The network file should be regenerated when:
- Road network geometry changes significantly (new highways, removed roads)
- A new forecast cycle uses a different base year
- Walk network coverage needs updating (OSM data vintage)

**Walk network update:**
1. Download updated OSM data for SE Michigan
2. Process into Pandana-compatible nodes/edges (`feet`, `minutes`, `meters` cost columns)
3. Snap parcel centroids to nearest walk node → update `parcels.nodeid_walk`
4. Write new tables into `semcog_2050_networks.h5`

**Drive network update:**
1. Export node/edge tables from the SEMCOG TDM (TransCAD) network
2. Ensure cost columns are `peak_mins`, `nonpk_mins`, `miles`
3. Generate "local" subsets (nodes/edges within the 7-county region)
4. Snap parcel centroids to nearest drive node → update `parcels.nodeid_drv`
5. Write new tables into `semcog_2050_networks.h5`
6. Update the network configuration file if table names change

**After updating the network file:**
- Verify `nodeid_walk` and `nodeid_drv` coverage in `parcels` — count nulls and parcels with no node match
- Run a test simulation year and check that neighborhood variables are non-null for most buildings
- If node IDs change, the parcel columns must be recomputed — new node IDs won't match old parcel assignments

---

## Common Issues

- **Parcels not snapping to network** — parcel coordinates outside the network coverage area; check for projection mismatches (all data should be in Michigan state plane)
- **Missing local edges** — the local network subset must fully cover the 7-county region; edges that cross county boundaries may be clipped
- **Future network not reflecting planned projects** — confirm that the 2030 network includes committed highway improvements from the TDM
