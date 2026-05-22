# Geography

**Primary source:** PostgreSQL boundary tables; some tables are constants loaded from a prior HDF  
**SQL config:** `config/sql.yaml`

This domain covers geographic reference tables that define the spatial hierarchy used throughout the model. These tables are used for aggregation, joining, and assigning spatial identifiers to parcels, buildings, households, and jobs.

---

## Tables in This Domain

| Table | Description | Source |
|---|---|---|
| `zones` | Traffic Analysis Zones (TAZ) | SQL |
| `semmcds` | Municipal entities (cities, townships) | SQL |
| `counties` | 7 SE Michigan counties | Constant (loaded from prior HDF) |
| `large_areas` | 8 modeling zones | Constant (loaded from prior HDF) |

---

## `zones`

Traffic Analysis Zones used for travel demand modeling. Parcels reference `zone_id`, and travel data is indexed by zone pairs.

**Index:** `zone_id` (unique)  
**SQL table:** TAZ boundary table — query configured in `config/sql.yaml`

| Column | Type | Rules | Notes |
|---|---|---|---|
| `zone_id` | int16 | range [1 – 2811], no null | Model TAZ identifier |
| `taz` | object | no null | Full TAZ GEOID string |
| `tazce10_n` | int32 | no null | Census TAZ code (used for joining with travel model) |
| `acres` | float64 | no null | Zone area in acres |

**Note:** TransCAD uses `tazce10_n` codes. When converting travel matrices from OMX, always map through this table to get `zone_id`.

---

## `semmcds`

Municipal entity reference table — cities, townships, villages in the 7-county region.

**Index:** `semmcd_id` (unique)  
**SQL table:** MCD boundary table — query configured in `config/sql.yaml`

| Column | Type | Rules | Notes |
|---|---|---|---|
| `semmcd_id` | int16 | range [5 – 7100], no null | SEMCOG MCD code |
| `county_id` | int16 | — | Parent county |
| `large_area_id` | int16 | — | Parent large area |
| `area_name` | object | no null | Municipality name |

---

## `counties`

The 7 SE Michigan counties. This is a **constant table** loaded from a prior HDF — it rarely needs updating.

**Index:** `county_id` (unique)

**Valid county codes:**

| Code | County |
|---|---|
| 93 | Monroe |
| 99 | Macomb |
| 115 | Oakland |
| 125 | Livingston |
| 147 | St. Clair |
| 161 | Detroit (Wayne — City of Detroit) |
| 163 | Wayne (excluding Detroit) |

| Column | Type | Rules |
|---|---|---|
| `county_id` | int16 | 7 valid codes, no null |
| `county_name` | object | no null |

---

## `large_areas`

The 8 modeling zones used for model segmentation, control totals, and output aggregation. This is a **constant table** — it does not change between forecast cycles.

**Index:** `large_area_id` (unique)

**Valid large area codes:**

| Code | Area |
|---|---|
| 3 | Wayne County (excl. Detroit) |
| 5 | Washtenaw County |
| 93 | Monroe County |
| 99 | Macomb County |
| 115 | Oakland County |
| 125 | Livingston County |
| 147 | St. Clair County |
| 161 | Detroit |

| Column | Type | Rules |
|---|---|---|
| `large_area_id` | int16 | 8 valid codes, no null |
| `large_area_name` | object | no null |
| `large_area_group_id` | int8 | values: 1,2,3,4 |
| `alt_large_id` | int8 | values: 1–8 |

---

## Update Notes

**`zones` and `semmcds`** — update when TAZ or MCD boundaries change (typically when census boundaries are updated or new municipalities form/dissolve).

**`counties` and `large_areas`** — these are stable definitions tied to the 7-county region and 8 modeling zones. Only update if the model's geographic scope changes, which is a major structural decision.

When updating any geography table, downstream joins must be verified:
- `parcels.zone_id` → `zones`
- `parcels.semmcd` → `semmcds`
- `parcels.county_id` → `counties`
- `parcels.large_area_id` → `large_areas`
