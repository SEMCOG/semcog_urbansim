# Geography

**Primary source:** PostgreSQL boundary tables; some tables are constants loaded from a prior model input

This domain covers geographic reference tables that define the spatial hierarchy used throughout the model. These tables are used for aggregation, joining, and assigning spatial identifiers to parcels, buildings, households, and jobs.

---

## Tables in This Domain

| Table | Description | Source | In Model Input |
|---|---|---|---|
| `zones` | Traffic Analysis Zones (TAZ) | SQL | Yes |
| `semmcds` | Municipal entities (cities, townships) | SQL | Yes |
| `counties` | 7 SE Michigan counties | Constant (loaded from prior model input) | Yes |
| `large_areas` | 8 modeling zones | Constant (loaded from prior model input) | Yes |
| `maz` | Micro Analysis Zones — TAZ sub-units for output aggregation | CSV | **No** — output use only |

---

## `zones`

Traffic Analysis Zones used for travel demand modeling. Parcels reference `zone_id`, and travel data is indexed by zone pairs.

**Index:** `zone_id` (unique)

| Column | Type | Rules | Notes |
|---|---|---|---|
| `zone_id` | int16 | range [1 – 2811], no null | Model TAZ identifier |
| `taz` | object | no null | Full TAZ GEOID string |
| `tazce10_n` | int32 | no null | Census TAZ code (used for joining with travel model) |
| `acres` | float64 | no null | Zone area in acres |

**Note:** TransCAD uses `tazce10_n` codes internally. When converting travel matrices from OMX, always map through this table to translate travel model TAZ codes to `zone_id`. Using raw travel model codes will silently produce incorrect accessibility measurements.

**Common issues:**
- Any parcel without a matching `zone_id` in this table will have null travel-based variables — verify full TAZ coverage for the region

---

## `semmcds`

Municipal entity reference table — cities, townships, villages in the 7-county region.

**Index:** `semmcd_id` (unique)

| Column | Type | Rules | Notes |
|---|---|---|---|
| `semmcd_id` | int16 | range [5 – 7100], no null | SEMCOG MCD code — **this is the authoritative municipal identifier** |
| `county_id` | int16 | — | Parent county |
| `large_area_id` | int16 | — | Parent large area |
| `area_name` | object | no null | Municipality name |

> **`semmcd` vs `city_id`:** Throughout the model, `semmcd` is the official municipal code. The column name `city_id` appears in many tables as a synonym for `semmcd` — **except in the City of Detroit**, where `city_id` refers to a neighborhood subdivision rather than the municipality as a whole. This is a legacy naming convention. When working with Detroit data, verify whether a `city_id` value represents the whole city (Detroit's semmcd is 5) or a neighborhood code. See the [Glossary](../index.md#glossary) for full context.

---

## `counties`

The 7 SE Michigan counties. This is a **constant table** — it rarely needs updating.

**Index:** `county_id` (unique)

**Valid county codes** (Michigan county FIPS codes):

| Code | County |
|---|---|
| 93 | Livingston |
| 99 | Macomb |
| 115 | Monroe |
| 125 | Oakland |
| 147 | St. Clair |
| 161 | Washtenaw |
| 163 | Wayne |

| Column | Type | Rules |
|---|---|---|
| `county_id` | int16 | 7 valid codes, no null |
| `county_name` | object | no null |

---

## `large_areas`

The 8 modeling zones used for model segmentation, control totals, and output aggregation. This is a **constant table** — it does not change between forecast cycles.

**Index:** `large_area_id` (unique)

**Valid large area codes:**

| Code | Area | Note |
|---|---|---|
| 3 | Wayne County (excl. Detroit) | Custom code — splits Wayne County |
| 5 | City of Detroit | Custom code — splits Detroit from rest of Wayne |
| 93 | Livingston County | Michigan FIPS code |
| 99 | Macomb County | Michigan FIPS code |
| 115 | Monroe County | Michigan FIPS code |
| 125 | Oakland County | Michigan FIPS code |
| 147 | St. Clair County | Michigan FIPS code |
| 161 | Washtenaw County | Michigan FIPS code |

Wayne County (FIPS 163) is split into two large areas — the City of Detroit (code 5) and the rest of Wayne County (code 3). All other large areas share their code with the Michigan county FIPS number. The `counties` table uses FIPS code 163 for all of Wayne County; the split into codes 3 and 5 only exists in `large_areas`.

| Column | Type | Notes |
|---|---|---|
| `large_area_id` | int16 | 8 valid codes, no null |
| `large_area_name` | object | Internal short name (e.g., "outway", "det", "oak") — use `large_area_id` for joins |
| `large_area_group_id` | int8 | Values 1–4: regional grouping — 1=Suburban Tri-County (Wayne suburbs, Oakland, Macomb), 2=Western Corridor (Washtenaw, Livingston), 3=Outer Counties (Monroe, St. Clair), 4=Detroit |
| `alt_large_id` | int8 | Values 1–8: sequential alias for the 8 large areas, used in some model output routines |

---

## `maz`

> **Not a model input.** This table is not assembled into the model input file and is not read by the simulation. It is maintained here as a geographic reference used for **post-processing simulation outputs** at the MAZ level.

Micro Analysis Zones (MAZ) are sub-units of TAZs used in activity-based travel demand modeling. Each TAZ is divided into approximately 10 MAZs on average, providing a finer spatial resolution for output aggregation and reporting.

**~28,647 MAZs** across the 7-county region, each nested within a TAZ.

| Column | Type | Notes |
|---|---|---|
| `MAZ_SEQID` | int | Unique MAZ identifier (sequential, range 1–99999) |
| `TAZ_ID` | int | Parent TAZ — joins to `zones.zone_id` |
| `TAZCE10_N` | int | Census TAZ code — joins to `zones.tazce10_n` |
| `Acres` | float | MAZ area in acres |
| `COUNTY` | int | County FIPS code (93, 99, 115, 125, 147, 161, 163) |

**Relationship to other geography tables:**

```
TAZ (2,811 zones)
  └── MAZ (~10 per TAZ on average → 28,647 total)
```

Parcels do not carry a `maz_id` column — the MAZ assignment for output is derived by spatially joining parcel centroids to the MAZ boundary layer.

**Update:** Regenerate when TAZ boundaries change or when a finer MAZ delineation is adopted. The current file is version 2 (October 2023). Coordinate with the travel demand modeling team when updating, as MAZ boundaries should align with the TAZ structure used in the travel model.

---

## Update Notes

**`zones` and `semmcds`** — update when TAZ or MCD boundaries change (typically when census boundaries are updated or new municipalities form or dissolve).

**`counties` and `large_areas`** — stable definitions tied to the 7-county region and 8 modeling zones. Only update if the model's geographic scope changes, which is a major structural decision.

When updating any geography table, verify all downstream joins:
- `parcels.zone_id` → `zones`
- `parcels.semmcd` → `semmcds`
- `parcels.county_id` → `counties`
- `parcels.large_area_id` → `large_areas`

Missing join matches will result in null spatial attributes on parcels, which propagates to all buildings, households, and jobs on those parcels.
