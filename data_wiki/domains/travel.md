# Travel Data

**Primary source:** SEMCOG travel demand model outputs (OMX matrix files → CSV)  
**Build scripts:** Conversion scripts in `travel_inputs/`

Travel data provides zone-to-zone travel times and mode choice logsums used as features in the HLCM and ELCM to capture commute accessibility and transit competitiveness.

---

## Tables in This Domain

| Table | Description | Source |
|---|---|---|
| `travel_data` | Base-year zone-to-zone travel skims | CSV (converted from OMX) |
| `travel_data_2030` | Future-year zone-to-zone travel skims | CSV (converted from OMX) |

---

## Source OMX Files

The travel demand model produces zone-to-zone skim matrices in OMX format. These must be converted to flat CSV before inclusion in the HDF. Source OMX files are stored in the `travel_inputs/` directory of the data inputs folder.

Typical skim files include:
- Auto travel time (AM peak and midday)
- Transit travel time (AM peak and midday)
- Home-based work logsums by income group (low/mid/high)
- Home-based other (non-work) logsum

---

## `travel_data`

Base-year zone-to-zone travel skims. One row per origin-destination pair.

**Index:** Multi-index `(from_zone_id, to_zone_id)` (unique pairs)  
**Source:** CSV — path in `config/files.yaml`  
**All columns cast to float32** in the build pipeline

**Columns:**

| Column | Type | Rules | Notes |
|---|---|---|---|
| `from_zone_id` | int16 | range [1 – max zone], no null | Origin TAZ (model `zone_id`) |
| `to_zone_id` | int16 | range [1 – max zone], no null | Destination TAZ |
| `am_auto_total_time` | float32 | no null | AM peak auto travel time (minutes) |
| `am_transit_total_time` | float32 | no null | AM peak transit total time (minutes) |
| `am_work_lowinc_logsum` | float32 | no null | AM work mode choice logsum, low income |
| `am_work_midinc_logsum` | float32 | no null | AM work mode choice logsum, mid income |
| `am_work_highinc_logsum` | float32 | no null | AM work mode choice logsum, high income |
| `midday_auto_total_time` | float64 | no null | Midday auto travel time (minutes) |
| `midday_transit_total_time` | float64 | no null | Midday transit total time |
| `midday_other_logsum` | float64 | no null | Midday non-work trip logsum |

---

## `travel_data_2030`

Future-year travel skims representing conditions after planned transportation investments. Same schema as `travel_data`. Allows the model to use time-varying travel conditions for later simulation years.

---

## TAZ ID Mapping

The travel demand model uses its own internal TAZ codes, which differ from the model's `zone_id` values. **Always map through the `zones` table** when converting OMX to CSV:

- Travel model TAZ code → `zones.tazce10_n` → `zones.zone_id`

Using the wrong TAZ codes will silently produce incorrect accessibility measurements.

---

## Conversion from OMX to CSV

General process:
1. Open OMX files using the `openmatrix` or `omx` Python library
2. Extract each skim as a zone × zone matrix
3. Flatten to long format: one row per origin-destination pair with columns for each skim
4. Map travel model TAZ codes to `zone_id` via the `zones` table
5. Join all skims on `(from_zone_id, to_zone_id)` to produce one combined CSV
6. Save with the multi-index `(from_zone_id, to_zone_id)` and update the path in `config/files.yaml`

---

## Update Checklist (New Forecast Cycle)

- [ ] Get new OMX skims from the travel model run (base year and future year)
- [ ] Run OMX-to-CSV conversion scripts in `travel_inputs/`
- [ ] Verify TAZ coverage — all zones should appear as both origin and destination
- [ ] Verify no null values in any skim column
- [ ] Check plausibility: auto times must be positive; near-zero values are expected for intra-zonal pairs
- [ ] Update file path in `config/files.yaml`
- [ ] Run validation after HDF assembly

**Common issues:**
- TAZ ID mismatch — always map through the `zones` table; see note above
- Zero or negative travel times — usually an OMX extraction error; auto times must be positive
- Missing zone pairs — fill with a high default time (e.g., 999 minutes)
