# Travel Data

**Primary source:** SEMCOG travel demand model outputs (OMX matrix files → CSV)

Travel data provides zone-to-zone travel times and mode choice logsums used as features in the HLCM and ELCM to capture commute accessibility and transit competitiveness.

---

## Tables in This Domain

| Table | Description | Source |
|---|---|---|
| `travel_data` | Base-year zone-to-zone travel skims | CSV (converted from OMX) |
| `travel_data_2030` | Future-year zone-to-zone travel skims | CSV (converted from OMX) |

---

## Source OMX Files

The travel demand model produces zone-to-zone skim matrices in OMX format. These must be converted to flat CSV before inclusion in the model input file. 

Typical skim files include:
- Auto travel time (AM peak and midday)
- Transit travel time (AM peak and midday)
- Home-based work logsums by income group (low/mid/high)
- Home-based other (non-work) logsum

---

## `travel_data`

Base-year zone-to-zone travel skims. One row per origin-destination pair.

**Index:** Multi-index `(from_zone_id, to_zone_id)` (unique pairs)  
**All columns cast to float32** in the build pipeline

**Columns:**

| Column | Type | Rules | Notes |
|---|---|---|---|
| `from_zone_id` | int16 | range [1 – max zone], no null | Origin TAZ (model `zone_id`) |
| `to_zone_id` | int16 | range [1 – max zone], no null | Destination TAZ |
| `am_auto_total_time` | float32 | no null | AM peak auto travel time (minutes) |
| `am_transit_total_time` | float32 | no null | AM peak transit total time (minutes) |
| `am_work_lowinc_logsum` | float32 | no null | AM work trip **mode choice logsum**, low income |
| `am_work_midinc_logsum` | float32 | no null | AM work trip **mode choice logsum**, mid income |
| `am_work_highinc_logsum` | float32 | no null | AM work trip **mode choice logsum**, high income |
| `midday_auto_total_time` | float32 | no null | Midday auto travel time (minutes) |
| `midday_transit_total_time` | float32 | no null | Midday transit total time |
| `midday_other_logsum` | float32 | no null | Midday non-work trip **mode choice logsum** |

> **What is a mode choice logsum?**
> The mode choice logsum is the expected maximum utility a traveler can achieve between an origin-destination pair, considering all available travel modes (auto, transit, walk, etc.) and their relative costs and times. A higher logsum value means better overall multimodal accessibility between two zones — not just faster driving, but better access by any mode.
>
> These logsums are **not** destination choice logsums (which aggregate over all destinations). They are zone-to-zone matrices used in UrbanSim as gravity-type impedance weights: the model computes zone-level accessibility as `Σ [population or jobs at destination] × exp(logsum)` across all destination zones, capturing how well each zone is connected to opportunities by all modes combined. Work-trip logsums are segmented by income because lower-income households are more sensitive to transit access and cost.

---

## `travel_data_2030`

Future-year travel skims representing conditions after planned transportation investments. Same schema as `travel_data`. Allows the model to use time-varying travel conditions for later simulation years.

---

## TAZ ID Mapping — Critical

The travel demand model uses its own internal TAZ codes, which differ from the model's `zone_id` values. **Always map through the `zones` table** when converting OMX to CSV:

```
Travel model TAZ code  →  zones.tazce10_n  →  zones.zone_id
```

Using the wrong TAZ codes will silently produce incorrect accessibility measurements without any error — this is one of the most common and hard-to-detect mistakes in this pipeline.

---

## Conversion from OMX to CSV

General process:
1. Open OMX files using the `openmatrix` or `omx` Python library
2. Extract each skim as a zone × zone matrix
3. Flatten to long format: one row per origin-destination pair, one column per skim
4. Map travel model TAZ codes to `zone_id` via the `zones` table
5. Join all skims on `(from_zone_id, to_zone_id)` to produce one combined CSV
6. Save with the multi-index `(from_zone_id, to_zone_id)`

---

## Update Checklist (New Forecast Cycle)

- [ ] Get new OMX skims from the travel model run (base year and future year)
- [ ] Run OMX-to-CSV conversion
- [ ] Verify TAZ ID mapping — travel model codes → `zones.tazce10_n` → `zones.zone_id`
- [ ] Verify TAZ coverage — all zones should appear as both origin and destination
- [ ] Verify no null values in any skim column
- [ ] Check plausibility: auto times must be positive; near-zero values are expected only for intra-zonal pairs
- [ ] Confirm row count = number of zones² (all origin-destination combinations present)

**Common issues:**
- TAZ ID mismatch — always map through the `zones` table; see note above
- Zero or negative travel times — usually an OMX extraction error; auto times must be positive
- Missing zone pairs — fill with a high default time (e.g., 999 minutes) rather than leaving null
