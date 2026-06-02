# Accessibility System

## Overview

Accessibility metrics quantify how easy it is to reach various destinations from each parcel. These metrics are key inputs to the HLCM and ELCM — households prefer locations with good access to jobs, schools, and services; employers prefer locations accessible to workers.

The accessibility system has two components:

1. **Pre-computed parcel indicators** — static tables computed externally using Pandana network analysis (run once, not updated each year)
2. **Dynamic network aggregations** — computed each year by the `build_networks_2050` step using Pandana

---

## Pre-Computed Parcel Indicators

Three CSV files (one per mode) are loaded as orca tables at startup:

| Table | File | Mode |
|---|---|---|
| `accessibility_walk_indicator_by_parcel` | `walk_indicators_by_parcel_20251111.csv` | Walking |
| `accessibility_bike_indicator_by_parcel` | `bike_indicators_by_parcel_20251111.csv` | Cycling |
| `accessibility_drive_indicator_by_parcel` | `drive_indicators_by_parcel_20251111.csv` | Driving |

All are indexed by `parcel_id`. These represent accessibility from the 2024 transportation network and are held constant throughout the simulation.

**Source:** SEMCOG 2025 Transportation Accessibility Analysis (`/mnt/hgfs/urbansim/Accessibility/access_to_core_2024/`)

---

## Near-Max Variables (Proximity)

These measure the distance or time to reach the nearest facility of each type, expressed as the **90th percentile** of distances (not the minimum — so it reflects typical access, not just proximity to one outlier location).

### Walk Near-Max (90-minute threshold)

| Variable | Destination |
|---|---|
| `hospitals_walk_near_max90` | Hospitals |
| `urgent_cares_walk_near_max90` | Urgent care centers |
| `health_centers_walk_near_max90` | Community health centers |
| `all_healthcare_walk_near_max90` | All healthcare (combined) |
| `grocery_stores_walk_near_max90` | Grocery stores |
| `libraries_walk_near_max90` | Public libraries |
| `parks_local_walk_near_max90` | Local parks |
| `parks_bike_walk_near_max90` | Bike-friendly parks |
| `parks_school_walk_near_max90` | School parks |
| `parks_local_school_walk_near_max90` | Local/school parks combined |
| `schools_k8_walk_near_max90` | K-8 schools |
| `schools_912_walk_near_max90` | High schools |
| `pharmacies_walk_near_max90` | Pharmacies |
| `childcare_walk_near_max90` | Childcare facilities |
| `fire_stations_walk_near_max90` | Fire stations |
| `fixed_route_bus_walk_near_max90` | Fixed-route bus stops |
| `american_job_centers_walk_near_max90` | American Job Centers |
| `community_colleges_walk_near_max90` | Community colleges |
| `passenger_train_stations_walk_near_max90` | Passenger rail stations |

### Bike Near-Max (120-minute threshold)

Same destinations as walk, plus:
- `passenger_airports_bike_near_max120`

### Drive Near-Max (150-minute threshold)

Subset of destinations:
- Hospitals, urgent care, health centers, all healthcare
- Grocery stores, libraries
- Parks (vehicle-accessible only)
- Schools K-8, schools 9-12
- Pharmacies, childcare, fire stations
- American Job Centers, community colleges
- Passenger airports, passenger train stations

---

## Cumulative Variables (Access to Jobs)

These count how many jobs are accessible within a given travel time — a measure of economic opportunity from each parcel.

### Walk Cumulative

| Variable | Meaning |
|---|---|
| `jobs_walk_cumulative_5min` | Jobs reachable within 5 min walk |
| `jobs_walk_cumulative_10min` | Jobs reachable within 10 min walk |
| `jobs_walk_cumulative_15min` | Jobs reachable within 15 min walk |
| `jobs_walk_cumulative_30min` | Jobs reachable within 30 min walk |
| `fixed_route_bus_weekday_walk_10min` | Bus stops within 10 min walk (weekday) |
| `fixed_route_bus_weekend_walk_10min` | Bus stops within 10 min walk (weekend) |

### Bike Cumulative

| Variable | Meaning |
|---|---|
| `jobs_bike_cumulative_5min` | Jobs reachable within 5 min bike |
| `jobs_bike_cumulative_10min` | Jobs within 10 min bike |
| `jobs_bike_cumulative_15min` | Jobs within 15 min bike |
| `jobs_bike_cumulative_30min` | Jobs within 30 min bike |
| `fixed_route_bus_weekday_bike_10min` | Bus stops within 10 min bike (weekday) |
| `fixed_route_bus_weekend_bike_10min` | Bus stops within 10 min bike (weekend) |

### Drive Cumulative

| Variable | Meaning |
|---|---|
| `jobs_drive_cumulative_10min` | Jobs within 10 min drive |
| `jobs_drive_cumulative_15min` | Jobs within 15 min drive |
| `jobs_drive_cumulative_20min` | Jobs within 20 min drive |
| `jobs_drive_cumulative_25min` | Jobs within 25 min drive |
| `jobs_drive_cumulative_30min` | Jobs within 30 min drive |
| `jobs_drive_cumulative_45min` | Jobs within 45 min drive |
| `jobs_drive_gravity_90min` | Gravity-weighted job access (90 min) |

---

## Dynamic Network Aggregations

The `build_networks_2050` step runs each year and builds Pandana street networks for walk and drive modes. This computes **node-level** aggregations:
- Jobs within N minutes of each network node
- Households within N minutes

These are registered on the `nodes_walk` and `nodes_drv` orca tables, then joined to buildings and parcels via the broadcast chain (`nodeid_walk`, `nodeid_drv` columns).

Unlike the static parcel indicators above, dynamic aggregations reflect the **current year's** job and household distribution.

**Network configuration:** `configs/available_networks_2050.yaml`

---

## How Accessibility Joins to Buildings

The pre-computed parcel indicators are indexed by `parcel_id`. They join to buildings via:

```
accessibility_walk_indicator_by_parcel (parcel_id index)
         ↓  merged on parcel_id
buildings (parcel_id column)
```

This is handled in `variables_access.py` which registers orca columns on the buildings table that look up accessibility values via the parcel foreign key.

The `fillna` values handle parcels with no network coverage:
- Walk: 95.0 minutes (imputed max)
- Bike: 125.0 minutes
- Drive: 155.0 minutes
- Cumulative: 0 (no jobs accessible)
