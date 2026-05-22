# SEMCOG UrbanSim Model

## What Is This Model?

The **SEMCOG UrbanSim** model is an agent-based land use simulation system that forecasts how households, jobs, and buildings distribute across Southeast Michigan from a **base year of 2020** through **2050** (branch `forecast_2055` extends this to 2055).

The model is used by the [Southeast Michigan Council of Governments (SEMCOG)](https://www.semcog.org) to inform regional transportation planning, housing policy, and long-range forecasts. Results are published publicly at [maps.semcog.org/forecast](https://maps.semcog.org/forecast/).

---

## What Does It Simulate?

Each simulated year, the model:

1. **Transitions** the regional population and employment to match externally-provided control totals (from the REMI economic model)
2. **Demolishes** existing buildings via scheduled events or probabilistic rates
3. **Builds** new residential and non-residential space where it is financially feasible
4. **Places** households into buildings using a neural network location choice model
5. **Places** jobs into buildings using a neural network location choice model
6. **Updates** real estate prices using XGBoost hedonic price models
7. **Writes** a snapshot of all entity tables to an HDF5 output file

The result is a year-by-year spatial picture of where people live and work across the region's ~1.9 million parcels.

---

## Quick Navigation

| I want to... | Go to |
|---|---|
| Understand how the whole system fits together | [Architecture Overview](architecture/overview.md) |
| See what runs in what order each year | [Annual Pipeline](architecture/annual-pipeline.md) |
| Understand the household location choice model | [HLCM](models/hlcm.md) |
| Understand the employment location choice model | [ELCM](models/elcm.md) |
| Understand how new buildings get built | [Developer Models](models/developer.md) |
| See what data goes in and comes out | [Data Flow](architecture/data-flow.md) |
| Run the simulation | [Running a Simulation](running/run-simulation.md) |
| Set up the Docker environment | [Environment Setup](running/environment.md) |
| Change scenario assumptions | [Scenario Controls](configuration/scenarios.md) |
| Resume a stopped run | [Checkpoints](configuration/checkpoints.md) |
| Look up what a variable means | [Glossary](reference/glossary.md) |

---

## Technology Stack

| Component | Technology |
|---|---|
| Pipeline orchestration | [orca](https://udst.github.io/orca/) (dependency injection framework) |
| Location choice models | PyTorch neural networks (`.pt` files) |
| Real estate price models | XGBoost gradient boosting |
| Developer feasibility | UrbanSim pro-forma analysis |
| Accessibility computation | [Pandana](https://udst.github.io/pandana/) street network analysis |
| Data storage | HDF5 (pandas HDFStore) |
| Output visualization | CartoDB / PostgreSQL |
| Runtime environment | Docker container (`forecast_simulation` image) |

---

## Repository

- **Branch:** `forecast_2055`
- **Base year:** 2020
- **Forecast horizon:** 2050 (extendable to 2055)
- **Primary entry point:** [`test_forecast_2050.py`](../test_forecast_2050.py)
- **Core simulation logic:** [`models.py`](../models.py) (~3,000 lines)
