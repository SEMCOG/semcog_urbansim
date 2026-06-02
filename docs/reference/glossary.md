# Glossary

Key terms used throughout the SEMCOG UrbanSim model documentation.

---

## A

**ARD-DCM**
Attention-based Relevance Detection Discrete Choice Model. The neural network framework used to train the HLCM and ELCM models. Extends traditional discrete choice modeling with attention mechanisms that learn variable importance.

**Accessibility**
A measure of how easily destinations (jobs, services, parks) can be reached from a location by walking, biking, or driving. Used as a feature in HLCM and ELCM. See [Accessibility System](accessibility.md).

**Orca**
The dependency injection and simulation orchestration framework (formerly urbansim3). Tables, columns, injectables, and steps are registered with orca; orca resolves dependencies and executes steps in order.

---

## B

**Base Year**
The starting year of the simulation — 2020. All entity tables (households, jobs, buildings, parcels) represent the real-world state as of this year.

**Broadcast**
An orca concept representing a foreign-key relationship between tables. Allows columns from one table (e.g., `parcels`) to be accessed on a related table (e.g., `buildings`) without explicit join code.

**Building Type Map**
The mapping from numeric `building_type_id` values to general category strings (Residential, Office, Industrial, etc.). Defined in `assumptions.py`.

---

## C

**Cap Rate**
Capitalization rate used in the pro-forma feasibility analysis. Converts annual net operating income to an estimated property value. Set to 0.01 in `proforma.yaml`.

**Checkpoint**
A saved simulation state at the end of a completed year, stored in the output HDF5. Can be used to resume a run. See [Checkpoints & Resume](../configuration/checkpoints.md).

**Control Totals**
External forecasts of total household and employment counts by geography and segment for each simulation year. Sourced from REMI economic model. The transition models use these to add or remove agents each year.

**Cumulative Access Variable**
An accessibility metric counting the number of destinations (typically jobs) reachable within a given travel time threshold. E.g., `jobs_walk_cumulative_10min`.

---

## D

**DCM**
Discrete Choice Model. A statistical model where agents choose among a set of discrete alternatives. Both HLCM and ELCM are discrete choice models (choosing a building from a set of alternatives).

**Developer Model**
The component that simulates market-rate real estate construction. Uses pro-forma analysis to determine which parcels are financially viable for new development and then builds on them.

---

## E

**ELCM**
Employment Location Choice Model. Assigns jobs to buildings. PyTorch neural network, segmented by large area × employment sector × home-based status. See [ELCM](../models/elcm.md).

**Employment Sector**
A category of economic activity (manufacturing, retail, healthcare, etc.). The ELCM is estimated and applied separately for each sector. See [Data Schema](../data/schema.md) for sector IDs.

**Event Building**
A building added via the `events_addition` table or `refiner_events`. Marked with `sp_filter = -1` to protect it from random relocation and demolition.

---

## F

**FAR (Floor Area Ratio)**
The ratio of total building floor area to parcel area. Higher FAR = denser development. The feasibility model tests many FAR values to find the most profitable development for each parcel.

**Feasibility**
The result of the pro-forma analysis — whether the expected revenue from development exceeds the costs (construction, land, financing) by the required profit margin.

**Fine-Tuning**
Retraining a pre-trained model on additional data. The current HLCM models (`models_survey_finetune`) were initially trained on observed household locations, then fine-tuned using travel survey behavioral data.

---

## H

**HDF5**
Hierarchical Data Format version 5. The primary data storage format used by the model. The input data and all run outputs are HDF5 files. Accessed via `pandas.HDFStore`.

**Hedonic ID**
A combined segment identifier used by the REPM: `building_type_id × large_area_id`. Each REPM model covers one hedonic_id segment.

**HLCM**
Household Location Choice Model. Assigns households to buildings. PyTorch neural network, segmented by large area × demographic characteristics. See [HLCM](../models/hlcm.md).

**Home-Based Job**
A job located in a residence (e.g., telecommuter, home business). `home_based_status >= 1`. Placed in residential buildings by the ELCM.

**HU**
Housing Unit. A residential unit in a building.

---

## I

**Injectable**
An orca concept for a scalar value or object (not a table) that can be injected into any step as a function argument. Examples: `year`, `hlcm_model_path`, `repm_step_names`.

---

## L

**Large Area**
One of 8 geographic modeling zones that roughly correspond to the 7 SE Michigan counties (with Detroit treated separately from the rest of Wayne County). Used for model segmentation and control total targeting.

**Landmark Worksite**
A major employer (large factory, hospital, university) whose jobs are protected from simulation randomness via `sp_filter = -1`.

**LCM**
Location Choice Model. Generic term for HLCM or ELCM.

---

## M

**MCD**
Minor Civil Division — the legal municipal entities (cities, townships, villages) in Michigan. The `semmcd` code is SEMCOG's internal MCD identifier. MCD-level targets drive `mcd_hu_sampling`.

**MCD Quota**
The `mcd_model_quota` column on buildings, set annually by `mcd_hu_sampling`. It caps how many new households the HLCM can place in a given building this year, ensuring city-level growth aligns with `mcd_total` targets.

**MNL**
Multinomial Logit. The traditional discrete choice model. Used in early UrbanSim implementations; replaced here by neural networks (PyTorch).

---

## N

**Near-Max Variable**
An accessibility metric measuring proximity to the nearest facility of a given type, expressed as the 90th-percentile of distances from a parcel. E.g., `hospitals_walk_near_max90`.

**Non-Home-Based Job**
A job located in a commercial or industrial building. `home_based_status = 0`. Placed in non-residential buildings by the ELCM.

---

## O

**Orca Step**
A Python function decorated with `@orca.step()` that represents one unit of simulation work. Steps are executed in order by `orca.run()`. Each step declares its data dependencies as function arguments.

---

## P

**Pandana**
Python library for network-based accessibility analysis. Used by `build_networks_2050` to compute job accessibility from every network node using preloaded street networks.

**Pro-Forma**
A financial feasibility analysis estimating revenue and costs for a hypothetical development project. The foundation of the developer model.

**Pseudo-Building**
A temporary building (ID > 90,000,000) added at startup for households and jobs with invalid building assignments. Removed by `drop_pseudo_buildings` after the location choice models have placed these agents.

---

## R

**REMI**
Regional Economic Models Inc. Provides the external demographic and economic forecasts (control totals) that drive household and employment transitions in the simulation.

**REPM**
Real Estate Price Model. XGBoost-based hedonic model that estimates `sqft_price_res` and `sqft_price_nonres` for each building annually. See [REPM](../models/repm.md).

**Refiner**
The component that applies event-based policy interventions — adding or removing buildings based on the `refiner_events` table. See [Policy Interventions](../models/refiner.md).

**Relocation**
The process of un-placing a household or job (setting `building_id = -1`) so it can be re-placed by the location choice model. Driven by rates in `annual_relocation_rates_for_households`.

---

## S

**SEMCOG**
Southeast Michigan Council of Governments. The regional planning agency that develops and uses this model.

**sp_filter**
Special filter column on buildings. Controls whether buildings are eligible for simulation processes. 0 = normal; -1 = protected event/landmark building; -2 = pseudo-building.

---

## T

**TAZ**
Traffic Analysis Zone. The spatial unit used in travel demand modeling. The HLCM uses TAZ-level trend variables to capture neighborhood change dynamics.

**Transition Model**
The component that adds or removes agents (households, jobs) to match control totals. Does not determine location — only quantity.

---

## U

**UrbanSim**
The open-source urban simulation framework that SEMCOG's model is built on. Provides the pro-forma developer model, transition/relocation utilities, and the orca framework.

**Unplaced Agent**
A household or job with `building_id = -1`. Needs to be assigned to a building by the location choice model.

---

## V

**Vacant Residential Unit**
The difference between `residential_units` and the count of households currently assigned to a building. The HLCM uses this as capacity when placing households.

**Variable**
In orca terms, a derived column registered with `@orca.column()`. Computed on demand from base table data. Defined in the `variables/` package.
