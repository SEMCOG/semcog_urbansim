# Re-estimating Models

The SEMCOG UrbanSim uses three types of pre-trained models: HLCM, ELCM, and REPM. Each has its own estimation workflow. Re-estimation is needed when:

- Base year data is updated
- New variables are added or removed
- Model performance has degraded
- A new forecast horizon requires different calibration

---

## Household Location Choice Model (HLCM)

### Script: `HLCM_estimation.py`

**Method:** ARD-DCM (Attention-based Relevance Detection Discrete Choice Model) — a PyTorch neural network trained on residential location decisions.

**Training data:** Observed household locations from the base-year HDF5, matched to building/neighborhood attributes. The `models_survey_finetune` path indicates models were additionally fine-tuned using travel survey responses.

**Output:** One `.pt` file per segment (large area × children × seniors × income) in the configured output directory.

**Key steps:**
1. Load household and building data from HDF5
2. For each segment: assemble choosers and alternatives
3. Train neural network with ARD-DCM framework
4. Save trained model and scaler to `.pt` file

**After estimation:**
1. Place `.pt` files in a new subdirectory under the models folder (e.g., `models_YYYYMMDD/pts/`)
2. Update `hlcm_model_path` in `test_forecast_2050.py` to point to the new directory
3. Run `lcm_utils.load_hlcm_model_configs_from_path()` will auto-discover the new files

---

## Employment Location Choice Model (ELCM)

### Script: `ELCM_estimation.py`

**Method:** Same ARD-DCM PyTorch framework as HLCM.

**Training data:** Observed job locations from base-year HDF5.

**Output:** One `.pt` file per segment (large area × sector × home_based_status).

**After estimation:**
1. Place `.pt` files in a new subdirectory (e.g., `elcm_models_YYYYMMDD/pts/`)
2. Update `elcm_model_path` in `test_forecast_2050.py`

---

## Real Estate Price Model (REPM)

### Scripts:
- `REPM_feature_selection.py` — selects the best variable subset for each segment
- `REPM_xgb_training.py` — trains XGBoost (or Ridge) models with hyperparameter search

**Method:** XGBoost gradient boosting on building transaction data. Ridge regression fallback for segments with < 100 observations.

**Training data:** Building transaction records joined to building/neighborhood attributes.

**Key training parameters (from `REPM_xgb_training.py`):**
- Grid search over XGBoost hyperparameters (n_estimators, max_depth, learning_rate, etc.)
- Cross-validation to select best parameters
- Feature selection run first to reduce dimensionality

**Output per segment:**
- `configs/repm_xgb/{segment_name}/model.pkl` — trained XGBoost or Ridge model
- `configs/repm_xgb/{segment_name}/metadata.pkl` — feature list, scaler, segment info

**After estimation:**
- Model files are stored directly in `configs/repm_xgb/` (in the project repo)
- `models.py` auto-discovers them at startup — no path changes needed
- If segment directories change, `repm_step_names` is automatically updated

---

## Variable Definitions for Estimation

**Script:** `estimation_variables_2050.py`

Contains orca variable definitions used only during estimation (not simulation). These variables may reference the full input HDF5 or external data sources that are not available during simulation runtime.

---

## Estimation Tips

### HLCM/ELCM Segment Coverage

Not all segments need a trained model. If a segment has very few observations (< ~50 households or jobs), estimation may fail or produce poor models. In that case:
- Skip estimation for that segment
- The simulation will fall back to random placement for that segment

### Checking Model Coverage

```bash
# Count HLCM models
ls /mnt/hgfs/RDF2050/estimation/models/models_survey_finetune/pts/*.pt | wc -l

# Count ELCM models
ls /mnt/hgfs/RDF2050/estimation/models/elcm_models_25May30/pts/*.pt | wc -l

# Count REPM models
ls -d configs/repm_xgb/*/ | grep -v grid_search | wc -l
```

### Testing After Re-estimation

After updating models, run a short test before a full 30-year run:

```bash
# Modify test_forecast_2050.py to set final_year = base_year + 3
python test_forecast_2050.py
```

Check that:
- All model steps load without error
- Households are being placed (check log output from HLCM steps)
- Jobs are being placed (check ELCM output)
- REPM is updating prices (check `sqft_price_res` changes in output HDF5)

---

## Analysis Notebooks

The `notebooks/` directory contains `.py` files (Jupyter-style) for analyzing model performance:

| Notebook | Purpose |
|---|---|
| `Location Choice Model.py` | HLCM/ELCM prediction accuracy analysis |
| `Residential Price Model.py` | REPM fit diagnostics |
| `Proforma Demo.py` | Developer feasibility exploration |
| `model_analysis/` | Additional estimation diagnostics |
