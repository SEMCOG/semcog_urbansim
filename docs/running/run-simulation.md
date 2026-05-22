# Running a Simulation

## Quick Start

```bash
# 1. Activate environment
micromamba activate forecast

# 2. Navigate to project
cd /mnt/semcog_urbansim

# 3. Launch simulation (background with full logging)
nohup python test_forecast_2050.py >> runs/run_stdout/simulation_log.txt 2>&1 &
```

The run number is assigned automatically. Output is written to `runs/runNNN.h5`.

---

## Before Running

### Check Configuration

Open `test_forecast_2050.py` and verify:

- [ ] `base_year` and `final_year` are correct
- [ ] `hlcm_model_path` points to the right model directory and that `pts/` exists inside it
- [ ] `elcm_model_path` points to the right ELCM model directory
- [ ] `ENABLE_SCENARIO` is set as intended (`False` for baseline)
- [ ] `use_checkpoint` is set as intended (`False` for fresh run)
- [ ] `upload_to_carto` is correct (`True` for production, `False` for test runs)

### Check Disk Space

The output HDF5 for a full 2020–2050 run is typically **10–20 GB**. Ensure sufficient disk space in the `runs/` directory.

```bash
df -h .
```

### Verify Input Data Is Accessible

```bash
ls /mnt/hgfs/urbansim/RDF2050/model_inputs/base_hdf/
ls /mnt/hgfs/RDF2050/estimation/models/models_survey_finetune/pts/ | head
ls /mnt/hgfs/RDF2050/estimation/models/elcm_models_25May30/pts/ | head
```

---

## Running Modes

### Background Run (recommended for full runs)

```bash
nohup python test_forecast_2050.py >> runs/run_stdout/simulation_log.txt 2>&1 &
echo $!   # print the background process PID
```

### Foreground Run (for debugging)

```bash
python test_forecast_2050.py
```

### Test Run (short)

Use `test.py` for a quick pipeline test without running the full 30-year loop:

```bash
python test.py
```

---

## Monitoring Progress

### Tail the Log

```bash
tail -f runs/run_stdout/simulation_log.txt
```

Each year's completion is logged. Look for:
```
Year 2025 complete
Year 2030 complete
...
Total run time: 04:32:17
```

### Check Which Runs Are Complete

```bash
ls -la runs/*.h5
```

To check the last completed year in an HDF5:

```python
import pandas as pd
store = pd.HDFStore('runs/runNNN.h5', 'r')
years = [int(k.split('/')[1]) for k in store.keys() if k.split('/')[1].isdigit()]
print("Last completed year:", max(years))
store.close()
```

---

## Typical Run Time

A full 2021–2050 run (30 years) typically takes **4–8 hours** depending on:
- Hardware (CPU cores, GPU availability)
- Number of HLCM/ELCM models loaded
- Network drive access speed for model files

---

## Stopping a Run

```bash
# Find the process
ps aux | grep test_forecast_2050

# Kill it
kill <PID>
```

The run output HDF5 is valid up to the last completed year — it can be resumed using a checkpoint.

---

## After the Run

### Verify Output

```python
import pandas as pd

store = pd.HDFStore('runs/runNNN.h5', 'r')
print("Keys:", [k for k in store.keys() if '/2050/' in k])
hh_2050 = store['/2050/households']
print("Households 2050:", len(hh_2050))
store.close()
```

### Check Indicator Upload

If `upload_to_carto = True` and `RUN_OUTPUT_INDICATORS = True`, check the log for Carto upload confirmation.

### Run Config

Review `runs/runNNN/run_config.yaml` to confirm the run used the expected model version and settings.

---

## Common Issues

### "Not enough locations for movers"

Printed by the HLCM or ELCM when there are more unplaced agents than available units/spaces in a large area. The model places as many as it can and logs the shortfall. This is handled downstream by `jobs_scaling_model` for jobs.

**Cause:** Developer model built less space than demand requires for a given large area/segment.

**Fix:** Check `mcd_model_quota` values; verify developer model is running and producing output; check target vacancy rates in `res_developer.yaml`.

### Network Drive Not Mounted

```
FileNotFoundError: /mnt/hgfs/urbansim/...
```

Run `sudo mount -a` inside the container.

### Out of Disk Space

```
OSError: [Errno 28] No space left on device
```

The HDF5 output is being written. Free space in the `runs/` directory or redirect output to a different drive.

### Import Error on `models.py`

Most import errors at startup indicate a missing PyTorch model file or a path mismatch. Check that `hlcm_model_path` and `elcm_model_path` are correct and that the `pts/` subdirectory exists and contains `.pt` files.
