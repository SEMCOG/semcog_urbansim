# Environment Setup

## Requirements

- **Docker** (with GPU support via NVIDIA Container Toolkit)
- **Host machine:** Windows with drives `D:` and network share `U:` mounted
- **Docker image:** `forecast-sim-image.tar` (stored in `D:\docker` on host / `urbansim4`)

---

## Docker Image

### Load the Image

```bash
# On the host machine (Windows)
cd D:\docker
docker load -i forecast-sim-image.tar
```

This loads the image as `forecast_simulation`.

### Run the Container

```bash
docker run \
  --gpus all \
  --name forecast-sim \
  --dns 192.168.182.10 \
  --dns-search semcogdom.local \
  --privileged \
  --cap-add SYS_ADMIN \
  --device /dev/fuse \
  -v D:\:/mnt/D \
  -v D:\RDF2050:/mnt/hgfs/RDF2050 \
  -v "U:\:/mnt/hgfs/urbansim" \
  -v D:\projects\semcog_urbansim:/mnt/semcog_urbansim \
  -itd forecast_simulation
```

**Volume mounts:**

| Host Path | Container Path | Contains |
|---|---|---|
| `D:\` | `/mnt/D` | Local D drive (travel survey data, etc.) |
| `D:\RDF2050` | `/mnt/hgfs/RDF2050` | Model estimation outputs, trained models |
| `U:\` | `/mnt/hgfs/urbansim` | Network share: input HDF5, accessibility data |
| `D:\projects\semcog_urbansim` | `/mnt/semcog_urbansim` | Project source code |

---

## After Container Start

### Mount Network Drives

```bash
# Inside the container
sudo mount -a
```

This mounts additional network drives defined in `/etc/fstab`. Required if model files or input data are on network shares not covered by the `-v` flags above.

### Verify Mounts

```bash
ls -l /mnt/semcog_urbansim         # project code
ls -l /mnt/hgfs/RDF2050            # model files
ls -l /mnt/hgfs/urbansim           # input data
ls -l /mnt/D                        # local data
```

---

## Python Environment

Inside the container, activate the conda environment:

```bash
micromamba activate forecast
```

All simulation commands must be run in this environment.

### Key Packages

| Package | Version | Purpose |
|---|---|---|
| `orca` | 1.8 | Simulation pipeline framework |
| `torch` | ≥2.6.0 | PyTorch (HLCM/ELCM models) |
| `xgboost` | latest | REPM training & prediction |
| `pandas` | latest | Data manipulation |
| `numpy` | latest | Numerical computation |
| `pandana` | latest | Street network accessibility |
| `scikit-learn` | latest | Scalers, Ridge regression |
| `cartoframes` | latest | CartoDB upload |
| `urbansim` | custom | UrbanSim utilities (transition, relocation, etc.) |
| `urbansim_parcels` | custom | Parcel utilities |
| `forecast_estimation` | custom | LCM estimation framework (ARD-DCM) |

---

## Working Directory

Inside the container, the working directory for running the simulation is:

```bash
cd /mnt/semcog_urbansim
```

All relative paths in the code (`configs/`, `runs/`, `data/`) resolve from this directory.

---

## Save & Export Container

After making changes to the container environment (installing packages, etc.):

```bash
# Commit current container state to image
docker commit forecast-sim forecast_simulation

# Export image to file for backup/sharing
docker save -o forecast-sim-image.tar forecast_simulation

# Load image on another machine
docker load -i forecast_simulation_final.tar
```

---

## GPU Access

The container is launched with `--gpus all`, giving PyTorch models access to all available GPUs. The HLCM and ELCM PyTorch models can use GPU acceleration for inference if a CUDA-compatible GPU is present.

Verify GPU availability inside the container:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```
