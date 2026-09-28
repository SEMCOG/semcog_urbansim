SEMCOG
======

[UrbanSim][] implementation for [SEMCOG][], used to produce the Regional Development Forecast (RDF) for Southeast Michigan. The current forecast cycle is RDF 2055.

Published forecast: [SEMCOG 2050 forecast][].

[UrbanSim]: https://github.com/UDST/urbansim
[SEMCOG]: https://www.semcog.org/
[SEMCOG 2050 forecast]: https://maps.semcog.org/forecast/

### Documentation

- [SEMCOG 2055 Model Input Wiki](https://semcog.github.io/semcog_urbansim/) — reference for data developers: input tables, schemas, validation, and update procedures

### Computing environment

The model runs inside a Docker container built from the included `Dockerfile`. The image provides a micromamba environment named `forecast` with the Python packages listed in `requirements.txt`. GPU support is used for location choice model estimation and simulation when available.

### Data organization

Input data and model outputs are kept outside the repository and mounted into the container:

- **Base-year inputs** — a single HDF5 store with households, persons, jobs, buildings, parcels, zones, and related tables (see the Model Input Wiki).
- **Control totals** — regional household and employment forecasts by large area.
- **Estimated models** — household and employment location choice models and real estate price models.
- **Accessibility and network data** — parcel-level accessibility indicators and travel networks.
- **Outputs** — simulation results written per run under `runs/`.

Input file locations are set in `input_paths.py`.

### Repository layout

| Path | Contents |
|---|---|
| `simulation_2055.py` | Main simulation entry point |
| `models.py`, `dataset.py`, `variables/` | Model steps, data tables, and computed variables |
| `configs/` | Model configuration (YAML) |
| `estimation/` | Model estimation code |
| `indicators/` | Output indicator summaries |
| `scripts/` | Data preparation and testing utilities |

### Running a simulation

From the project folder inside the container:

```
micromamba activate forecast
python simulation_2055.py
```
