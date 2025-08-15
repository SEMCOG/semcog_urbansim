SEMCOG
======

[UrbanSim2][] implementation for [SEMCOG][].

[SEMCOG 2050 forecast][].

[UrbanSim2]: https://github.com/synthicity/urbansim
[SEMCOG]: http://www.semcog.org/
[SEMCOG 2050 forecast]: https://maps.semcog.org/forecast/

### Locate `semcog_urbansim` project folder
#### Project folder and data
Make sure folder `D:\projects\semcog_urbansim` has both repo and `data` in it.
(Copy from `urbansim4`)

#### Required libraries
Make sure folder `D:\RDF2055\libs` exist and contains all required libs
(Copy from `urbansim4`)

### Docker Image loading
`forecast-sim-image.tar` is located in `urbansim4` `D:\docker`. Inside the folder, do
```
docker load -i forecast-sim-image.tar
```

### Docker Run Script
```
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

### After creating container
#### Mount all network drive
mount via fstab, do
```
sudo mount -a
```
#### Check if drives mounted correctly
```
ls -l /mnt/semcog_urbansim
ls -l /mnt/hgfs/RDF2050
ls -l /mnt/hgfs/urbansim
```

### Development
Project folder will be located in `/mnt/semcog_urbansim`
#### Activate `forecast` env
```
micromamba activate forecast
```

#### Running similation with complete loggings
```
nohup python test_forecast_2050.py >> runs/run_stdout/simulation_log.txt 2>&1 &
```