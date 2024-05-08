## Map Training

Whenever training and evaluating, please edit the config paths (and checkpoint paths if testing) in the bash file. Also, change the data paths in the config files to your nuscenes raw data and processed annotation data. 

### MapTR

**Training**

Run
```
cd MapTR_modified/
source train.sh      
```

Or by running:
```
export PYTHONPATH="${PYTHONPATH}:/MapUncertaintyPrediction/MapTR_modified"

python tools/train.py \
  projects/configs/maptr/maptr_tiny_r50_24e.py \
  --deterministic \
  --no-validate

```

**Evaluation**

Run
```
cd MapTR_modified/
source test.sh                                  
```

Or by running:

```
export PYTHONPATH="${PYTHONPATH}:/MapUncertaintyPrediction/MapTR_modified"

python tools/test.py \
  projects/configs/maptr/maptr_tiny_r50_24e.py \
  work_dirs/maptr_tiny_r50_24e/YOURCHECKPOINT.pth \
  --eval chamfer

```


### MapTRv2

**Training**

Run
```
cd MapTRv2_modified/
source train.sh      
```

Or by running:
```
export PYTHONPATH="${PYTHONPATH}:/MapUncertaintyPrediction/MapTRv2_modified"

python tools/train.py \
  projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py \
  --deterministic \
  --no-validate

```

**Evaluation**

Run
```
cd MapTRv2_modified/
source test.sh                                  
```

Or by running:

```
export PYTHONPATH="${PYTHONPATH}:/MapUncertaintyPrediction/MapTRv2_modified"

python tools/test.py \
  projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py \
  work_dirs/maptrv2_nusc_r50_24ep/YOURCHECKPOINT.pth \
  --eval chamfer

```
