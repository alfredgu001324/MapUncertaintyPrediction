## Map Training

### MapTRv2

Whenever training and evaluating, please edit the config paths (and checkpoint paths if testing) in the bash file. Also, edit the data paths in the config files to your nuscenes raw data and processed annotation data. 

**Training**
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
