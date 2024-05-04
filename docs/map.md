## Map Training

### MapTRv2

Whenever training and evaluating, please edit the config paths (and checkpoint paths if testing) in the bash file. Also, edit the data paths in the config files to your nuscenes raw data and processed annotation data. 

**Training**
```
cd /MapUncertaintyPrediction/MapTRv2_modified/
source train.sh                                  
```

**Evaluation**
```
cd /MapUncertaintyPrediction/MapTRv2_modified/
source test.sh                                  
```
