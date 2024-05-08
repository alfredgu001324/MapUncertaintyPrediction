## Prepare Dataset

### NuScenes Download
**Make folders to store raw data and processed data**
```
mkdir nuscenes processed
cd processed
mkdir maptr maptrv2 stream_new stream
```
Download nuScenes V1.0 full dataset data and CAN bus expansion data [HERE](https://www.nuscenes.org/download). Unzip them into the `nuscenes` directory

**Download CAN bus expansion**
```
# download 'can_bus.zip'
cd nuscenes
unzip can_bus.zip 
```

**Folder Structure**
```
MapUncertaintyPrediction
├── nuscenes/
│   ├── can_bus/
│   ├── maps/
│   ├── samples/
│   ├── sweeps/
│   ├── v1.0-mini/
│   ├── v1.0-test/
|   ├── v1.0-trainval/
├── processed/
│   ├── maptr/
│   ├── maptrv2/
│   ├── stream/
│   ├── stream_new/
```

### MapTR

**Prepare Pre-trained Model**
```
cd MapTR_modified
mkdir ckpts

cd ckpts 
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
wget https://download.pytorch.org/models/resnet18-f37072fd.pth
```

**Prepare nuScenes data**

```
python tools/create_data.py nuscenes --root-path ../nuscenes --out-dir ../processed/maptr --extra-tag nuscenes --version v1.0 --canbus ../nuscenes
```

Using the above code will generate `nuscenes_infos_temporal_{train,val}.pkl`.

**Folder structure**
```
MapUncertaintyPrediction
├── nuscenes/
├── processed/
│   ├── maptr/
|   |   ├── nuscenes_infos_temporal_train.pkl
|   |   ├── nuscenes_infos_temporal_val.pkl
│   ├── maptrv2/
│   ├── stream/
│   ├── stream_new/
```

### MapTRv2

**Prepare Pre-trained Model**
```
cd MapTRv2_modified
mkdir ckpts

cd ckpts 
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
wget https://download.pytorch.org/models/resnet18-f37072fd.pth
```

**Prepare nuScenes Data**
```
python tools/maptrv2/custom_nusc_map_converter.py --root-path ../nuscenes --out-dir ../processed/maptrv2 --extra-tag nuscenes --version v1.0 --canbus ../nuscenes
```
Using the above code will generate `nuscenes_map_infos_temporal_{train,val}.pkl`, which contain local vectorized map annotations.

**Folder structure**
```
MapUncertaintyPrediction
├── nuscenes/
├── processed/
│   ├── maptr/
│   ├── maptrv2/
|   |   ├── nuscenes_map_infos_temporal_train.pkl
|   |   ├── nuscenes_map_infos_temporal_val.pkl
│   ├── stream/
│   ├── stream_new/
```

### StreamMapNet

**Prepare nuScenes Data**
```
cd StreamMapNet_modified

python tools/nuscenes_converter.py --data-root ../nuscenes --newsplit --dest_path ../processed/stream_new
python tools/nuscenes_converter.py --data-root ../nuscenes --dest_path ../processed/stream
```
Using the above code will generate `nuscenes_map_infos_{train,val}_{newsplit,}.pkl`, which contain local vectorized map annotations.

**Folder structure**
```
MapUncertaintyPrediction
├── nuscenes/
├── processed/
│   ├── maptr/
│   ├── maptrv2/
│   ├── stream/
|   |   ├── nuscenes_infos_temporal_train.pkl
|   |   ├── nuscenes_infos_temporal_val.pkl
│   ├── stream_new/
|   |   ├── nuscenes_infos_temporal_train_newsplit.pkl
|   |   ├── nuscenes_infos_temporal_val_newsplit.pkl
```
